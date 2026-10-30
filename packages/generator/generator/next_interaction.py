"""One-pass pose/contact predictor with unbounded joint and translation outputs.

Forward accepts observations only. Demonstration material points exist exclusively
in the objective; no learned free offset or pose projection can hide FK error.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F
from somaforge_core.g1_kinematics import CanonicalG1ForwardKinematics, _quaternion_matrix_wxyz, _matrix_from_rotation6d
from contact_solver.collision_geometry import CanonicalG1CollisionPoints, full_geometry_box_ground_penetration


from contact_solver.surface_geometry import joint_feasibility_loss
from generator.endpoint_objective import (POSE_POSITION_SCALE_M,
    POSE_ROTATION_MATRIX_SCALE, POSE_JOINT_SCALE_RAD)


def structural_pose_loss(model, q, position, rotation, target, *, joint_max_weight=0.):
    """Keep every limb near its demonstrated endpoint without averaging it away.

    This is label-only supervision.  It does not alter predictor inputs, define
    contact, or project an output pose.  Newton contact terms remain separate.
    """
    with torch.no_grad():
        _, target_rotation6d = model.fk(target['q'][:, None])
        target_rotation = _matrix_from_rotation6d(target_rotation6d[:, 0])
    # Exclude the root body here: root translation/orientation is already in
    # the global pose term.  mean+max preserves ordinary whole-body fitting
    # while preventing one free limb from disappearing among six bodies.
    body_position_error = ((position[:, 1:7] - target['body_position'][:, 1:7]) / POSE_POSITION_SCALE_M).square().sum(-1)
    body_rotation_error = ((rotation[:, 1:7] - target_rotation[:, 1:7]) / POSE_ROTATION_MATRIX_SCALE).square().mean((-1, -2))
    joint_error = ((q[:, 7:] - target['q'][:, 7:]) / POSE_JOINT_SCALE_RAD).square()
    value = (body_position_error.mean(-1) + body_position_error.amax(-1)
             + body_rotation_error.mean(-1) + body_rotation_error.amax(-1))
    # New endpoint training supervises joints once in the pose term. The
    # explicit opt-in only reproduces historical non-endpoint objectives.
    if joint_max_weight:
        value = value + joint_max_weight*joint_error.amax(-1)
    return value, dict(
        body_position_max_cm=100 * (position[:, 1:7] - target['body_position'][:, 1:7]).norm(dim=-1).amax(-1),
        body_orientation_max=body_rotation_error.sqrt().amax(-1),
        joint_error_max_rad=(q[:, 7:] - target['q'][:, 7:]).abs().amax(-1),
    )


from contact_solver.surface_geometry import nearest_face


@dataclass
class NextInteraction:
    qpos: torch.Tensor
    role_logits: torch.Tensor
    surface_logits: torch.Tensor
    duration: torch.Tensor

    @property
    def role(self): return self.role_logits.argmax(-1)
    @property
    def contact(self): return self.role != 0
    @property
    def touchdown(self): return self.role == 1
    @property
    def persistent(self): return self.role == 2
    @property
    def surface(self): return self.surface_logits.argmax(-1)


class NextInteractionPredictor(nn.Module):
    def __init__(self,width=192,layers=3):
        super().__init__()
        self.fk=CanonicalG1ForwardKinematics()
        self.geometry=CanonicalG1CollisionPoints(64)
        self.global_encoder=nn.Linear(38,width)
        self.part_encoder=nn.Linear(13,width)
        self.face_encoder=nn.Linear(15,width)
        self.relation_encoder=nn.Sequential(nn.Linear(8,width),nn.GELU(),nn.Linear(width,width))
        self.part_identity=nn.Embedding(6,width)
        layer=nn.TransformerEncoderLayer(width,6,2*width,dropout=0.,activation='gelu',batch_first=True,norm_first=True)
        self.shared=nn.TransformerEncoder(layer,layers,enable_nested_tensor=False)
        self.norm=nn.LayerNorm(width)
        self.pose_head=nn.Linear(width,36)
        self.role_head=nn.Linear(width,4)
        self.surface_query=nn.Linear(width,width,bias=False)
        self.surface_key=nn.Linear(width,width,bias=False)
        self.duration_head=nn.Linear(width,1)
        nn.init.normal_(self.pose_head.weight,std=.001)
        nn.init.zeros_(self.pose_head.bias)
        with torch.no_grad():self.pose_head.bias[3]=1.

    def decode_pose(self,raw,current_q):
        # Only quaternion normalization defines the rotation representation.
        # No delta limit, tanh, prototype, joint clipping or corrective solver.
        quat=F.normalize(raw[:,3:7],dim=-1)
        return torch.cat((current_q[:,:3]+raw[:,:3],quat,raw[:,7:]),-1)

    def forward(self,current_q,current_contact,current_anchor,current_surface,faces):
        B=len(current_q);S=faces.shape[1]
        if faces.shape[-1]!=15 or S<1:raise ValueError('Missing actual face geometry')
        positions,rot6=self.fk(current_q[:,None]);p=positions[:,0,1:7];r6=rot6[:,0,1:7]
        rotation=_quaternion_matrix_wxyz(current_q[:,3:7])
        state=torch.cat((current_q[:,:3],rotation[:,:,:2].flatten(1),current_q[:,7:]/math.pi),-1)
        global_token=self.global_encoder(state)[:,None]
        part=self.part_encoder(torch.cat((p,r6,current_anchor*current_contact[...,None],current_contact[...,None].float()),-1))
        part=part+self.part_identity.weight[None]
        face=self.face_encoder(faces)
        nearest,signed=nearest_face(p,faces)
        delta=nearest-p[:,:,None]
        local_normal=torch.einsum('bpji,bsj->bpsi',_matrix_from_rotation6d(r6),faces[:,:,3:6])
        observed=(current_surface[:,:,None]==torch.arange(S,device=p.device)[None,None]) & current_contact[:,:,None].bool()
        relation=self.relation_encoder(torch.cat((delta,local_normal,signed[...,None],observed[...,None].float()),-1))
        pair=(part[:,:,None]+face[:,None]+relation).flatten(1,2)
        tokens=self.shared(torch.cat((global_token,part,face,pair),1))
        body=self.norm(tokens[:,0]+tokens[:,1:7].mean(1))
        part=self.norm(tokens[:,1:7]);face=self.norm(tokens[:,7:7+S])
        return self.decode_interaction(body,part,face,current_q,faces)

    def decode_interaction(self,body,part,face,current_q,faces):
        raw=self.pose_head(body)
        return NextInteraction(self.decode_pose(raw,current_q),self.role_head(part),
            torch.einsum('bpw,bsw->bps',self.surface_query(part),self.surface_key(face))/math.sqrt(part.shape[-1]),
            F.softplus(self.duration_head(body)[:,0]))

    def geometry_points(self,prediction,faces):
        """Real robot sample nearest intended face; diagnostic, not Newton contact."""
        points,parts=self.geometry(self.fk,prediction.qpos[:,None]);points=points[:,0]
        closest,_=nearest_face(points,faces)
        distances=(points[:,:,None]-closest).square().sum(-1)
        chosen=[]
        for part in range(6):
            score=distances.gather(2,prediction.surface[:,part,None,None].expand(-1,len(parts),1))[:,:,0]
            score=score.masked_fill(parts[None]!=part,float('inf'))
            index=score.argmin(-1)
            chosen.append(points[torch.arange(len(points),device=points.device),index])
        return torch.stack(chosen,1)


def objective(model,prediction,target,scene,*,contact_override=None,additional_penetration=None):
    """All future quantities are labels only, never inputs to model.forward."""
    q=prediction.qpos
    pos,rot6=model.fk(q);rot=_matrix_from_rotation6d(rot6)
    material=pos[:,1:7]+torch.einsum('bpij,bpj->bpi',rot[:,1:7],target['material_local'])
    error=(material-target['anchor']).norm(dim=-1)
    active=target['role']!=0
    count=active.sum(-1).clamp_min(1)
    normalized=(error/.02).square()*active
    contact=normalized.sum(-1)/count+.25*normalized.amax(-1)
    if contact_override is not None:contact=contact_override
    role=F.cross_entropy(prediction.role_logits.transpose(1,2),target['role'],reduction='none').mean(-1)
    surface=F.cross_entropy(prediction.surface_logits.transpose(1,2),target['surface'].clamp_min(0),reduction='none')
    surface=(surface*active).sum(-1)/count
    duration=(prediction.duration.log()-target['duration'].log()).square()
    rotation_error=(_quaternion_matrix_wxyz(q[:,3:7])-_quaternion_matrix_wxyz(target['q'][:,3:7])).square().mean((1,2))
    pose=((q[:,:3]-target['q'][:,:3])/.1).square().mean(-1)+rotation_error/.3**2+((q[:,7:]-target['q'][:,7:])/.5).square().mean(-1)
    structure,structure_metrics=structural_pose_loss(model,q,pos,rot,target,joint_max_weight=1.)
    points,parts=model.geometry(model.fk,q)
    box,ground=full_geometry_box_ground_penetration(points[:,None],parts,q.new_zeros(len(q),1,6),**scene,
                                                   margin_m=0.,planned_contact_tolerance_m=0.)
    depth=torch.maximum(box,ground)[:,0]
    if additional_penetration is not None:
        depth=torch.cat((depth,additional_penetration[:,None]),dim=1)
    collision=(depth/.02).square().mean(-1)+(depth/.02).square().amax(-1)
    joint_limit,violation=joint_feasibility_loss(q[:,7:],model.fk.joint_lower,model.fk.joint_upper)
    loss=5*contact+role+surface+duration+.2*pose+.1*structure+collision+joint_limit
    metrics=dict(loss=loss,contact_cm=100*(error*active).sum(-1)/count,
                 contact_max_cm=100*(error*active).amax(-1),
                 role_exact=(prediction.role==target['role']).all(-1).float(),
                 touchdown_exact=(prediction.touchdown==(target['role']==1)).all(-1).float(),
                 contact_exact=(prediction.contact==active).all(-1).float(),
                 root_cm=100*(q[:,:3]-target['q'][:,:3]).norm(dim=-1),
                 joint_rmse_rad=(q[:,7:]-target['q'][:,7:]).square().mean(-1).sqrt(),
                 body_cm=100*(pos-target['body_position']).norm(dim=-1).mean(-1),
                 structure_loss=structure,
                 penetration_cm=100*depth.amax(-1),joint_violation_rad=violation.amax(-1),
                 duration_mae=(prediction.duration-target['duration']).abs())
    metrics.update(structure_metrics)
    return loss,metrics
