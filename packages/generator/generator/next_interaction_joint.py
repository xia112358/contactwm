"""Observation-only contact-conditioned interaction; no pose correction stage.

Surface proximity is a differentiable compatibility proxy, never Newton truth.
Unobserved solver part/face margins are explicitly excluded and reported.
"""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from generator.next_interaction import NextInteractionPredictor, NextInteraction, objective as reference_objective
from contact_solver.surface_geometry import nearest_face


class JointInteractionPredictor(NextInteractionPredictor):
    def __init__(self,width=192,layers=3):
        super().__init__(width,layers)
        self.role_embedding=nn.Linear(4,width,bias=False)
        self.interaction_decoder=nn.TransformerEncoder(
            nn.TransformerEncoderLayer(width,6,2*width,dropout=0.,activation='gelu',
                                       batch_first=True,norm_first=True),1,enable_nested_tensor=False)

    def decode_interaction(self,body,part,face,current_q,faces):
        roles=self.role_head(part)
        surfaces=torch.einsum('bpw,bsw->bps',self.surface_query(part),self.surface_key(face))/part.shape[-1]**.5
        # Predictions, never teacher labels: identical train/inference path.
        contact_tokens=(part+self.role_embedding(roles.softmax(-1))
                        +torch.einsum('bps,bsw->bpw',surfaces.softmax(-1),face))
        interaction=self.interaction_decoder(torch.cat((body[:,None],contact_tokens),1))
        shared=self.norm(interaction[:,0]+interaction[:,1:].mean(1))
        return NextInteraction(self.decode_pose(self.pose_head(shared),current_q),roles,surfaces,
                               F.softplus(self.duration_head(shared)[:,0]))


def compatibility_cost(points,parts,prediction,faces,margin,known):
    """Selected intent only, no averaging contradictory surfaces or changing labels.

    The discrete intent is detached: this cost moves geometry, not its contact
    flag. All flags/surfaces still receive the unchanged Newton-label CE loss.
    """
    from contact_solver.surface_contact_proxy import primary_face_cost
    per_cost,distance,per_excess,per_edge=primary_face_cost(points,parts,faces,margin)
    selected=prediction.surface.detach()[...,None]
    gap=distance.gather(2,selected)[...,0]
    available=known.gather(2,selected)[...,0]
    intended=prediction.contact.detach()
    used=intended & available
    residual=per_excess.gather(2,selected)[...,0]*used
    cost=per_cost.gather(2,selected)[...,0]*used/.02**2
    n=used.sum(-1).clamp_min(1)
    return cost.sum(-1)/n+cost.amax(-1),dict(
        intent_geometry_gap_cm=100*(gap*used).sum(-1)/n,
        intent_margin_excess_cm=100*residual.amax(-1),
        intent_face_ownership_excess_cm=100*(per_edge.gather(2,selected)[...,0]*used).amax(-1),
        intent_margin_unknown=(intended & ~available).sum(-1).float())


def objective(model,prediction,target,scene):
    loss,metrics=reference_objective(model,prediction,target,scene)
    cloud,parts=model.geometry(model.fk,prediction.qpos[:,None])
    consistency,extra=compatibility_cost(cloud[:,0],parts,prediction,target['observed_faces'],
                                       target['solver_margin'],target['solver_margin_known'])
    loss=loss+consistency
    metrics.update(extra);metrics['consistency_loss']=consistency;metrics['loss']=loss
    return loss,metrics


def prepare_supervision(manifest,inputs,target,samples):
    """Read actual solver margins for loss only; no change to inference inputs.

    Only observed part/face pairs with uniform margins may use a part-level
    proxy. Missing/heterogeneous shape margins remain unknown, not guessed.
    """
    entries=json.loads(Path(manifest).read_text())['motion_files']
    margins=[];known=[];missing=0;heterogeneous=0
    for entry in entries:
        path=Path(entry['newton_contact_file'])
        if not path.is_absolute():path=Path(manifest).parent/path
        with np.load(path,allow_pickle=False) as z:
            if 'task_contact_pairs_json' not in z:raise ValueError(f'Missing solver task pairs: {path}')
            pairs=json.loads(z['task_contact_pairs_json'].item())
        values=[[set() for _ in range(2)] for _ in range(6)]
        for frame in pairs:
            for pair in frame:
                if not pair['allocated']:raise ValueError(f'Unallocated active contact: {path}')
                value=float(pair['includemargin'])
                if not np.isfinite(value) or not float(pair['dist'])<value:
                    raise ValueError(f'Invalid solver active pair: {path}')
                p,s=int(pair['part']),int(pair['surface'])
                if not (0<=p<6 and 0<=s<2):raise ValueError('Unexpected effective part/face mapping')
                values[p][s].add(value)
        a=np.zeros((6,2),np.float32);valid=np.zeros((6,2),bool)
        for p in range(6):
            for s in range(2):
                if len(values[p][s])==1:a[p,s]=next(iter(values[p][s]));valid[p,s]=True
                elif not values[p][s]:missing+=1
                else:heterogeneous+=1
        margins.append(a);known.append(valid)
    ids=np.array([s.motion_id for s in samples])
    device=inputs['current_q'].device
    target['observed_faces']=inputs['faces']
    target['solver_margin']=torch.tensor(np.array(margins)[ids],device=device)
    target['solver_margin_known']=torch.tensor(np.array(known)[ids],device=device)
    return dict(schema='joint_interaction_v2',geometry_proxy_not_contact_truth=True,
                contact_proxy='legacy_surface_approach_v1',
                margin_source='actual allocated Newton task pairs, uniform per part/face',
                unknown_motion_part_faces=missing,heterogeneous_motion_part_faces=heterogeneous,
                predicted_topology_conditions_pose_and_duration=True)
