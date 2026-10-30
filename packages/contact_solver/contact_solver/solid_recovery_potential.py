"""Diagnostic task-conditioned penetration potential; excluded from training.

The C1 gate has a verified wrong-side derivative amplification near shallow
penetration. Keep this prototype only for reproducing the diagnosis; the
training objective uses raw complete-solid distance and fixed contact skin.

This scalar changes both the forward loss and its FK derivative. It neither
changes native contact truth nor replaces physical EPA penetration reports.
Near the solid boundary it returns to physical distance with a C1 blend;
deep inside it uses a separating face compatible with the issued contact plan
or the collision-free side observed at the current pose. No future pose or
external correction solver supplies a direction.
"""
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from somaforge_core import CONTACT_BODY_NAMES_BY_PART
from somaforge_core.motion_contracts import BODY_NAMES
from somaforge_core.g1_kinematics import _matrix_from_rotation6d,_rotation6d


class SolidRecoveryPotential:
    def __init__(self,geometry,link_names):
        import trimesh
        self.link_names=tuple(link_names)
        self.robot={};self.terrain={}
        self._descendants=None
        for shape in geometry['shapes']:
            transform=np.asarray(shape['transform'],dtype=float)
            sr=Rotation.from_quat(transform[3:]).as_matrix();sp=transform[:3]
            if shape['body'] is None:
                if shape['kind']=='box':
                    pieces=[trimesh.creation.box(extents=2*np.asarray(shape['scale']))]
                elif shape['kind'] in ('mesh','convex_mesh'):
                    mesh=trimesh.Trimesh(vertices=np.asarray(shape['vertices'])*shape['scale'],faces=shape['faces'],process=False)
                    pieces=mesh.split(only_watertight=False)
                else:
                    # Curved static primitives retain the verified physical
                    # distance; they have no polyhedral face branch here.
                    continue
                for component,piece in enumerate(pieces):
                    normals=np.asarray(piece.face_normals)@sr.T
                    points=np.asarray(piece.triangles_center)@sr.T+sp
                    offsets=(normals*points).sum(-1)
                    planes=np.c_[normals,offsets]
                    _,unique=np.unique(np.round(planes,9),axis=0,return_index=True)
                    self.terrain[(shape['shape'],component)]=planes[np.sort(unique)]
            else:
                if shape['body'] not in self.link_names:raise ValueError('body missing from canonical FK')
                record=dict(shape,rotation=sr,position=sp,link=self.link_names.index(shape['body']))
                if shape['kind'] in ('mesh','convex_mesh'):
                    record['points']=(np.asarray(shape['vertices'])*shape['scale'])@sr.T+sp
                self.robot[shape['shape']]=record
        self.direct={name:p for p,key in enumerate(('left_foot','right_foot','left_hand','right_hand','left_knee','right_knee')) for name in CONTACT_BODY_NAMES_BY_PART[key]}

    def descendants(self,fk):
        if self._descendants is not None:
            return self._descendants
        parents={child:parent for parent,child,_,_ in fk.joint_records}
        kinds={child:kind for _,child,kind,_ in fk.joint_records}
        descendants=[]
        for name in self.link_names:
            attachment=name
            while kinds.get(attachment)=='fixed':attachment=parents[attachment]
            parts=[]
            for p,contact_link in enumerate(BODY_NAMES[1:7]):
                child=contact_link
                while True:
                    if child==attachment:
                        parts.append(p);break
                    if child not in parents:break
                    child=parents[child]
            descendants.append(parts)
        self._descendants=descendants
        return descendants

    def support_min(self,shape,position,rotation,normals):
        """Exact support for realized convex vertices and analytical primitives."""
        n=torch.as_tensor(normals,device=position.device,dtype=position.dtype)
        if shape['kind'] in ('mesh','convex_mesh'):
            local=torch.as_tensor(shape['points'],device=position.device,dtype=position.dtype)
            world=local@rotation.T+position
            projections=world@n.T
            values,ids=projections.min(0)
            return values,world[ids]
        center=position+rotation@torch.as_tensor(shape['position'],device=position.device,dtype=position.dtype)
        local_n=n@(rotation@torch.as_tensor(shape['rotation'],device=position.device,dtype=position.dtype))
        scale=torch.as_tensor(shape['scale'],device=position.device,dtype=position.dtype)
        kind=shape['kind']
        if kind=='sphere':
            radius=scale[0].expand(len(n));local_point=-scale[0]*local_n/local_n.norm(dim=-1,keepdim=True).clamp_min(1e-24)
        elif kind=='cylinder':
            radius=scale[0]*local_n[:,:2].norm(dim=-1)+scale[1]*local_n[:,2].abs()
            local_point=torch.cat((-scale[0]*local_n[:,:2]/local_n[:,:2].norm(dim=-1,keepdim=True).clamp_min(1e-24),
                                   -scale[1]*local_n[:,2:3].sign()),-1)
        elif kind=='capsule':
            radius=scale[0]*local_n.norm(dim=-1)+scale[1]*local_n[:,2].abs()
            local_point=-scale[0]*local_n/local_n.norm(dim=-1,keepdim=True).clamp_min(1e-24)
            local_point=local_point+torch.cat((local_n[:,:2]*0,-scale[1]*local_n[:,2:3].sign()),-1)
        elif kind=='box':radius=(scale*local_n.abs()).sum(-1);local_point=-scale*local_n.sign()
        elif kind=='ellipsoid':
            radius=(scale*local_n).norm(dim=-1);local_point=-scale.square()*local_n/radius[:,None].clamp_min(1e-24)
        else:raise ValueError('unsupported realized support '+kind)
        shape_world_rotation=rotation@torch.as_tensor(shape['rotation'],device=position.device,dtype=position.dtype)
        return center@n.T-radius,center[None]+local_point@shape_world_rotation.T

    def __call__(self,fk,rows,current_q_world,points_world,active,surface,*,sample_mask=None,audit=False):
        q=rows.q
        if active.shape != surface.shape or active.shape != points_world.shape[:2]:
            raise ValueError('Recovery plan part/point/surface shapes disagree')
        if current_q_world.shape != q.shape:
            raise ValueError('Recovery needs one observed current pose per prediction')
        required=('face_normal','face_offset','face_surface','configured_margin')
        if any(key not in rows.observed for key in required):
            raise ValueError('Recovery requires actual solver margin and surface catalog')
        if (not torch.isfinite(points_world[active]).all()
                or not torch.isfinite(rows.observed['configured_margin']).all()
                or (rows.observed['configured_margin'] <= 0).any()):
            raise ValueError('Recovery needs finite issued targets and positive actual margins')
        solid=rows.solid;p=solid.pair
        environment=p['full_kind'] == 0
        if sample_mask is not None:
            environment=environment & sample_mask[p['sample']]
        if not bool(environment.any()):
            return solid.distances, []
        descendants=self.descendants(fk)
        position,rotation=fk.link_poses(q.double(),self.link_names)
        rotation=_matrix_from_rotation6d(_rotation6d(rotation))
        if rows.world_frame is not None:
            origin,basis=rows.world_frame
            basis=_matrix_from_rotation6d(_rotation6d(basis.double()))
            position=origin[:,None].double()+torch.einsum('bij,bnj->bni',basis,position)
            rotation=basis[:,None]@rotation
        with torch.no_grad():
            initial_pos,initial_rot=fk.link_poses(current_q_world.double(),self.link_names)
            initial_rot=_matrix_from_rotation6d(_rotation6d(initial_rot))
        raw_depth=(-solid.distances).relu().double()
        margin=rows.observed['configured_margin'].double()
        recovery=[];evidence=[]
        goals=points_world.detach().double()
        for i in range(len(p['dist'])):
            if int(p['full_kind'][i])!=0 or (sample_mask is not None and not bool(sample_mask[int(p['sample'][i])])):
                recovery.append(solid.distances[i]);continue
            side=1 if int(p['body_link0'][i])<0 else 0
            sid=1-side;s=int(p['sample'][i]);link=int(p[f'body_link{side}'][i])
            shape=self.robot[int(p[f'shape{side}'][i])]
            terrain_key=(int(p[f'shape{sid}'][i]),int(p[f'component{sid}'][i]))
            if terrain_key not in self.terrain:
                recovery.append(solid.distances[i])
                if audit:
                    evidence.append(dict(row=i,sample=s,body=self.link_names[link],status='physical_curved_static_distance'))
                continue
            planes=self.terrain[terrain_key]
            normals=planes[:,:3];offsets=planes[:,3]
            support,locations=self.support_min(shape,position[s,link],rotation[s,link],normals)
            h=torch.as_tensor(offsets,device=q.device,dtype=support.dtype)
            branch=h-support
            name=self.link_names[link];direct=self.direct.get(name,-1)
            parts=([direct] if direct>=0 and bool(active[s,direct]) else
                   [part for part in descendants[link] if bool(active[s,part])])
            eligible=np.zeros(len(planes),dtype=bool)
            for part in parts:
                point=goals[s,part].cpu().numpy()
                eligible|=(normals@point-offsets)>1e-7
                # Boundary coincidence is admissible only for the actual
                # requested primary support face, not an obstacle bottom.
                obs=rows.observed
                matching=(obs['face_surface'][s]==surface[s,part]).nonzero().flatten()
                for face in matching.tolist():
                    normal=obs['face_normal'][s,face].double().cpu().numpy()
                    offset=float(obs['face_offset'][s,face])
                    eligible|=np.isclose(normals,normal,atol=1e-7,rtol=0).all(-1)&np.isclose(offsets,offset,atol=1e-6,rtol=0)
            if not parts or not eligible.any():
                # A collision-free observed shape can provide an existing
                # side of the obstacle. No GT endpoint or inferred IK pose.
                ref,_=self.support_min(shape,initial_pos[s,link],initial_rot[s,link],normals)
                eligible=(ref.detach().cpu().numpy()-offsets)>1e-7
            if not eligible.any():
                recovery.append(solid.distances[i])
                if audit:
                    evidence.append(dict(row=i,sample=s,body=name,status='physical_distance_no_known_exit_side'))
                continue
            chosen=np.flatnonzero(eligible)
            indices=torch.as_tensor(chosen,device=q.device)
            depth,j=branch[indices].min(0)
            # Use the actual solver's activation width to make a C1 transition
            # between local physical clearance and task-side recovery. This
            # changes no contact threshold and introduces no extra loss term.
            gate=raw_depth[i]
            t=(gate/margin[s]).clamp(0,1)
            blend=t*t*(3-2*t)
            value=raw_depth[i]+blend*(depth-raw_depth[i])
            recovery.append(-value.to(q))
            if audit:
                index=int(indices[j])
                evidence.append(dict(row=i,sample=s,body=name,task_parts=parts,raw_depth_m=float(raw_depth[i].detach()),
                task_depth_m=float(depth.detach()),recovery_depth_m=float(value.detach()),blend=float(blend.detach()),gate_depth_m=float(gate.detach()),
                normal=normals[index].tolist(),shape=int(p[f'shape{side}'][i]),component=int(p[f'component{sid}'][i]),
                support_point=locations[index].detach().cpu().tolist(),eligible_face_count=int(eligible.sum())))
        distance=torch.stack(recovery) if recovery else q.new_empty(0)
        return distance,evidence
