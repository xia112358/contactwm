"""Support qualification above native activation; never changes contact truth.

Match loaded native rows in the part frame, preserving robot/terrain shape
identity. Transport the *previous material points* with same-solve body poses;
representative-point switching is not material slip. Terrain is static here.
"""
import torch

CONTRACT=dict(version='loaded_rows_material_motion_v1',max_material_speed_m_s=.5,
    min_load_n=30.,max_relative_load_change=.25,row_match_distance_m=.01,
    rows='positive upward force in the shared native primary-surface view',
    correspondence='greedy minimum-distance bijection with robot and terrain shape identity',
    motion='maximum previous loaded material-point displacement / control dt; same-solve poses',
    qualification='two-frame pair qualification plus consecutive native active observations; raw native activation unchanged')


class NativeContactStability:
    def __init__(self,dt):
        self.dt=dt
        self.previous=None

    def update(self,view,poses,step):
        p,R=poses;shape=view.evidence.known.shape;G=view.evidence.known.numel()
        p=p.reshape(G,3);R=R.reshape(G,3,3)
        rows=view.evidence.rows;loaded=view.force_w[:,2]>0
        group=view.groups[loaded];index=view.rows[loaded];side=view.robot_side[loaded]
        force=view.force_w[loaded,2]
        order=group.argsort(stable=True);group=group[order];index=index[order];side=side[order];force=force[order]
        counts=torch.bincount(group,minlength=G);K=max(1,int(counts.max()))
        rank=torch.arange(len(group),device=p.device)-torch.repeat_interleave(counts.cumsum(0)-counts,counts)
        local=p.new_zeros(G,K,3);world=local.clone();identity=torch.full((G,K,2),-1,device=p.device,dtype=torch.long)
        valid=torch.zeros(G,K,device=p.device,dtype=torch.bool)
        points=rows['position_w'][index]
        local[group,rank]=(R[group].transpose(-1,-2)@(points-p[group])[...,None]).squeeze(-1)
        world[group,rank]=points;valid[group,rank]=True
        robot_shape=torch.where(side==0,rows['shape0'][index],rows['shape1'][index])
        terrain_shape=torch.where(side==0,rows['shape1'][index],rows['shape0'][index])
        identity[group,rank]=torch.stack((robot_shape,terrain_shape),-1)
        load=p.new_zeros(G).index_add_(0,group,force)
        stable=torch.zeros(G,device=p.device,dtype=torch.bool);row_stable=stable.clone();load_stable=stable.clone();speed=load.clone()*0
        if self.previous is not None:
            old=self.previous;J=old['local'].shape[1]
            consecutive=(step==old['step']+1)[:,None].expand(shape).reshape(-1)
            distance=(local[:,:,None]-old['local'][:,None]).norm(dim=-1)
            compatible=valid[:,:,None]&old['valid'][:,None]&((identity[:,:,None]==old['identity'][:,None]).all(-1))
            distance=distance.masked_fill(~compatible,float('inf'))
            matched=torch.zeros_like(counts)
            for _ in range(min(K,J)):
                best,ij=distance.flatten(1).min(-1);i=ij//J;j=ij%J;g=torch.arange(G,device=p.device)
                matched+=(best<=CONTRACT['row_match_distance_m'])
                distance[g,i,:]=float('inf');distance[g,:,j]=float('inf')
            row_stable=consecutive&(counts>0)&(counts==old['valid'].sum(-1))&(matched==counts)
            transported=p[:,None]+(R[:,None]@old['local'][...,None]).squeeze(-1)
            speed=((transported-old['world']).norm(dim=-1)/self.dt).masked_fill(~old['valid'],0).amax(-1)
            speed=torch.where(consecutive,speed,torch.zeros_like(speed))
            load_stable=consecutive&(load>CONTRACT['min_load_n'])&(old['load']>CONTRACT['min_load_n'])&((load-old['load']).abs()<=CONTRACT['max_relative_load_change']*torch.maximum(load,old['load']))
            stable=row_stable&load_stable&(speed<=CONTRACT['max_material_speed_m_s'])
        self.previous=dict(local=local,world=world,identity=identity,valid=valid,load=load,step=step.clone())
        return dict(contact_stable=stable.reshape(shape),contact_row_stable=row_stable.reshape(shape),
                    contact_material_speed=speed.reshape(shape),contact_load=load.reshape(shape),contact_load_stable=load_stable.reshape(shape))
