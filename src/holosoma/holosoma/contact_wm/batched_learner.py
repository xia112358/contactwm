"""Tensor-only batched WM/BPTT kernels; synchronization stays at update boundaries."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from holosoma.contact_wm.observations import rotation_xyzw,yaw_local,unpack_history,pack_history
from holosoma.contact_wm.target_response import target_grid


def native_fit_tensor(prediction,target,active,force,known,valid,balanced_contacts=False):
    state=prediction['state']
    sign=torch.where((state[...,3:7]*target[...,3:7]).sum(-1,keepdim=True)>=0,1.,-1.)
    error=torch.cat((state[...,:3]-target[...,:3],state[...,3:7]-sign*target[...,3:7],state[...,7:]-target[...,7:]),-1)
    scale=error.new_tensor([.02]*7+[.2]*6+[.05]*29+[1.]*29)
    state_loss=torch.where(valid,(error/scale).square().mean(-1),0.).sum()/valid.sum().clamp_min(1)
    mask=known.bool()&valid[...,None]
    bce=F.binary_cross_entropy_with_logits(prediction['contact_logits'],active.float(),reduction='none')
    safe_force=torch.where(mask[...,None],force,0.)
    ferr=((prediction['force_w']-safe_force)/300.).square().mean(-1)
    if balanced_contacts:
        # Equal positive/negative contribution for each observed part. Labels
        # remain actual native activations, never desired target contacts.
        pos=mask&active.bool();neg=mask&~active.bool()
        np,nm=pos.sum(0),neg.sum(0)
        positive=torch.where(pos,bce,0.).sum(0)/np.clamp_min(1)
        negative=torch.where(neg,bce,0.).sum(0)/nm.clamp_min(1)
        classes=(np>0).float()+(nm>0).float()
        classification=((positive+negative)/classes.clamp_min(1)).sum()/(classes>0).sum().clamp_min(1)
        return state_loss+classification+torch.where(mask,ferr,0.).sum()/mask.sum().clamp_min(1)
    return state_loss+torch.where(mask,bce+ferr,0.).sum()/mask.sum().clamp_min(1)


class BatchedLearner(nn.Module):
    names=('loss','geometry','activation','load','slip','regularization','motion','stability','target_geometry','support_geometry')
    def __init__(self,task,atlas,*,compile_kernels=True,wm_window=8,horizon=35,hand_surface=False):
        super().__init__()
        self.use_panorama=hasattr(task['policy'],'encoder')
        self.policy=task['policy'];self.prior=task['prior'];self.wm=task['wm'];self.geometry=task['geometry']
        device=next(self.policy.parameters()).device
        self.wm_window=wm_window;self.horizon=horizon
        self.target_tangent_scale=task.get('target_tangent_scale',.05)
        tensor=lambda x:torch.as_tensor(x,device=device,dtype=torch.float32)
        a=np.load(atlas)
        self.register_buffer('raster',tensor(a['height'][:1])[:,None]);self.register_buffer('minimum',tensor(a['minimum'][:1]))
        self.resolution=float(a['resolution'])
        yy,xx=torch.meshgrid(torch.linspace(-.8,.8,11),torch.linspace(-.5,1.5,11),indexing='ij')
        self.register_buffer('source_grid',tensor(torch.stack((xx.flatten(),yy.flatten()),-1)))
        xx,yy=torch.meshgrid(torch.linspace(-.4,.4,9),torch.linspace(-.3,.3,7),indexing='ij')
        self.register_buffer('php_grid',tensor(torch.stack((xx.flatten(),yy.flatten()),-1)))
        self.register_buffer('default_q',tensor(task['meta']['default_joint_pos']))
        self.register_buffer('hint',task['hint'][:1].detach().clone())
        for name in ('normal_w','establish','maintain','min_normal_force'):
            self.register_buffer(name,getattr(task['goal'],name)[:1].clone())
        self.register_buffer('target',task['base_target'][:1].clone())
        self.register_buffer('offsets',target_grid(task.get('target_span',.04),device))
        self.hand_surfaces=nn.ModuleDict()
        if hand_surface:
            from holosoma.contact_wm.surface_geometry import HandPlaneContact
            parts=self.establish[0].nonzero().flatten().tolist()
            if not parts or any(p not in (2,3) for p in parts):raise ValueError('Surface placement currently supports new hand contacts only')
            for part in parts:
                if not torch.allclose(self.normal_w[0,part],tensor([0.,0.,1.])):
                    raise ValueError('Height-atlas plane lookup requires upward horizontal targets')
                self.hand_surfaces[str(part)]=HandPlaneContact(task['meta']['joint_names'],'left' if part==2 else 'right',
                    collision_profile=task['meta']['robot_asset']['collision_profile']).to(device)
                with torch.no_grad():
                    self.hand_surfaces[str(part)](tensor(task['z']['state'][266:267]),self.normal_w[:,part],self.target[:,part])
        # Cache canonical FK traversal before tracing; no dynamic cache mutation in kernels.
        self.geometry(tensor(task['z']['state'][266:267]))
        self.wm_rollout=torch.compile(self._wm_rollout,fullgraph=True) if compile_kernels else self._wm_rollout
        self.imag_step=torch.compile(self._imag_step,fullgraph=True) if compile_kernels else self._imag_step
        self.objective=torch.compile(self._objective,fullgraph=True) if compile_kernels else self._objective
    def height(self,state,local,php=False):
        R=rotation_xyzw(state[:,3:7]);d=R[:,:2,0];d=d/(d.square().sum(-1,keepdim=True)+1e-9).sqrt()
        c,s=d.unbind(-1);x,y=local.unbind(-1)
        world=torch.stack((c[:,None]*x-s[:,None]*y,s[:,None]*x+c[:,None]*y),-1)+state[:,None,:2]
        indices=(world-self.minimum[:,None])/self.resolution
        limits=state.new_tensor([self.raster.shape[-1]-1,self.raster.shape[-2]-1])
        valid=((indices>=0)&(indices<=limits)).all((1,2))
        grid=2*indices/limits-1
        h=F.grid_sample(self.raster.expand(len(state),-1,-1,-1),grid[:,None],align_corners=True).squeeze(1).squeeze(1)
        valid=valid&torch.isfinite(h).all(-1)
        relative=h-state[:,2:3]
        return (relative if php else relative.clamp(-2,2)),valid
    def physics(self,state):
        return torch.cat((state.new_full((len(state),1),(1.-.75)/.45),state.new_zeros(len(state),32)),-1)
    def _wm_rollout(self,x):
        state=x['state'][0];force=x['previous_force_w'][0];loss=state.new_zeros(())
        physics=self.physics(state)
        for t in range(self.wm_window):
            p=self.wm.source_step(state,x['action'][t],physics,x['height'][t],force)
            loss=loss+native_fit_tensor(p,x['next_state'][t],x['active'][t],x['force_w'][t],x['known'][t],x['valid'][t],balanced_contacts=True)/self.wm_window
            if hasattr(self.wm,'world_field'):
                if 'spatial_target_mass' in x:
                    target=(x['spatial_target_mass'][t].float(),x['spatial_target_force'][t].float(),x['spatial_observed'][t])
                else:
                    target=self.wm.world_field.targets(x['spatial_points_w'][t],x['spatial_forces_w'][t],
                        x['spatial_mask'][t],x['known'][t],x['valid'][t])
                loss=loss+self.wm.world_field.fit_loss(p,*target[:3])/self.wm_window
            state=p['state'];force=p['force_w']
        return loss
    def _imag_step(self,state,contact,history,previous,force,known,targets,support,remaining,initial,features,view,first,native=None):
        # Finished shorter horizons use a safe initial pose; masked padding cannot
        # extrapolate the map and introduce NaN gradients into active samples.
        state=torch.where((remaining>=-8)[:,None],state,initial)
        R=rotation_xyzw(state[:,3:7]);angular=(R.transpose(-1,-2)@state[:,10:13,None]).squeeze(-1)
        proprio=torch.cat((previous,angular,state[:,13:42]-self.default_q,state[:,42:71],-R[:,2,:]),-1)
        history=torch.cat((history[:,:-1] if first else history[:,1:],proprio[:,None]),1)
        active=self.establish|self.maintain
        target=torch.where(self.maintain[...,None],support,targets)
        local=yaw_local(target-state[:,None,:3],state[:,3:7])
        command=torch.cat((torch.where(active[...,None],local,0.).flatten(1),active.expand(len(state),-1).float()),-1)
        php_height,ok_php=self.height(state,self.php_grid,True)
        obs=torch.cat((contact,known.float(),command,self.hint.expand(len(state),-1),php_height,pack_history(history)),-1)
        normals=yaw_local(self.normal_w.expand(len(state),-1,-1),state[:,3:7])
        extra=torch.cat((normals.flatten(1),self.maintain.expand(len(state),-1).float(),remaining[:,None].clamp_min(0)*.02),-1)
        if native is not None:
            from supermimic.contact_dmo import swap
            obs=swap(obs,native['policy_observation'])
            extra=swap(extra,native['policy_extra'])
        if self.use_panorama:
            from holosoma.contact_wm.panorama_policy import relative_view
            action=self.policy.forward_encoded(obs,extra,features,relative_view(state,view))
        else:action=self.policy(obs,extra)
        if native is not None:
            native['action_error']=(action.detach()-native['action']).abs().masked_fill(~native['dmo_live'][:,None],0.).max()
            action=swap(action,native['action'])
        with torch.no_grad():prior=self.prior(obs)
        height,ok_wm=self.height(state,self.source_grid)
        if native is not None:height=swap(height,native['height'])
        p=self.wm.source_step(state,action,self.physics(state),height,force)
        return p['state'],p['contact'],history,action,p['force_w'],p['contact_logits'],prior,ok_php&ok_wm
    def prepare(self,sample):
        expand=lambda x:x.repeat_interleave(9,0)
        state=expand(sample['state']);history=unpack_history(expand(sample['observation'][:,102:]))
        target=self.target.expand(len(state),-1,-1)+self.establish[...,None]*self.offsets.repeat(len(sample['state']),1)[:,None]
        result=dict(state=state,history=history,previous=history[:,-1,:29],contact=expand(sample['observation'][:,:6]),
            known=expand(sample['previous_known']),force=expand(sample['previous_force_w']),target=target,
            support=expand(sample['support']),arrival=26-(expand(sample['age'])-5))
        if self.use_panorama:
            result.update(features=expand(self.policy.encode(sample['panorama'])),view=expand(sample['pano_view']))
        else:result.update(features=state.new_zeros(len(state),63),view=state.new_zeros(len(state),4))
        return result
    def policy_loss(self,sample,compiled=True):
        x=self.prepare(sample)
        state,contact,history,previous,force=x['state'],x['contact'],x['history'],x['previous'],x['force']
        states=[];actions=[];forces=[];logits=[];priors=[];valid=[]
        # Model parameters are frozen by the caller for both forward and backward.
        # State/action derivatives remain intact through every imagined time step.
        step_fn=self.imag_step if compiled else self._imag_step
        for step in range(self.horizon):
            state,contact,history,previous,force,logit,prior,ok=step_fn(state,contact,history,previous,force,
                x['known'],x['target'],x['support'],x['arrival']-step,x['state'],x['features'],x['view'],step==0)
            states.append(state);actions.append(previous);forces.append(force);logits.append(logit);priors.append(prior);valid.append(ok)
        tensors=tuple(torch.stack(v) for v in (states,actions,forces,logits,priors,valid))
        return (self.objective if compiled else self._objective)(x,*tensors)
    def _objective(self,x,states,actions,forces,logits,priors,valid,native=None):
        if native is not None and hasattr(self,'segment_surface'):
            from holosoma.contact_wm.single_segment import objective
            return objective(self,x,states,actions,priors,valid,native)
        H,B=states.shape[:2]
        points=self.geometry(states.flatten(0,1)).reshape(H,B,6,3)
        placement=points
        if self.hand_surfaces:
            placement=points.clone()
            for part_name,surface in self.hand_surfaces.items():
                part=int(part_name);plane=x['target'][:,part].clone()
                indices=(plane[:,:2]-self.minimum)/self.resolution
                limits=plane.new_tensor([self.raster.shape[-1]-1,self.raster.shape[-2]-1])
                grid=(2*indices/limits-1).reshape(1,1,B,2)
                plane[:,2]=F.grid_sample(self.raster,grid,align_corners=True).reshape(B)
                valid=valid&((indices>=0)&(indices<=limits)).all(-1)[None]&torch.isfinite(plane).all(-1)[None]
                placement[:,:,part]=surface(states.flatten(0,1),self.normal_w[:,part].expand(H*B,-1),plane[None].expand(H,-1,-1).reshape(H*B,3)).reshape(H,B,3)
        initial_points=self.geometry(x['state'])
        if native is not None:
            from supermimic.contact_dmo import swap
            observed=native['known']&native['active']
            placement=swap(placement,torch.where(observed[...,None],native['point'],placement.detach()))
        t=torch.arange(1,H+1,device=states.device)[:,None]
        hold=(t>=x['arrival'])&(t<x['arrival']+10)
        live=t<x['arrival']+10
        if native is not None:live=live&native['live']
        required=(self.maintain[None]|(hold[...,None]&self.establish[None]))&live[...,None]
        mean=lambda value,mask:(torch.where(mask,value,0.).sum(0)/mask.sum(0).clamp_min(1)).mean()
        # Average each independent start/goal equally, including different remaining horizons.
        def contact_mean(value):
            return (torch.where(required,value,0.).sum((0,2))/required.sum((0,2)).clamp_min(1)).mean()
        target=torch.where(self.maintain[...,None],x['support'],x['target'])
        error=placement-target[None];normal=(error*self.normal_w[None]).sum(-1)
        tangent=error-normal[...,None]*self.normal_w[None]
        from holosoma.contact_wm.core import contact_geometry
        target_geometry,support_geometry=contact_geometry(normal,tangent,
            hold[...,None]&self.establish[None]&live[...,None],
            self.maintain[None]&live[...,None],self.target_tangent_scale)
        if hasattr(self.wm,'world_field'):
            field_cost,field_points=self.wm.world_field.expected_target_cost(states.flatten(0,1),
                target[None].expand(H,-1,-1,-1).reshape(H*B,6,3),
                self.normal_w.expand(H*B,-1,-1),self.target_tangent_scale)
            field_cost=field_cost.reshape(H,B,6)
            mask=hold[...,None]&self.establish[None]&live[...,None]
            target_geometry=(torch.where(mask,field_cost,0.).sum((0,2))/mask.sum((0,2)).clamp_min(1)).mean()
            placement=torch.where(self.establish[None,...,None],field_points.reshape(H,B,6,3),placement)
        geometry=target_geometry+support_geometry
        activation=contact_mean(F.softplus(-logits))
        fn=(forces*self.normal_w[None]).sum(-1)
        load=contact_mean((F.relu(self.min_normal_force[None]-fn)/100.).square())
        velocity=(points-torch.cat((initial_points[None],points[:-1]),0))/.02
        vn=(velocity*self.normal_w[None]).sum(-1);vt=velocity-vn[...,None]*self.normal_w[None]
        slip=contact_mean(vt.square().sum(-1)+.1*vn.square())
        previous=torch.cat((x['previous'][None],actions[:-1]),0)
        regularization=mean((actions-previous).square().mean(-1)+.01*actions.square().mean(-1),live)
        motion=mean((actions-priors).square().mean(-1),live)
        up=rotation_xyzw(states[...,3:7])[...,2,2];initial_up=rotation_xyzw(x['state'][:,3:7])[:,2,2]
        stable=100*torch.relu(torch.maximum(initial_up-.05,initial_up.new_tensor(.3))-up).square()+10*torch.relu(x['state'][:,2]-.12-states[:,:,2]).square()
        stability=mean(stable,live)
        endpoints=(placement[:,:,2]*hold[...,None]).sum(0)/hold.sum(0)[:,None].clamp_min(1)
        loss=geometry+activation+load+.1*slip+.01*regularization+.01*motion+stability
        values=torch.stack((loss,geometry,activation,load,slip,regularization,motion,stability,target_geometry,support_geometry))
        ok=(valid|~live).all()&torch.isfinite(values).all()
        if self.use_panorama:return values,ok,endpoints
        return values,ok
