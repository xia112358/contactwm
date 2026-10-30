"""World-fixed surface force distributions, labelled by actual native contacts.

The query coordinates never follow a robot or a requested target. A part's
resultant force is distributed over its fixed environment patch. This first
version shares force direction within each part; it cannot represent opposing
forces on the same part. Keep all native point/force pairs for that audit.
"""
import torch
from torch import nn
from torch.nn import functional as F


def pack_native_contacts(snapshot, selected, shifts, num_envs, slots):
    """Pack eligible solver pairs without truncation or geometric relabelling."""
    count=int(snapshot['count']);device=shifts.device
    eligible=selected['eligible'][:count]
    rows=eligible.nonzero().flatten()
    groups=selected['groups'][:count][rows]
    order=groups.argsort(stable=True);rows=rows[order];groups=groups[order]
    counts=torch.bincount(groups,minlength=num_envs*6)
    if bool((counts>slots).any()):
        raise ValueError(f'Native spatial contact capacity exceeded: {int(counts.max())} > {slots}')
    rank=torch.arange(len(rows),device=device)-torch.repeat_interleave(counts.cumsum(0)-counts,counts)
    points=torch.zeros(num_envs*6,slots,3,device=device)
    forces=torch.zeros_like(points);mask=torch.zeros(num_envs*6,slots,dtype=torch.bool,device=device)
    side=selected['robot_side'][rows]
    sign=torch.where(side==1,1.,-1.)
    points[groups,rank]=snapshot['position_w'][rows]-shifts[groups//6]
    forces[groups,rank]=snapshot['force_on_body1_w'][rows]*sign[:,None]
    mask[groups,rank]=True
    return dict(spatial_points_w=points.reshape(num_envs,6,slots,3),
                spatial_forces_w=forces.reshape(num_envs,6,slots,3),spatial_mask=mask.reshape(num_envs,6,slots))


class WorldContactField(nn.Module):
    """A categorical force field on fixed external surface nodes, per body part."""
    def __init__(self, points_w, spacing=.02):
        super().__init__()
        if points_w.ndim!=3 or points_w.shape[0]!=6 or points_w.shape[-1]!=3:
            raise ValueError('Expected six world-fixed surface query patches')
        if not torch.isfinite(points_w).all():raise ValueError('Nonfinite surface queries')
        self.spacing=spacing
        self.register_buffer('points_w',points_w.clone())
        self.register_buffer('origin',points_w.mean((0,1)))
        self.register_buffer('state_scale',points_w.new_tensor([1.]*7+[2.]*6+[1.]*29+[10.]*29))
        self.encoder=nn.Sequential(nn.Linear(71,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),
                                   nn.Linear(128,6*points_w.shape[1]))
        # Initial broad distributions; training must establish their locations.
        nn.init.normal_(self.encoder[-1].weight,std=.001)
        nn.init.zeros_(self.encoder[-1].bias)
        self.feedback=nn.Sequential(nn.Linear(18,128),nn.SiLU(),nn.Linear(128,70))
        nn.init.zeros_(self.feedback[-1].weight);nn.init.zeros_(self.feedback[-1].bias)

    def forward(self,state):
        relative=torch.cat((state[:,:3]-self.origin,state[:,3:]),-1)/self.state_scale
        logits=self.encoder(relative).reshape(-1,6,self.points_w.shape[1])
        mass=logits.softmax(-1)
        point=(mass[...,None]*self.points_w).sum(-2)
        return dict(spatial_logits=logits,spatial_mass=mass,spatial_point_w=point)

    def dynamics_correction(self,state,force_w,heading):
        field=self(state)
        torque=torch.linalg.cross(field['spatial_point_w']-state[:,None,:3],force_w,dim=-1)
        torque=(heading.transpose(-1,-2)[:,None]@torque[...,None]).squeeze(-1)
        # Normalized output increments; zero initial correction preserves source WM.
        return .1*torch.tanh(self.feedback((torque/100.).flatten(1)))

    def on_nodes(self,prediction):
        """Materialize force and contact probability at fixed environment nodes."""
        mass=prediction['spatial_mass']
        return dict(position_w=self.points_w,
            force_on_robot_w=mass[...,None]*prediction['force_w'][...,None,:],
            contact_probability=mass*prediction['contact_logits'].sigmoid()[...,None])

    def targets(self,points,forces,mask,known,valid):
        """Soft splat every measured contact independently onto external nodes.

        Force magnitude weights spatial mass. Zero-force active constraints still
        have position evidence and receive uniform weights. No force threshold
        defines activation. Gaussian splats allow sub-grid means.
        """
        mask=mask&known[...,None]&valid[...,None,None]
        safe_points=torch.where(mask[...,None],points,0.)
        safe_forces=torch.where(mask[...,None],forces,0.)
        d2=(safe_points[...,None,:]-self.points_w[:,None]).square().sum(-1)
        nearest=d2.amin(-1)
        inside=~mask|(nearest<(2*self.spacing)**2)
        kernel=torch.softmax(-d2/(2*(self.spacing*.5)**2),-1)*mask[...,None]
        weight=safe_forces.norm(dim=-1)
        weight=torch.where(weight.sum(-1,keepdim=True)>0,weight,mask.float())
        weight=weight/weight.sum(-1,keepdim=True).clamp_min(1e-12)
        target_mass=(kernel*weight[...,None]).sum(-2)
        target_force=(kernel[...,None]*safe_forces[...,None,:]).sum(-3)
        observed=mask.any(-1)
        return target_mass,target_force,observed,inside

    def fit_loss(self,prediction,target_mass,target_force,observed):
        logits=prediction['spatial_logits'];mass=prediction['spatial_mass']
        log_target=target_mass.clamp_min(1e-12).log()
        kl=(target_mass*(log_target-F.log_softmax(logits,-1))).sum(-1)
        predicted_point=prediction['spatial_point_w']
        actual_point=(target_mass[...,None]*self.points_w).sum(-2)
        position=((predicted_point-actual_point)/.02).square().sum(-1)
        predicted_force=mass[...,None]*prediction['force_w'][...,None,:]
        field_force=((predicted_force-target_force)/300.).square().sum((-2,-1))
        def mean(x):return torch.where(observed,x,0.).sum()/observed.sum().clamp_min(1)
        return mean(kl)+mean(position)+mean(field_force)

    def expected_target_cost(self,state,target,normal,tangent_scale):
        """Expected spatial error, so two wrong peaks cannot cancel at a goal."""
        field=self(state);error=self.points_w[None]-target[:,:,None]
        dn=(error*normal[:,:,None]).sum(-1)
        dt=error-dn[...,None]*normal[:,:,None]
        cost=(dn/.02).square()+dt.square().sum(-1)/tangent_scale**2
        return (field['spatial_mass']*cost).sum(-1),field['spatial_point_w']
