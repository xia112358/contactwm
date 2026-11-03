"""Stable-load scoring gates; native constraint activation remains unchanged."""
import torch

CONTRACT = dict(version='stable_material_contact_v1', min_load_n=30.,
    max_load_change_n=10., loaded_row_n=1., match_distance_m=.01,
    stable_frames=3, row_slots=32,
    matching='bijective nearest neighbours in part-local coordinates',
    sampling='last native solve points; post-integration body poses',
    switching='exclude from position loss; retain local material anchor')


class StableContactTracker:
    def __init__(self, batch, device):
        self.streak=torch.zeros(batch,6,dtype=torch.long,device=device)
        self.latched=torch.zeros(batch,6,dtype=torch.bool,device=device)
        self.anchor=torch.zeros(batch,6,3,device=device)
        self.previous_local=torch.zeros(batch,6,CONTRACT['row_slots'],3,device=device)
        self.previous_loaded=torch.zeros(batch,6,CONTRACT['row_slots'],dtype=torch.bool,device=device)
        self.previous_load=torch.zeros(batch,6,device=device)
        self.previous_contact=torch.zeros(batch,6,dtype=torch.bool,device=device)

    @torch.no_grad()
    def update(self,position,rotation,points,forces,mask,normal,active,known,reset):
        local=(rotation.transpose(-1,-2)[...,None,:,:]@(points-position[...,None,:])[...,None]).squeeze(-1)
        normal_force=(forces*normal[...,None,:]).sum(-1).clamp_min(0)
        loaded=mask&(normal_force>CONTRACT['loaded_row_n'])
        load=torch.where(mask,normal_force,0.).sum(-1)
        contact=active&known&~reset[:,None]
        # A permutation of rows is harmless. Each nearest assignment must be
        # unique and within tolerance; no row may appear or disappear.
        distance=torch.cdist(local,self.previous_local).masked_fill(~self.previous_loaded[...,None,:],float('inf'))
        nearest=distance.argmin(-1)
        close=distance.amin(-1)<=CONTRACT['match_distance_m']
        assigned=torch.zeros_like(loaded,dtype=torch.long)
        assigned.scatter_add_(-1,nearest,loaded.long())
        same=(loaded.sum(-1)==self.previous_loaded.sum(-1))&loaded.any(-1)
        same&=(~loaded|close).all(-1)&(assigned==self.previous_loaded.long()).all(-1)
        stable=contact&self.previous_contact&same&(load>CONTRACT['min_load_n'])
        stable&=(self.previous_load>CONTRACT['min_load_n'])&((load-self.previous_load).abs()<=CONTRACT['max_load_change_n'])
        self.streak=torch.where(stable,self.streak+1,0)
        score=stable&(self.streak>=CONTRACT['stable_frames'])
        self.latched&=contact
        take=score&~self.latched
        weight=torch.where(loaded,normal_force,0.)
        center=(local*weight[...,None]).sum(-2)/weight.sum(-1,keepdim=True).clamp_min(1e-12)
        self.anchor=torch.where(take[...,None],center,self.anchor)
        self.latched|=take
        self.previous_local.copy_(local);self.previous_loaded.copy_(loaded)
        self.previous_load.copy_(load);self.previous_contact.copy_(contact)
        return dict(stable_contact_mask=score,stable_contact_local=self.anchor.clone())
