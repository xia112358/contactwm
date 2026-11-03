"""Shared dynamics/force trunk, preserving the original state integration and losses."""
import torch
from torch import nn
from . import networks


class FeatureSequential(nn.Sequential):
    """Original net.* checkpoint layout plus access to the last hidden activation."""
    def features(self, inputs):
        for index in range(len(self)-1):
            inputs=self[index](inputs)
        return inputs


def shared_force_world_model(base, context):
    class SharedForceWorldModel(base):
        contact_architecture='shared_trunk_v1'

        def __init__(self, hidden=(2048,2048,2048)):
            # Consume exactly the legacy initialization RNG so actor/critic and
            # the state network start identically in both comparison arms.
            super().__init__(hidden)
            self.net=FeatureSequential(*self.net.children())
            with torch.random.fork_rng(devices=[]):
                self.contact_head=nn.Sequential(nn.Linear(hidden[-1],context.contact_dim))
                nn.init.zeros_(self.contact_head[-1].weight)
                nn.init.zeros_(self.contact_head[-1].bias)

        def contact_logits(self,state,action,xi_n):
            inputs=self.in_norm(self.inputs(state,action,xi_n))
            return networks.run_mlp(self.contact_head,networks.run_mlp(self.net.features,inputs))

        def contact_loss(self,window,normalized_inputs=None):
            if normalized_inputs is None:
                from .mdp import unflatten_state,normalize_xi
                normalized_inputs=self.in_norm(self.inputs(unflatten_state(window['s']),window['a'],normalize_xi(window['xi'])))
            hidden=networks.run_mlp(self.net.features,normalized_inputs)
            return super().contact_loss(window,normalized_inputs=hidden)

        def joint_step(self,state,action,xi,xi_n):
            inputs=self.in_norm(self.inputs(state,action,xi_n))
            hidden=networks.run_mlp(self.net.features,inputs)
            normalized_delta=networks.run_mlp(self.net[-1],hidden)
            next_state=self.state_from_prediction(state,normalized_delta)
            prediction=networks.run_mlp(self.contact_head,hidden)
            contacts=context.world_forces(state,prediction)
            return next_state,torch.cat((xi[...,:34],contacts,xi[...,34+context.contact_dim:]),-1)

    return SharedForceWorldModel
