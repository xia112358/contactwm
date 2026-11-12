"""Strict reuse of SuperMimic force6 dynamics, without importing its policy.

The source uses manifold increments, not the diagnostic residual WM. Do not
reshape its weights into another architecture. PHP and source action semantics
must be explicitly reconciled before a live policy optimization experiment.
"""
import importlib.util
import sys
import json
from pathlib import Path
import torch
from torch import nn
from holosoma.contact_wm.core import PARTS
from holosoma.contact_wm.observations import rotation_xyzw


class RunningNorm(nn.Module):
    def __init__(self,n,eps):
        super().__init__();self.eps=eps
        self.register_buffer('mean',torch.zeros(n));self.register_buffer('var',torch.ones(n));self.register_buffer('count',torch.zeros(()))
    def std(self):return torch.sqrt(self.var+self.eps**2)
    def forward(self,x):return (x-self.mean)/self.std()


def mlp(widths,layer_norm=False):
    layers=[]
    for i,(a,b) in enumerate(zip(widths[:-1],widths[1:])):
        layers.append(nn.Linear(a,b))
        if i<len(widths)-2:
            if layer_norm:layers.append(nn.LayerNorm(b))
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


class SuperMimicForceWM(nn.Module):
    """Original 272 -> 70 dynamics, 272 -> 18 heading-force prediction.

    Native states here use xyzw; the source quaternion helpers use wxyz.
    Existing force head and both normalizers are inherited exactly. The new
    six-way activation head is separate and must be fitted to native labels.
    No optimizer from the old task is implicitly restored on this transfer.
    """
    def __init__(self,rotations_path=None):
        super().__init__()
        if rotations_path is None:
            from supermimic import rotations
        else:
            # Explicit historical source overrides remain available for replay.
            path=Path(rotations_path).resolve()
            spec=importlib.util.spec_from_file_location('_contact_wm_source_rotations',path)
            rotations=importlib.util.module_from_spec(spec)
            sys.modules[spec.name]=rotations
            spec.loader.exec_module(rotations)
        self.rotations=rotations
        self.net=mlp([272,2048,2048,2048,70],True)
        self.in_norm=RunningNorm(272,1e-2);self.out_norm=RunningNorm(70,1e-4)
        self.contact_head=mlp([272,256,256,18])
        self.activation_head=nn.Linear(272,6)
        nn.init.zeros_(self.activation_head.weight);nn.init.zeros_(self.activation_head.bias)
        self.activation_calibrated=False
        self.transfer_report=None

    def load_state_dict(self,state_dict,strict=True,assign=False):
        if 'world_field.points_w' in state_dict and not hasattr(self,'world_field'):
            from holosoma.contact_wm.world_contact_field import WorldContactField
            self.world_field=WorldContactField(state_dict['world_field.points_w']).to(next(self.net.parameters()).device)
        return super().load_state_dict(state_dict,strict=strict,assign=assign)

    def load_source(self,checkpoint):
        from somaforge_core.robot_assets import validate_g1_asset_metadata
        c=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
        validate_g1_asset_metadata(json.loads(c['robot_asset_json']),context='SuperMimic WM transfer')
        force=json.loads(c['force_contract'])
        if (tuple(force['parts'])!=PARTS or force['scale_N']!=200. or force['head_frame']!='current pelvis heading'
                or c['observation_schema']!='local_heightmap_11x11_force6x3_heading_v1'):
            raise ValueError('Unsupported SuperMimic force/context contract')
        original={k:v for k,v in self.state_dict().items() if not k.startswith('activation_head.')}
        if original.keys()!=c['wm'].keys():raise ValueError('WM architecture/state keys differ')
        for k,v in original.items():
            if v.shape!=c['wm'][k].shape or not torch.isfinite(c['wm'][k]).all():
                raise ValueError('Invalid/incompatible source WM tensor: '+k)
        state={**c['wm'],**{k:v for k,v in self.state_dict().items() if k.startswith('activation_head.')}}
        self.load_state_dict(state,strict=True)
        self.transfer_report=dict(source=str(Path(checkpoint).resolve()),iteration=c['iteration'],
            inherited_tensors=len(c['wm']),inherited_parameters=sum(v.numel() for k,v in c['wm'].items() if not k.startswith(('in_norm.','out_norm.'))),
            inherited_force_head=True,inherited_normalizers=True,policy_imported=False,
            new_head='six native activation logits; not yet calibrated',activation_calibrated=False,
            task_manifest_sha256=c['task_manifest_sha256'],robot_asset=json.loads(c['robot_asset_json']),
            action_contract_verified=False,physics_contract_verified=False,optimization_allowed=False)
        return self.transfer_report

    def source_step(self,state,source_action,physics_normalized,heights,previous_force_w):
        """Exact inherited contract; caller supplies source-ordered/scaled inputs.

        33 physics inputs: normalized friction, 29 joint offsets, COM offset.
        Heights: 121 source-layout relative/clamped samples, not the PHP 63 scan.
        Previous force is N in world coordinates. New targets never enter WM.
        """
        b=len(state)
        if (state.shape!=(b,71) or source_action.shape!=(b,29) or physics_normalized.shape!=(b,33)
                or heights.shape!=(b,121) or previous_force_w.shape!=(b,6,3)):
            raise ValueError('SuperMimic state/context shape mismatch')
        r=self.rotations;R=rotation_xyzw(state[:,3:7]);H=r.yaw_mat(R);Rt=R.transpose(-1,-2)
        p,v,w,q,qd=state[:,:3],state[:,7:10],state[:,10:13],state[:,13:42],state[:,42:]
        vec=lambda matrix,value:(matrix@value[...,None]).squeeze(-1)
        invariant=torch.cat((p[:,2:3],r.mat_to_6d(H.transpose(-1,-2)@R),vec(Rt,v),vec(Rt,w),q,qd),-1)
        local_force=vec(H.transpose(-1,-2)[:,None],previous_force_w/200.).flatten(1)
        x=self.in_norm(torch.cat((invariant,source_action,physics_normalized,heights,local_force),-1))
        increments=self.net(x)
        if hasattr(self,'world_field'):
            increments=increments+self.world_field.dynamics_correction(state,previous_force_w,H)
        d=increments*self.out_norm.std()+self.out_norm.mean
        dp,dr,dv,dw,dq,dqd=d.split((3,3,3,3,29,29),-1)
        next_R=R@r.so3_exp(dr)
        wxyz=r.mat_to_quat(next_R)
        next_state=torch.cat((p+vec(H,dp),wxyz[:,[1,2,3,0]],v+vec(R,dv),w+vec(R,dw),q+dq,qd+dqd),-1)
        force=vec(H[:,None],self.contact_head(x).reshape(b,6,3))*200.
        logits=self.activation_head(x)
        result=dict(state=next_state,force_w=force,contact_logits=logits,contact=logits.sigmoid())
        if hasattr(self,'world_field'):result.update(self.world_field(next_state))
        return result

    def forward(self,state,action,contact,context):
        # No silent identity conversion of a PHP action into source control space.
        if not context.get('action_contract_verified',False) or not context.get('physics_contract_verified',False):
            raise ValueError('Verify source/PHP action and physics contracts before BPTT')
        return self.source_step(state,context['source_action'](action),context['physics_normalized'],
                                context['heights'],context['previous_force_w'])
