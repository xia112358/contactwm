"""Direct contact control from causal panoramic depth; no height input to actor."""
import torch
from torch import nn
from torch.nn import functional as F
from holosoma.contact_wm.observations import yaw_local

class SphericalDownsample(nn.Module):
    def __init__(self,inputs,outputs):
        super().__init__();self.conv=nn.Conv2d(inputs,outputs,3,stride=2)
    def forward(self,x):
        return F.elu(self.conv(F.pad(F.pad(x,(1,1,0,0),mode='circular'),(0,0,1,1))))

class PanoramaContactPolicy(nn.Module):
    schema='contact_panorama_64x128_v1'
    def __init__(self,base_policy,initial_feature=None):
        super().__init__();self.net=base_policy.net;self.action_clip=base_policy.action_clip
        old=self.net[0];new=nn.Linear(old.in_features+5,old.out_features,device=old.weight.device,dtype=old.weight.dtype)
        with torch.no_grad():
            new.weight.zero_();new.weight[:,:old.in_features].copy_(old.weight);new.bias.copy_(old.bias)
        self.net[0]=new
        self.encoder=nn.Sequential(SphericalDownsample(1,16),SphericalDownsample(16,24),
            SphericalDownsample(24,32),nn.AdaptiveAvgPool2d((4,8)),nn.Flatten(),nn.Linear(1024,128),nn.ELU(),nn.Linear(128,63)).to(old.weight.device)
        nn.init.zeros_(self.encoder[-1].weight);nn.init.zeros_(self.encoder[-1].bias)
        if initial_feature is not None:
            with torch.no_grad():self.encoder[-1].bias.copy_(initial_feature)
        self.config=dict(schema=self.schema,height=64,width=128,channels=['depth'],invalid_depth=0.,
            max_range=2.5,max_age=15.,view_offset_z=.73,features=63,view_pose_features=5,
            height_input=False,feature_initialization='fixed prior feature bias; no runtime oracle input')
    def encode(self,image):
        if image.shape[1:]!=(1,64,128):raise ValueError('Expected B x 1 x 64 x 128 panoramic depth')
        known=torch.isfinite(image)&(image>0)
        safe=torch.where(known,image,0.)
        return self.encoder(safe)
    def forward_encoded(self,observation,condition,features,view_relative):
        # The original privileged height slice is deliberately never read.
        actor=torch.cat((observation[:,:39],features,observation[:,102:],condition,view_relative),-1)
        return self.net(actor).clamp(-self.action_clip,self.action_clip)
    def forward(self,observation,condition,*,panorama,state,view_pose):
        return self.forward_encoded(observation,condition,self.encode(panorama),relative_view(state,view_pose))

def relative_view(state,view_pose):
    local=yaw_local(view_pose[:,:3]-state[:,:3],state[:,3:7])
    x,y,z,w=state[:,3:7].unbind(-1)
    angle=torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
    difference=view_pose[:,3]-angle
    return torch.cat((local,difference.sin()[:,None],difference.cos()[:,None]),-1)

def pack_panorama(result,max_range=2.5,max_age=15.):
    known=result['valid']
    return torch.where(known,result['depth']/max_range,0.)[:,None]


class PanoramaReplay:
    """Store sensor frames once at 10 Hz, not once per 50 Hz control transition."""
    def __init__(self,capacity,num_envs,device):
        self.images=torch.empty(capacity,num_envs,1,64,128,dtype=torch.float16,device=device)
        self.generations=torch.full((capacity,),-1,dtype=torch.long,device=device)
        self.capacity=capacity;self.frame=-1
    def add(self,image):
        self.frame+=1;slot=self.frame%self.capacity
        self.images[slot].copy_(image);self.generations[slot]=self.frame
        return self.frame
    def gather(self,frames,env_ids):
        slot=frames%self.capacity
        if (self.generations[slot]!=frames).any():raise RuntimeError('Panorama replay frame expired; cannot substitute a newer observation')
        return self.images[slot,env_ids].float()
