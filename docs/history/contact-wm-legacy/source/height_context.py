"""Differentiable reading of a static world height raster for inherited WM tests."""
import torch
from torch import nn
from torch.nn import functional as F
from holosoma.contact_wm.observations import rotation_xyzw

class HeightRaster(nn.Module):
    def __init__(self,height,minimum,resolution,php=False):
        super().__init__()
        if height.ndim!=3 or minimum.shape!=(height.shape[0],2) or resolution<=0:raise ValueError('Invalid raster')
        self.register_buffer('height',height[:,None]);self.register_buffer('minimum',minimum);self.resolution=resolution
        yy,xx=torch.meshgrid(torch.linspace(-.8,.8,11),torch.linspace(-.5,1.5,11),indexing='ij')
        if php:xx,yy=torch.meshgrid(torch.linspace(-.4,.4,9),torch.linspace(-.3,.3,7),indexing='ij')
        self.php=php
        self.register_buffer('local',torch.stack((xx.flatten(),yy.flatten()),-1))
    def forward(self,state):
        R=rotation_xyzw(state[:,3:7]);direction=R[:,:2,0]
        direction=direction/(direction.square().sum(-1,keepdim=True)+1e-9).sqrt()
        c,s=direction.unbind(-1);x,y=self.local.unbind(-1)
        world=torch.stack((c[:,None]*x-s[:,None]*y,s[:,None]*x+c[:,None]*y),-1)+state[:,None,:2]
        indices=(world-self.minimum[:,None])/self.resolution
        limits=state.new_tensor([self.height.shape[-1]-1,self.height.shape[-2]-1])
        if ((indices<0)|(indices>limits)).any():raise ValueError('Imagined view outside recorded terrain atlas')
        grid=2*indices/limits-1
        height=F.grid_sample(self.height,grid[:,None],align_corners=True,padding_mode='zeros').squeeze(1).squeeze(1)
        relative=height-state[:,2:3]
        return relative if self.php else relative.clamp(-2,2)
