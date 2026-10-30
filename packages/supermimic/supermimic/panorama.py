"""Asymmetric full-resolution depth: causal actor history, oracle WM rendering.

The ray renderer represents the validated static upright boxes and z=0 floor.
It is an optical model, never a replacement for Newton contact forces.
"""
from __future__ import annotations
import hashlib
import math
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
import warp as wp
from .lowstep_observations import ForceOnlyContext
from .rotations import quat_to_mat, yaw_mat

from supermimic.paths import module_root
PERCEPTION_SCHEMA = 'actor_causal_panorama64x128_wm_oracle_depth_force6_v1'
ROLLING_SCHEMA = 'actor_rolling_panorama64x128_wm_oracle_depth_force6_v2'
ORACLE_SCHEMA = 'actor_and_wm_oracle_panorama64x128_force6_actor_no_force_v3'
H, W, FEATURES = 64, 128, 63

@wp.func
def surface_hit(o: wp.vec3, d: wp.vec3, task: int, boxes: wp.array3d(dtype=float), limit: float, minimum: float):
    best=limit+1.0
    normal=wp.vec3(0.0)
    if wp.abs(d[2])>1.e-8:
        t=-o[2]/d[2]
        if t>=minimum and t<=limit:
            best=t;normal=wp.vec3(0.0,0.0,1.0)
    for b in range(boxes.shape[1]):
        if boxes[task,b,6]>0.0:
            center=wp.vec3(boxes[task,b,0],boxes[task,b,1],boxes[task,b,2])
            half=wp.vec3(boxes[task,b,3],boxes[task,b,4],boxes[task,b,5])
            a=boxes[task,b,7];co=wp.cos(a);si=wp.sin(a);v=o-center
            lo=wp.vec3(co*v[0]+si*v[1],-si*v[0]+co*v[1],v[2])
            ld=wp.vec3(co*d[0]+si*d[1],-si*d[0]+co*d[1],d[2])
            near=-1.e20;far=1.e20;near_axis=int(0);far_axis=int(0);valid=int(1)
            for j in range(3):
                if wp.abs(ld[j])<1.e-8:
                    if wp.abs(lo[j])>half[j]:valid=0
                else:
                    t0=(-half[j]-lo[j])/ld[j];t1=(half[j]-lo[j])/ld[j]
                    low=wp.min(t0,t1);high=wp.max(t0,t1)
                    if low>near:near=low;near_axis=j
                    if high<far:far=high;far_axis=j
            t=near;axis=near_axis
            if near<minimum:t=far;axis=far_axis
            if valid==1 and far>=near and t>=minimum and t<best and t<=limit:
                local_normal=wp.vec3(0.0);local_normal[axis]=1.0
                normal=wp.vec3(co*local_normal[0]-si*local_normal[1],si*local_normal[0]+co*local_normal[1],local_normal[2]);best=t
    return wp.vec4(best,normal[0],normal[1],normal[2])

@wp.kernel
def oracle_forward(origins:wp.array(dtype=wp.vec3),angles:wp.array(dtype=float),tasks:wp.array(dtype=int),
                   directions:wp.array(dtype=wp.vec3),boxes:wp.array3d(dtype=float),depth:wp.array2d(dtype=float)):
    e,r=wp.tid();a=angles[e];v=directions[r]
    d=wp.vec3(wp.cos(a)*v[0]-wp.sin(a)*v[1],wp.sin(a)*v[0]+wp.cos(a)*v[1],v[2])
    hit=surface_hit(origins[e],d,tasks[e],boxes,2.5,.12)
    depth[e,r]=0.0
    if hit[0]<=2.5:depth[e,r]=hit[0]/2.5

@wp.kernel
def oracle_backward(origins:wp.array(dtype=wp.vec3),angles:wp.array(dtype=float),tasks:wp.array(dtype=int),
                    directions:wp.array(dtype=wp.vec3),boxes:wp.array3d(dtype=float),grad:wp.array2d(dtype=float),
                    grad_o:wp.array(dtype=wp.vec3),grad_a:wp.array(dtype=float)):
    e,r=wp.tid();a=angles[e];v=directions[r]
    d=wp.vec3(wp.cos(a)*v[0]-wp.sin(a)*v[1],wp.sin(a)*v[0]+wp.cos(a)*v[1],v[2])
    hit=surface_hit(origins[e],d,tasks[e],boxes,2.5,.12)
    n=wp.vec3(hit[1],hit[2],hit[3]);den=wp.dot(n,d)
    if hit[0]<=2.5 and wp.abs(den)>1.e-8:
        g=grad[e,r]/2.5
        wp.atomic_add(grad_o,e,-g*n/den)
        dd=wp.vec3(-d[1],d[0],0.0)
        wp.atomic_add(grad_a,e,-g*hit[0]*wp.dot(n,dd)/den)

@wp.kernel
def lidar_cast(origins:wp.array(dtype=wp.vec3),rotations:wp.array(dtype=wp.mat33),tasks:wp.array(dtype=int),
               rays:wp.array(dtype=wp.vec3),boxes:wp.array3d(dtype=float),phase:float,points:wp.array2d(dtype=wp.vec3),valid:wp.array2d(dtype=wp.bool)):
    e,r=wp.tid();v=rays[r]
    scan=wp.vec3(wp.cos(phase)*v[0]-wp.sin(phase)*v[1],wp.sin(phase)*v[0]+wp.cos(phase)*v[1],v[2])
    d=rotations[e]*scan;hit=surface_hit(origins[e],d,tasks[e],boxes,5.0,.001)
    valid[e,r]=hit[0]<=5.0
    points[e,r]=origins[e]+hit[0]*d


def launch(kernel,shape,inputs,device):
    with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(device))):
        wp.launch(kernel,dim=shape,inputs=inputs,device=str(device))


class OracleDepth(torch.autograd.Function):
    @staticmethod
    def forward(ctx,origins,angles,tasks,directions,boxes):
        origins=origins.contiguous();angles=angles.contiguous();tasks=tasks.to(torch.int32).contiguous()
        depth=torch.empty((len(origins),len(directions)),device=origins.device)
        launch(oracle_forward,depth.shape,[wp.from_torch(origins.detach(),dtype=wp.vec3),wp.from_torch(angles.detach()),wp.from_torch(tasks),
               wp.from_torch(directions,dtype=wp.vec3),wp.from_torch(boxes),wp.from_torch(depth)],origins.device)
        ctx.save_for_backward(origins,angles,tasks,directions,boxes)
        return depth.reshape(len(origins),1,-1,W)
    @staticmethod
    def backward(ctx,grad):
        origins,angles,tasks,directions,boxes=ctx.saved_tensors
        go=torch.zeros_like(origins);ga=torch.zeros_like(angles)
        launch(oracle_backward,(len(origins),len(directions)),[wp.from_torch(origins.detach(),dtype=wp.vec3),wp.from_torch(angles.detach()),wp.from_torch(tasks),
               wp.from_torch(directions,dtype=wp.vec3),wp.from_torch(boxes),wp.from_torch(grad.reshape(len(origins),-1).contiguous()),
               wp.from_torch(go,dtype=wp.vec3),wp.from_torch(ga)],origins.device)
        return go,ga,None,None,None


class SphericalDownsample(nn.Module):
    def __init__(self,inputs,outputs):
        super().__init__();self.conv=nn.Conv2d(inputs,outputs,3,stride=2)
    def forward(self,x):
        return F.elu(self.conv(F.pad(F.pad(x,(1,1,0,0),mode='circular'),(0,0,1,1))))


def encoder(kind='cnn',resolution=(H,W)):
    if kind=='mlp':
        return nn.Sequential(nn.Flatten(),nn.Linear(math.prod(resolution),128),nn.ELU(),nn.Linear(128,FEATURES))
    if kind!='cnn':raise ValueError('Unknown depth encoder')
    return nn.Sequential(SphericalDownsample(1,16),SphericalDownsample(16,24),SphericalDownsample(24,32),
                         nn.AdaptiveAvgPool2d((4,8)),nn.Flatten(),nn.Linear(1024,128),nn.ELU(),nn.Linear(128,FEATURES))


def encode_depth(module,image):
    from .networks import AMP_DTYPE
    with torch.autocast(device_type='cuda',dtype=AMP_DTYPE or torch.float32,enabled=AMP_DTYPE is not None):
        return module(image).float()


class ImageBank:
    """Store full-resolution frames once per scan; references never enter networks."""
    def __init__(self,steps,n,device,cpu_replay=False):
        self.capacity=math.ceil(steps/5)+4;self.generation=-1
        self.device=torch.device(device);self.cpu_replay=cpu_replay
        self.images=torch.empty((self.capacity,n,1,H,W),dtype=torch.float16,
                                device='cpu' if cpu_replay else device,pin_memory=cpu_replay)
        self.generations=torch.full((self.capacity,),-1,device='cpu' if cpu_replay else device,dtype=torch.long)
        self.events=[None]*self.capacity;self.current=None
    def add(self,images):
        self.generation+=1;i=self.generation%self.capacity
        if self.cpu_replay:
            self.current=images.to(torch.float16)
            self.images[i].copy_(self.current,non_blocking=True)
            self.events[i]=torch.cuda.Event();self.events[i].record()
        else:
            self.images[i].copy_(images)
        self.generations[i]=self.generation
        return self.generation
    def gather(self,generations,envs):
        generations=generations.long();envs=envs.long();valid=generations>=0;slots=generations.clamp(min=0)%self.capacity
        if self.cpu_replay:
            if self.current is not None and bool(((generations==self.generation)|~valid).all()):
                return torch.where(valid[:,None,None,None],self.current[envs].float(),0.)
            gc=generations.cpu();ec=envs.cpu();sc=slots.cpu();vc=valid.cpu()
            if bool((vc & (self.generations[sc]!=gc)).any()):
                raise RuntimeError('Historical panorama expired from replay; refusing stale/newer frame substitution')
            for slot in sc[vc].unique().tolist():
                if self.events[slot] is not None:self.events[slot].synchronize()
            batch=self.images[sc,ec].pin_memory().to(self.device,non_blocking=True).float()
            return torch.where(valid[:,None,None,None],batch,0.)
        if bool((valid & (self.generations[slots]!=generations)).any()):
            raise RuntimeError('Historical panorama expired from replay; refusing stale/newer frame substitution')
        image=self.images[slots,envs].float()
        return torch.where(valid[:,None,None,None],image,0.)


class PanoramaContext(ForceOnlyContext):
    dim=FEATURES+18
    policy_dim=7  # 5 view-pose features, 2 replay routing IDs (not network inputs)
    extra_xi_dim=6
    schema=PERCEPTION_SCHEMA
    def __init__(self,tasks,buffer_steps=1000,rolling=False,oracle=False,depth_layout='full',depth_encoder='cnn'):
        wp.init()
        super().__init__(tasks);self.buffer_steps=buffer_steps;self.cache={};self.bank=None;self.depth_encoder=None
        self.rolling=rolling
        self.oracle=oracle
        if depth_layout not in ('full','lower_band') or (depth_layout!='full' and not oracle):
            raise ValueError('Lower-band depth requires oracle perception')
        self.depth_layout=depth_layout
        self.encoder_kind=depth_encoder
        self.resolution=(40,W) if depth_layout=='lower_band' else (H,W)
        self.schema=ORACLE_SCHEMA if oracle else ROLLING_SCHEMA if rolling else PERCEPTION_SCHEMA
        if oracle:
            self.policy_dim=5  # current xyz/yaw/task routing, excluded from decoder
            self.extra_xi_dim=0
        boxes=torch.zeros(len(tasks),max(1,max(len(t['boxes']) for t in tasks)),8)
        for i,task in enumerate(tasks):
            for j,b in enumerate(task['boxes']):
                b=torch.tensor(b);R=quat_to_mat(b[3:7])
                if not torch.allclose(R[2],torch.tensor([0.,0.,1.]),atol=1e-5):raise ValueError('Only validated upright boxes supported')
                boxes[i,j,:3]=b[:3];boxes[i,j,3:6]=b[7:10]/2;boxes[i,j,6]=1;boxes[i,j,7]=torch.atan2(R[1,0],R[0,0])
        self.boxes_cpu=boxes
        az=-torch.pi+(torch.arange(W)+.5)*2*torch.pi/W
        if depth_layout=='lower_band':
            el=torch.deg2rad(15-(torch.arange(40)+.5)*105/40)
        else:
            el=torch.pi/2-(torch.arange(H)+.5)*torch.pi/H
        el,az=torch.meshgrid(el,az,indexing='ij')
        self.directions_cpu=torch.stack((el.cos()*az.cos(),el.cos()*az.sin(),el.sin()),-1).reshape(-1,3).contiguous()
    def tensors(self,device):
        key=str(device)
        if key not in self.cache:self.cache[key]=(self.boxes_cpu.to(device),self.directions_cpu.to(device))
        return self.cache[key]
    def oracle_image(self,state,xi):
        p,R=state[:2];shape=p.shape[:-1];p=p.reshape(-1,3);R=R.reshape(-1,3,3)
        origin=p+torch.tensor([0.,0.,.73],device=p.device);angle=torch.atan2(R[:,1,0],R[:,0,0])
        boxes,dirs=self.tensors(p.device)
        return OracleDepth.apply(origin,angle,xi[...,33].reshape(-1),dirs,boxes)
    def features(self,state,xi):
        if self.depth_encoder is None:raise RuntimeError('World-model depth encoder is not installed')
        shape=state[0].shape[:-1]
        def encode(p,R,ids):return encode_depth(self.depth_encoder,self.oracle_image((p,R),ids))
        # Bound activation memory across imagined steps; re-render on backward.
        if torch.is_grad_enabled() and not (self.rolling or self.oracle):z=checkpoint(encode,state[0],state[1],xi,use_reentrant=False)
        else:z=encode(state[0],state[1],xi)
        return torch.cat((z.reshape(*shape,FEATURES),self.local_forces(state,xi[...,34:52])),-1)
    def policy_features(self,state,xi):
        p,R=state[:2]
        if self.oracle:
            angle=torch.atan2(R[...,1,0],R[...,0,0])
            return torch.cat((p,angle[...,None],xi[...,33:34]),-1)
        view=xi[...,52:56]
        local=(yaw_mat(R).transpose(-1,-2)@(view[...,:3]-p).unsqueeze(-1)).squeeze(-1)
        angle=torch.atan2(R[...,1,0],R[...,0,0]);delta=view[...,3]-angle
        relative=torch.cat((local,delta.sin()[...,None],delta.cos()[...,None]),-1)
        relative=torch.where((xi[...,56]>=0)[...,None],relative,torch.zeros_like(relative))
        return torch.cat((relative,xi[...,56:58]),-1)
    @staticmethod
    def contract(rolling=False,oracle=False,depth_layout='full',depth_encoder='cnn'):
        files=['acp/panorama.py','acp/panorama_gpu.py','contact_wm/panorama_sensor.py','contact_wm/panorama_policy.py']
        if oracle:
            files=[]  # Oracle perception has no dependency on historical fusion code.
        result = dict(schema=ROLLING_SCHEMA if rolling else PERCEPTION_SCHEMA,resolution=[H,W],encoder_features=FEATURES,
                    actor='causal observed surface history only; no force inputs',world_model='oracle static-scene depth + six-part force; no history fusion',
                    critic='world-model oracle depth features + forces',reset='empty map; no reference/future/complete-map prefill',
                    imagination='freeze actor image at rollout start; update relative view pose each step; re-render WM oracle depth at predicted pose',
                    scan_hz=10,scan_rays=20000,voxel_resolution=.03,history_seconds=15,max_depth=2.5,
                    source_hashes={f:hashlib.sha256((module_root('holosoma')/f).read_bytes()).hexdigest() for f in files},
                    limitations='synthetic terrain-only rays; exact odometry; no robot self-occlusion, sensor noise or deskew')
        if rolling:
            result.update(history_representation='single-layer rolling spherical image: exact observed world point and timestamp per pixel; reproject then fuse new scans',
                          voxel_resolution=None,replay_storage='pinned CPU float16 images, only current frame and sampled batches on GPU',
                          activation_checkpointing=False,shared_motion_force_encoding=True,
                          occlusion_limitation='second-layer occluded points are discarded; not equivalent to full voxel surface history')
        if oracle:
            result.update(schema=ORACLE_SCHEMA,actor='oracle static-scene depth; no force inputs',
                          history_representation=None,history_seconds=0,voxel_resolution=None,scan_hz=None,scan_rays=0,
                          reset='no perception state',replay_storage='robot states only; depth regenerated at sampled state',
                          imagination='both actor and WM regenerate truth depth at each predicted pose',
                          shared_motion_force_encoding=True,actor_activation_checkpointing=True,
                          limitations='privileged static-scene geometry for both actor and WM; terrain-only optical model; not a deployable history-fusion perception policy')
            if depth_layout=='lower_band':
                result.update(resolution=[40,128],depth_layout=depth_layout,
                              azimuth_degrees=[-180,180],elevation_degrees=[-90,15],
                              ray_count=5120,angular_pixel_degrees=[2.625,2.8125])
            if depth_encoder=='mlp':
                result.update(depth_encoder='mlp',encoder_layers=[math.prod(result['resolution']),128,FEATURES],
                              actor_activation_checkpointing=False)
        return result


class HistoricalSensor:
    def __init__(self,env,context):
        self.env=env;self.context=context
        if context.rolling:
            from .rolling_panorama import RollingPanorama
            self.map=RollingPanorama(env.N,env.device)
        else:
            from holosoma.acp.panorama_gpu import BatchedSurfacePanorama
            self.map=BatchedSurfacePanorama(env.N,env.device)
        self.bank=context.bank=ImageBank(context.buffer_steps,env.N,env.device,cpu_replay=context.rolling)
        self.last_step=-1;self.ray_count=20000
        self.stats={}
        index=torch.arange(self.ray_count,device=env.device,dtype=torch.float32)
        z=math.sin(math.radians(-7))+(math.sin(math.radians(52))-math.sin(math.radians(-7)))*(index+.5)/self.ray_count
        r=(1-z*z).sqrt();az=index*2.399963229728653
        self.rays=torch.stack((r*az.cos(),-r*az.sin(),-z),-1).contiguous()
        self.env.xi[:,56]=-1;self.env.xi[:,57]=torch.arange(env.N,device=env.device)
    @torch.no_grad()
    def reset(self,ids):
        self.map.reset(ids);self.env.xi[ids,56]=-1
        self.env.xi[ids,52:56]=0
    @torch.no_grad()
    def update(self,step):
        if step%5 or self.last_step==step:return
        self.last_step=step;env=self.env;t=step*.02
        pose=env.robot.data.body_link_pose_w.torch[:,env.torso_id]
        R=quat_to_mat(torch.cat((pose[:,6:7],pose[:,3:6]),-1))
        center=pose[:,:3]-env.scene.env_origins+(R@torch.tensor([0.,0.,.35],device=env.device)).squeeze(-1)
        boxes,_=self.context.tensors(env.device)
        points=torch.empty((env.N,self.ray_count,3),device=env.device);valid=torch.empty((env.N,self.ray_count),device=env.device,dtype=torch.bool)
        launch(lidar_cast,valid.shape,[wp.from_torch(center.contiguous(),dtype=wp.vec3),wp.from_torch(R.contiguous(),dtype=wp.mat33),
               wp.from_torch(env.task_ids.to(torch.int32)),wp.from_torch(self.rays,dtype=wp.vec3),wp.from_torch(boxes),-.37*t,
               wp.from_torch(points,dtype=wp.vec3),wp.from_torch(valid)],env.device)
        state=env.state();origin=state[0].clone();origin[:,2]+=.73
        angle=torch.atan2(state[1][:,1,0],state[1][:,0,0])
        if self.context.rolling:
            render=self.map.update(points,valid,origin,angle,t)
        else:
            self.map.integrate_world(points,valid,center,t)
            render=self.map.render(origin,angle,t)
        image=torch.where(render['valid'],render['depth']/2.5,0.)[:,None]
        generation=self.bank.add(image)
        env.xi[:,52:56]=torch.cat((origin,angle[:,None]),-1);env.xi[:,56]=generation
        self.stats=dict(schema=self.context.schema,control_step=step,image_generation=generation,
                        resolution=[H,W],scan_hz=10,scan_rays=self.ray_count,
                        observed_pixel_fraction=float(render['valid'].float().mean()),
                        actor_force_input=False,wm_oracle_depth=True,
                        image_bank_gib=self.bank.images.numel()*self.bank.images.element_size()/2**30,
                        image_bank_device=str(self.bank.images.device),
                        rolling_map_gpu_gib=self.map.gpu_bytes/2**30 if self.context.rolling else None)


def policy_class(networks,context):
    vector_dim=99 if context.oracle else 104
    class ObservationNorm(networks.RunningNorm):
        def __init__(self):super().__init__(vector_dim)
        def update(self,x):return super().update(x[...,:vector_dim])
        def forward(self,x):return super().forward(x[...,:vector_dim])
    class PanoramaPolicy(nn.Module):
        def __init__(self,obs_dim,goal_dim,hist=10,z_dim=32):
            super().__init__();self.hist=hist
            self.enc=networks.mlp(goal_dim,[2048,1024,512,256],z_dim)
            self.depth_encoder=encoder(context.encoder_kind,context.resolution)
            self.dec=networks.mlp(z_dim+vector_dim*hist+FEATURES,[2048,1024,512,256],29)
            with torch.no_grad():self.dec[-1].weight.mul_(.01);self.dec[-1].bias.zero_()
            self.obs_norm=ObservationNorm();self.goal_norm=networks.RunningNorm(goal_dim)
        def perception_features(self,obs_hist):
            if context.oracle:
                camera=obs_hist[:,-1,99:104]
                boxes,dirs=context.tensors(camera.device)
                def encode_camera(camera):
                    origin=camera[:,:3]+torch.tensor([0.,0.,.73],device=camera.device)
                    image=OracleDepth.apply(origin,camera[:,3],camera[:,4],dirs,boxes)
                    return encode_depth(self.depth_encoder,image)
                return checkpoint(encode_camera,camera,use_reentrant=False) if torch.is_grad_enabled() and context.encoder_kind=='cnn' else encode_camera(camera)
            if context.bank is None:raise RuntimeError('Historical sensor bank not ready')
            refs=obs_hist[:,-1,104:106]
            return encode_depth(self.depth_encoder,context.bank.gather(refs[:,0],refs[:,1]))
        def forward_features(self,obs_hist,goal,features):
            z=torch.tanh(networks.run_mlp(self.enc,self.goal_norm(goal)))
            return networks.run_mlp(self.dec,torch.cat((z,self.obs_norm(obs_hist).flatten(-2),features),-1))
        def forward(self,obs_hist,goal):
            return self.forward_features(obs_hist,goal,self.perception_features(obs_hist))
    return PanoramaPolicy
