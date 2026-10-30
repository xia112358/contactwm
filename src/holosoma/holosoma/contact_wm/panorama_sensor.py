"""Native synthetic LiDAR -> causal voxel history -> elevated gravity-aligned depth.

Historical reset seeding uses only sensor rays from the recorded *past* prefix.
This is an explicit recorded-history task reset, not a complete terrain fill.
"""
import math
import torch
from holosoma.acp.panorama_gpu import BatchedSurfacePanorama
from holosoma.contact_wm.panorama_policy import pack_panorama
from holosoma.contact_wm.observations import rotation_xyzw
from somaforge_core import G1_29DOF_JOINT_ORDER
from somaforge_core.batched_g1_kinematics import BatchedG1ForwardKinematics

class NativePanorama:
    def __init__(self,env,recording,joint_names,shifts,start_frame=261,ray_count=20000,period_steps=5):
        self.env=env;self.device=env.device;self.shifts=shifts;self.period_steps=period_steps;self.ray_count=ray_count
        self.body_id=env.simulator.body_names.index('torso_link')
        self.mesh=env.terrain_manager.get_state('locomotion_terrain').warp_mesh
        self.map=BatchedSurfacePanorama(env.num_envs,env.device)
        self.seed=BatchedSurfacePanorama(1,env.device)
        self.image=torch.zeros(env.num_envs,1,64,128,device=env.device)
        self.view=torch.zeros(env.num_envs,4,device=env.device)
        index=torch.arange(ray_count,device=self.device,dtype=torch.float32)
        low,high=math.sin(math.radians(-7)),math.sin(math.radians(52))
        z=low+(high-low)*(index+.5)/ray_count;r=(1-z*z).sqrt();az=index*2.399963229728653
        # Existing torso mount rotates sensor 180 degrees around X.
        self.rays=torch.stack((r*az.cos(),-r*az.sin(),-z),-1)
        self.offset=torch.tensor([0.,0.,.35],device=self.device)
        frames=torch.arange(0,start_frame,period_steps,device=self.device)
        s=torch.as_tensor(recording['state'],device=self.device)[frames]
        order=[joint_names.index(n) for n in G1_29DOF_JOINT_ORDER]
        qpos=torch.cat((s[:,:3],s[:,[6,3,4,5]],s[:,13:42][:,order]),-1)
        pos,rot=BatchedG1ForwardKinematics().to(self.device).link_poses(qpos,['torso_link'])
        centers=pos[:,0]+(rot[:,0]@self.offset[:,None]).squeeze(-1)
        points=self._cast(centers,rot[:,0],(frames-start_frame)*env.dt)
        for i,frame in enumerate(frames.cpu().tolist()):
            valid=torch.isfinite(points[i:i+1]).all(-1)
            self.seed.integrate_world(points[i:i+1],valid,centers[i:i+1],(frame-start_frame)*env.dt)
        self.reset(torch.arange(env.num_envs,device=self.device),0.)
        self.contract=dict(schema='causal_panorama_depth_v1',width=128,height=64,channels=1,invalid_depth=0.,
            voxel_resolution=.03,max_range=2.5,max_age=15.,sensor_hz=1/(env.dt*period_steps),sensor_body='torso_link',
            sensor_offset=[0,0,.35],sensor_rotation_xyzw=[1,0,0,0],rays=ray_count,
            view='root position + world Z .73; yaw only; no roll/pitch',
            reset_history=f'observed synthetic scans from native recorded frames 0..{start_frame-1}; no future frames',
            limitations='synthetic terrain-only instantaneous scans, exact odometry; no robot self-occlusion or deskew noise')
    def _cast(self,centers,rotations,timestamps):
        from holosoma.utils.warp_utils import ray_cast
        phase=-.37*torch.as_tensor(timestamps,device=self.device).expand(len(centers))
        c,s=phase.cos()[:,None],phase.sin()[:,None]
        x,y,z=self.rays.unbind(-1)
        rays=torch.stack((c*x-s*y,s*x+c*y,z[None].expand(len(centers),-1)),-1)
        directions=torch.einsum('bij,brj->bri',rotations,rays).contiguous()
        starts=centers[:,None].expand_as(directions).contiguous()
        return ray_cast(starts,directions,self.mesh)
    @torch.no_grad()
    def reset(self,ids,timestamp):
        self.map.keys[ids]=self.seed.keys[0]
        self.map.stamps[ids]=self.seed.stamps[0]+timestamp
        self.map.counts[ids]=self.seed.counts[0]
        self.map.clock[ids]=self.seed.clock[0]+timestamp
        self.image[ids]=0.
    @torch.no_grad()
    def update(self,step,state):
        if step%self.period_steps:return False
        timestamp=step*self.env.dt
        R=rotation_xyzw(self.env.simulator._rigid_body_rot[:,self.body_id])
        center=self.env.simulator._rigid_body_pos[:,self.body_id]+(R@self.offset[:,None]).squeeze(-1)
        points=self._cast(center,R,timestamp)-self.shifts[:,None]
        self.map.integrate_world(points,torch.isfinite(points).all(-1),center-self.shifts,timestamp)
        origin=state[:,:3].clone();origin[:,2]+=.73
        x,y,z,w=state[:,3:7].unbind(-1);angle=torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
        self.image=pack_panorama(self.map.render(origin,angle,timestamp))
        self.view=torch.cat((origin,angle[:,None]),-1)
        return True
