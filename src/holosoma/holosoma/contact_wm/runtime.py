"""Direct goal-conditioned inference. No world model or future motion is loaded."""
import torch
from holosoma.contact_wm.core import load_php_policy,ContactGoal
from holosoma.contact_wm.observations import yaw_local


def execution_failure(env):
    """Numerical/root-height failure guard; climbing inclination is not a fall.

    Stable support and slip are evaluated separately from native contacts.
    """
    root=env.simulator.robot_root_states[:]
    return (~torch.isfinite(root).all(-1))|(root[:,2]<.15)


def goal_observation(observation,state,goal,hint,step):
    obs=observation.clone();active=goal.establish|goal.maintain
    targets=torch.where(goal.maintain[...,None],goal.support_position_w,goal.position_w)
    local=yaw_local(targets-state[:,None,:3],state[:,3:7])
    obs[:,12:36]=torch.cat((torch.where(active[...,None],local,0.).flatten(1),active.float()),-1)
    obs[:,36:39]=hint
    deadline=state.new_full((len(state),1),max(0,goal.arrival_step-step)*.02)
    extra=torch.cat((yaw_local(goal.normal_w,state[:,3:7]).flatten(1),goal.maintain.float(),deadline),-1)
    return obs,extra


class ContactExecutor:
    def __init__(self,checkpoint,device,origin,goal_offset=(0.,0.,0.),planned_contact_points=None):
        saved=torch.load(checkpoint,map_location=device,weights_only=False)
        if saved['schema']!='contact_wm_direct_policy_v1':raise ValueError('Invalid direct policy checkpoint')
        self.policy,_,_=load_php_policy(saved['prior']['source']);self.policy.to(device)
        self.perception=saved.get('perception')
        if self.perception:
            from holosoma.contact_wm.panorama_policy import PanoramaContactPolicy
            if self.perception['schema']!=PanoramaContactPolicy.schema:raise ValueError('Unknown panorama policy schema')
            self.policy=PanoramaContactPolicy(self.policy).to(device)
        self.policy.load_state_dict(saved['policy'],strict=True);self.policy.eval()
        self.goal=ContactGoal(**{k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in saved['goal'].items()})
        shift=origin-torch.tensor(saved['reference_origin_w'],device=device)
        self.goal.position_w+=shift;self.goal.support_position_w+=shift
        if planned_contact_points is not None:
            if any(goal_offset):raise ValueError('Explicit planned points cannot be combined with goal offsets')
            from holosoma.contact_wm.core import load_planned_contact_points
            planned=load_planned_contact_points(planned_contact_points,device=device)
            # Planner positions are already in this execution's world frame.
            self.goal.position_w=torch.where(self.goal.establish[...,None],planned,self.goal.position_w)
        else:
            self.goal.position_w+=self.goal.establish[...,None]*torch.tensor(goal_offset,device=device)
        self.hint=saved['goal_hint'].to(device);self.start_step=saved['start_step'];self.horizon=saved['horizon']
        self.motion_index=saved['motion_index'];self.initialized=False

    def act(self,observation,state,step,support_points,*,panorama=None,view_pose=None):
        if not self.initialized:
            self.goal.support_position_w=support_points.clone();self.initialized=True
        obs,condition=goal_observation(observation,state,self.goal,self.hint,step-self.start_step)
        if self.perception:
            if panorama is None or view_pose is None:raise ValueError('Panorama policy requires causal fused depth and its reading pose')
            return self.policy(obs,condition,panorama=panorama,state=state,view_pose=view_pose)
        return self.policy(obs,condition)
