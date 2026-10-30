"""Small experimental PHP + panoramic planner bootstrap, not complete ACP Stage 2.

PPO/DAgger retains the published tracker contract. Planner supervision uses the
published fixed-q native contact intentions; successful live hindsight replay
is not implemented here. Reference-command warmup precedes limited mixing.
"""
from collections import deque
from dataclasses import dataclass, replace
import json
from pathlib import Path
import numpy as np
import torch

from holosoma.acp.panorama import SurfacePanorama
from holosoma.acp.panorama_model import PanoramaContactPlanner, panorama_planner_loss
from holosoma.acp.php_bridge import PLANNER_PARTS, PHP_PARTS, reorder, tracker_contacts
from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager
from holosoma.php_contact import segments, terms
from holosoma.php_contact.distill import SegmentDistillPPO


class PanoramaPHPEnvironment(WholeBodyTrackingManager):
    def __init__(self, tyro_config, *, device):
        self.pano_settings=tyro_config.command.setup_terms['motion_command'].params['panorama']
        self.pano_planner=None
        self.pano_sensor=None
        self.pano_maps=[]
        self.pano_iteration=0
        self.pano_replay=deque(maxlen=512)
        self.pano_inputs=None
        self.pano_prediction=None
        self.pano_time=0.
        self.pano_reference_count=0
        self.pano_predicted_count=0
        super().__init__(tyro_config,device=device)
        self.pano_planner=PanoramaContactPlanner().to(device)
        self.pano_maps=[SurfacePanorama(backend='warp') for _ in range(self.num_envs)]
        self.reset_all(step_after_reset=False)

    def _init_buffers(self):
        super()._init_buffers()
        self.pano_history=torch.zeros(self.num_envs,5,93,device=self.device)

    def _update_counters_each_step(self):
        super()._update_counters_each_step()
        self.pano_time+=self.dt

    def _compute_observations(self):
        self.pano_inputs=None;self.pano_prediction=None
        self.pano_history[:,:-1]=self.pano_history[:,1:].clone()
        self.pano_history[:,-1]=terms.proprioception(self)
        super()._compute_observations()

    def _reset_buffers_callback(self, env_ids, target_buf=None):
        super()._reset_buffers_callback(env_ids,target_buf)
        if hasattr(self,'pano_history'):self.pano_history[env_ids]=0
        if self.pano_sensor is not None:self.pano_sensor.reset(env_ids)
        for i in env_ids.tolist():
            if i<len(self.pano_maps):self.pano_maps[i].reset()
        self.pano_inputs=None;self.pano_prediction=None

    @torch.no_grad()
    def read_panorama(self):
        if self.pano_inputs is not None:return self.pano_inputs
        from holosoma.acp.perception import ACPPerception
        if self.pano_sensor is None:
            self.acp_settings={'terrain_observation':'fused_raw','lidar_points':20000,'lidar_hz':10.}
            self.pano_sensor=ACPPerception(self)
        self.pano_sensor.update(self.pano_time)
        scan=self.pano_sensor.latest_scan
        if scan is not None:
            for i in scan['sampled_envs'].nonzero().flatten().tolist():
                self.pano_maps[i].integrate(*(scan[k][i].detach().cpu().numpy() for k in
                    ('points_sensor','valid','position_w','quaternion_xyzw')),float(scan['timestamps'][i]))
        root=self.simulator.robot_root_states
        angle=terms.yaw(root[:,3:7]);origin=root[:,:3].clone();origin[:,2]+=.73
        result=[]
        for i,m in enumerate(self.pano_maps):
            a=float(angle[i]);q=[0,0,np.sin(a/2),np.cos(a/2)]
            result.append(m.render(origin[i].cpu().numpy(),q,max(self.pano_time,m.clock)))
        x={k:torch.as_tensor(np.stack([r[k] for r in result]),device=self.device) for k in ('depth','valid','age','confidence')}
        x.update(history=self.pano_history.flatten(1).clone(),goal=segments.goal(self),
                 base_position_w=root[:,:3].clone(),view_origin_w=origin,yaw=angle)
        self.pano_inputs=x
        self.log_dict['Panorama/coverage']=x['valid'].float().mean()
        return x

    def _pre_physics_step(self, actions):
        if self.pano_planner is not None:
            x=self.read_panorama();cmd=terms.command(self)
            targets=reorder(cmd.goal_position_w+cmd._reference_env_origins[:,None],PHP_PARTS,PLANNER_PARTS)
            active=reorder(cmd.goal_active,PHP_PARTS,PLANNER_PARTS)
            for i in range(self.num_envs):
                row={k:v[i].detach().cpu().clone() for k,v in x.items()}
                row.update(target_w=targets[i].detach().cpu().clone(),occupancy=active[i].detach().cpu().clone(),
                           known=torch.ones(6,dtype=torch.bool))
                self.pano_replay.append(row)
        super()._pre_physics_step(actions)


def future_contacts(env):
    reference=segments.future_contacts(env)
    if env.pano_planner is None:return reference
    x=env.read_panorama()
    with torch.no_grad():
        if env.pano_prediction is None:env.pano_prediction=env.pano_planner(**x)
        prediction=tracker_contacts(env.pano_prediction)
    s=env.pano_settings
    probability=min(s['max_mix'],max(0.,(env.pano_iteration-s['warmup'])/max(1,s['ramp'])*s['max_mix']))
    # A complete environment command is chosen; never interpolate contact XYZ.
    use=(torch.rand(env.num_envs,1,device=env.device)<probability)
    env.pano_predicted_count+=int(use.sum());env.pano_reference_count+=int((~use).sum())
    env.log_dict['Panorama/command_probability']=torch.tensor(probability,device=env.device)
    return torch.where(use,prediction,reference)


class PanoramaPHPPPO(SegmentDistillPPO):
    def _setup_models_and_optimizer(self):
        super()._setup_models_and_optimizer()
        self.pano_optimizer=torch.optim.Adam(self.env.pano_planner.parameters(),lr=1e-4)
        self.pano_updates=0
        self.pano_start_iteration=0

    def _rollout_step(self, obs_dict):
        self.env.pano_iteration=self.current_learning_iteration-self.pano_start_iteration
        return super()._rollout_step(obs_dict)

    def _training_step(self):
        losses=super()._training_step()
        replay=self.env.pano_replay
        if not replay:return losses
        total=0.;positives=0
        for _ in range(4):
            rows=[replay[i] for i in torch.randint(len(replay),(4,)).tolist()]
            batch={k:torch.stack([r[k] for r in rows]).to(self.device) for k in rows[0]}
            labels={k:batch.pop(k) for k in ('target_w','occupancy','known')}
            self.pano_optimizer.zero_grad(set_to_none=True)
            result=panorama_planner_loss(self.env.pano_planner(**batch),**labels)
            if not torch.isfinite(result['loss']):raise RuntimeError('Nonfinite panoramic planner loss')
            result['loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.env.pano_planner.parameters(),1.)
            self.pano_optimizer.step();self.pano_updates+=1
            total+=float(result['loss'].detach());positives+=int(result['positive_count'])
        losses['PanoramaPlanner']=total/4
        record=dict(iteration=self.current_learning_iteration,stage2_iteration=self.current_learning_iteration-self.pano_start_iteration,planner_updates=self.pano_updates,
                    planner_loss=total/4,supported_positive_labels=positives,replay_samples=len(replay),
                    reference_commands=self.env.pano_reference_count,predicted_commands=self.env.pano_predicted_count,
                    supervision='published_fixed_q_contact_intentions',mode='experimental_stage2_bootstrap')
        with open(Path(self.log_dir)/'panorama_metrics.jsonl','a') as f:f.write(json.dumps(record)+'\n')
        print('PANORAMA_UPDATE '+json.dumps(record),flush=True)
        return losses

    def _checkpoint_metadata(self, **kwargs):
        metadata=super()._checkpoint_metadata(**kwargs)
        metadata['panorama_stage2']=dict(schema='php_panorama_bootstrap_v1',config=self.env.pano_planner.config,
            settings=self.env.pano_settings,model=self.env.pano_planner.state_dict(),optimizer=self.pano_optimizer.state_dict(),
            updates=self.pano_updates,start_iteration=self.pano_start_iteration,replay=list(self.env.pano_replay),parts=PLANNER_PARTS)
        return metadata

    def load(self,path):
        checkpoint=self._load_checked_checkpoint(path) if path else None
        saved=checkpoint.get('panorama_stage2') if checkpoint else None
        if checkpoint and not saved:
            from somaforge_core.published_data import resolve_published_path
            from holosoma.agents.ppo.ppo import PPO
            relocated={str(resolve_published_path(k)):v for k,v in checkpoint['routed_teacher_sha256'].items()}
            if relocated!=self.teacher_actor.hashes:raise ValueError('Stage 1 teacher hash mismatch after relocation')
            layout={k:self.algo_obs_dim_dict[k] for k in self.actor_obs_keys}
            if checkpoint['student_observation_layout']!=layout:raise ValueError('Stage 1 observation layout mismatch')
            result=PPO.load(self,path)
            self._routed_resuming=False
        else:
            result=super().load(path)
        if path:
            if saved:
                if saved['settings']!=self.env.pano_settings:raise ValueError('Panorama resume settings changed')
                self.env.pano_planner.load_state_dict(saved['model'])
                self.pano_optimizer.load_state_dict(saved['optimizer']);self.pano_updates=saved['updates']
                self.pano_start_iteration=saved.get('start_iteration',0)
                self.env.pano_replay.extend(saved['replay'])
            else:
                self.pano_start_iteration=self.current_learning_iteration
                self._pending_env_state=None
            self.logging_helper.num_learning_iterations=self.current_learning_iteration+self.config.num_learning_iterations
        return result


@dataclass(frozen=True)
class Options:
    checkpoint: str
    manifest: str='runtime/current/motions/php_stage1_20261025_v1/prepared/train.json'
    teacher_spec: str='runtime/current/motions/php_stage1_20261025_v1/prepared/teachers.json'
    num_envs: int=4
    iterations: int=200
    save_interval: int=25
    warmup: int=50
    ramp: int=100
    max_mix: float=.2
    name: str='panorama_six_bootstrap'


def main():
    from holosoma.utils.sim_utils import parse_isaaclab_launcher_args,sync_launcher_headless_config
    launcher=parse_isaaclab_launcher_args(__doc__)
    import tyro
    o=tyro.cli(Options)
    if not (0<=o.max_mix<=1) or o.warmup<0 or o.ramp<1:raise ValueError('Invalid panorama schedule')
    from holosoma.php_contact.distill import Options as BaseOptions,build_config
    cfg=build_config(BaseOptions(manifest=o.manifest,teacher_spec=o.teacher_spec,checkpoint=o.checkpoint,
        num_envs=o.num_envs,iterations=o.iterations,save_interval=o.save_interval,name=o.name,
        output='runtime/current/models/php_panorama_stage2',segment_mode='continuous',pure_distill_iters=200,regularizer_mse_coef=.1))
    term=cfg.command.setup_terms['motion_command']
    term=replace(term,params={**term.params,'panorama':dict(warmup=o.warmup,ramp=o.ramp,max_mix=o.max_mix)})
    actor=cfg.observation.groups['actor_obs']
    actor=replace(actor,terms={**actor.terms,'contacts':replace(actor.terms['contacts'],func='holosoma.acp.php_stage2:future_contacts')})
    cfg=replace(cfg,env_class='holosoma.acp.php_stage2.PanoramaPHPEnvironment',
        command=replace(cfg.command,setup_terms={'motion_command':term}),
        observation=replace(cfg.observation,groups={**cfg.observation.groups,'actor_obs':actor}),
        algo=replace(cfg.algo,_target_='holosoma.acp.php_stage2.PanoramaPHPPPO',config=replace(cfg.algo.config,
            export_onnx=False,num_mini_batches=4)),)
    cfg=sync_launcher_headless_config(cfg,launcher)
    from holosoma.train_agent import train
    train(cfg,launcher_args=launcher)

if __name__=='__main__':main()
