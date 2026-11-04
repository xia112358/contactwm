"""Lossless native contact evidence at every solver substep, for offline labels."""
from pathlib import Path
import json
import torch


class SubstepCapture:
    def __init__(self, env, output):
        from .contact_native import NativeContacts
        self.env, self.manager = env, env.sim.backend_manager
        self.reader = NativeContacts(env, direct=True, primary_faces=True)
        self.output = Path(output)
        self.output.mkdir(parents=True,exist_ok=False)
        self.rows = []
        from .contact_substep_support import SubstepContactMonitor
        self.monitor = SubstepContactMonitor(env,self.reader,self._sample)

    def __enter__(self):
        from somaforge_core.robot_assets import canonical_g1_asset_metadata
        from .contact_control import PARTS
        (self.output/'contract.json').write_text(json.dumps(dict(
            schema='native_substep_contact_evidence_v2',dt=self.manager.get_solver_dt(),
            support_contract=self.monitor.contract,
            timing='constraint solve and part poses before integration; independent of post-step state',
            selection='primary_horizontal_upward_faces_v1',
            robot_asset=canonical_g1_asset_metadata(collision_profile=self.env.collision_profile),
            part_order=PARTS,world_task_ids=self.env.task_ids.cpu().tolist(),
            source_config='../config.json',
            terrain_faces=self.reader.world_faces,
            world_origins=self.env.scene.env_origins.cpu().tolist(),
            body_labels=list(self.reader.model.body_label)),indent=2))
        self.monitor.__enter__()
        return self

    def _sample(self,tick,substep,observation,features,support):
        evidence=self.reader.last_evidence
        selected=self.reader.last_primary_selection
        p,R=self.reader.last_part_poses
        self.rows.append(dict(tick=tick,substep=substep,episode_step=(self.env.ep_step*self.monitor.substeps+substep).cpu().clone(),
            observation={k:v.cpu().clone() for k,v in observation.items()},
            features={k:v.cpu().clone() for k,v in features.items()},
            support={k:v.cpu().clone() for k,v in support.items()},rows={k:v.cpu() for k,v in evidence.rows.items()},
            active=evidence.active.cpu(),allocated=evidence.allocated.cpu(),
            eligible=selected['eligible'][:evidence.count].cpu(),
            groups=selected['groups'][:evidence.count].cpu(),
            surfaces=selected['surfaces'][:evidence.count].cpu(),
            robot_side=selected['robot_side'][:evidence.count].cpu(),
            part_position=p.cpu(),part_rotation=R.cpu()))

    def flush(self,control_step,live_worlds):
        torch.save(dict(control_step=control_step,live_worlds=list(sorted(live_worlds)),
                        summary={k:v.cpu() for k,v in self.monitor.summary().items()},substeps=self.rows),self.output/f'{control_step:06d}.pt')
        self.rows=[]

    def __exit__(self,*args):
        self.monitor.__exit__(*args)
