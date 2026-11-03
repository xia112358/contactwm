"""Lossless native contact evidence at every solver substep, for offline labels."""
from pathlib import Path
import json
import torch


class SubstepCapture:
    def __init__(self, env, output):
        from .contact_native import NativeContacts
        self.env, self.manager = env, env.sim.backend_manager
        self.reader = NativeContacts(env, direct=True, primary_faces=True)
        self.reader._known_mask = lambda: torch.ones((env.N, 6),device=env.device,dtype=torch.bool)
        self.output = Path(output)
        self.output.mkdir(parents=True,exist_ok=False)
        self.rows = []
        self.tick = 0

    def __enter__(self):
        from somaforge_core.robot_assets import canonical_g1_asset_metadata
        from .contact_control import PARTS
        self.original = self.manager._step_solver
        self.graph = self.manager._cfg.use_cuda_graph
        self.manager._cfg.use_cuda_graph = False
        self.manager._invalidate_graph()
        self.manager._step_solver = self._step
        (self.output/'contract.json').write_text(json.dumps(dict(
            schema='native_substep_contact_evidence_v1',dt=self.manager.get_solver_dt(),
            timing='constraint solve and part poses before integration; independent of post-step state',
            selection='primary_horizontal_upward_faces_v1',
            robot_asset=canonical_g1_asset_metadata(collision_profile=self.env.collision_profile),
            part_order=PARTS,world_task_ids=self.env.task_ids.cpu().tolist(),
            source_config='../config.json',
            terrain_faces=self.reader.world_faces,
            world_origins=self.env.scene.env_origins.cpu().tolist(),
            body_labels=list(self.reader.model.body_label)),indent=2))
        return self

    def _step(self,*args,**kwargs):
        self.original(*args,**kwargs)
        self.reader.read_device()
        evidence=self.reader.last_evidence
        selected=self.reader.last_primary_selection
        p,R=self.reader.last_part_poses
        self.rows.append(dict(tick=self.tick,rows={k:v.cpu() for k,v in evidence.rows.items()},
            active=evidence.active.cpu(),allocated=evidence.allocated.cpu(),
            eligible=selected['eligible'][:evidence.count].cpu(),
            groups=selected['groups'][:evidence.count].cpu(),
            surfaces=selected['surfaces'][:evidence.count].cpu(),
            robot_side=selected['robot_side'][:evidence.count].cpu(),
            part_position=p.cpu(),part_rotation=R.cpu()))
        self.tick+=1

    def flush(self,control_step,live_worlds):
        torch.save(dict(control_step=control_step,live_worlds=list(sorted(live_worlds)),
                        substeps=self.rows),self.output/f'{control_step:06d}.pt')
        self.rows=[]

    def __exit__(self,*args):
        self.manager._step_solver=self.original
        self.manager._cfg.use_cuda_graph=self.graph
        self.manager._invalidate_graph()
