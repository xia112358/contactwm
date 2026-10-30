"""Teacher uses its original task/reference clock, sharing the student's live state."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch

from holosoma.managers.command.terms.wbt import FAKE_BODY_NAME_ALIASES, MultiMotionLoader
from holosoma.php_contact.dataset import load_prepared
from holosoma.php_contact.segments import TeacherEnvironment


def reference_view(env, cmd):
    if not hasattr(cmd, '_teacher_motion'):
        source_manifest = cmd.cfg.params['teacher_training_manifest']
        names = [FAKE_BODY_NAME_ALIASES.get(name, name) for name in env.simulator._body_list]
        cmd._teacher_motion = MultiMotionLoader('', names, env.simulator.dof_names, device=env.device,
            motion_manifest=source_manifest,
            canonicalize_motion_order_on_load=cmd.motion_cfg.canonicalize_motion_order_on_load)
        original = {str(Path(path).resolve()): i for i, path in enumerate(cmd._teacher_motion.motion_files)}
        prepared = load_prepared(cmd.motion_cfg.motion_manifest)
        rows = {str(Path(row['motion_file']).resolve()): row for row in prepared['motion_files']}
        source_ids, frames = [], []
        for filename in cmd.motion.motion_files:
            row = rows[str(Path(filename).resolve())]
            source_id = original[str(Path(row['teacher_reference_file']).resolve())]
            source_ids.append(source_id)
            with np.load(row['contact_file'], allow_pickle=False) as z:
                clock = z['teacher_frame'].copy()
            count = int(cmd._teacher_motion.motion_end_idx[source_id]-cmd._teacher_motion.motion_start_idx[source_id])
            if clock.ndim != 1 or not np.issubdtype(clock.dtype, np.integer) or (clock < 0).any() or (clock >= count).any():
                raise ValueError('Recorded teacher reference clock is invalid')
            frames.append(clock)
        cmd._teacher_source_ids = torch.tensor(source_ids, dtype=torch.long, device=env.device)
        cmd._teacher_source_frames = torch.as_tensor(np.concatenate(frames), device=env.device)
    view = copy.copy(cmd)
    view.motion = cmd._teacher_motion
    view.motion_ids = cmd._teacher_source_ids[cmd.motion_ids]
    view.time_steps = cmd._teacher_source_frames[cmd.time_steps]+view.motion.motion_start_idx[view.motion_ids]
    view._body_indexes_in_motion = view.motion._body_indexes
    view._joint_indexes_in_motion = view.motion._joint_indexes
    view._future_joint_cache = {}
    view._future_ref_body_cache = {}
    view._policy_reference_window_cache = None
    return TeacherEnvironment(env, view)
