"""Audit saved native DAgger labels, student execution history and sampling."""
from supermimic.paths import runtime_path, runtime_root, workspace_root, module_root
import argparse
from pathlib import Path
import sys
pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    import json
    import torch
    from supermimic.lowstep import install
    config = json.loads((args.run/'config.json').read_text())
    install(config['task_manifest'], contact_model='force6',collision_profile=config.get('collision_profile','legacy_v1'))
    from supermimic.motion import MotionLib
    from supermimic.kinematics import G1Kinematics
    from supermimic.mdp import OBS_DIM
    from supermimic.networks import Policy
    import supermimic.networks as networks
    from supermimic.contact_dagger import balanced_indices
    torch.set_num_threads(4)
    torch.manual_seed(0)
    # Strongly unequal lengths must still have equal segment probability.
    groups = {i: dict(action=torch.full((n, 1), float(i))) for i, n in enumerate((1, 1000))}
    sample = balanced_indices(groups, 20000)['action']
    assert .48 < float(sample.mean()) < .52
    source = torch.load(config['checkpoint'], map_location='cuda:0', weights_only=False)
    networks.AMP_DTYPE = torch.bfloat16 if source['args'].get('amp', 'none') == 'bf16' else None
    hist = source['args'].get('hist', 10)
    goal_frames = source['args'].get('goal_frames', 10)
    teacher = Policy(OBS_DIM, 73*goal_frames, hist=hist).cuda().eval().requires_grad_(False)
    teacher.load_state_dict(source['pi'])
    spec = install(config['task_manifest'], contact_model='force6',collision_profile=config.get('collision_profile','legacy_v1'))
    motion = MotionLib.from_dir(spec['motion_dir'], G1Kinematics('cuda:0'), 'cuda:0')
    checked = 0
    for path in sorted((args.run/'dagger').glob('rollout_*.pt')):
        saved = torch.load(path, map_location='cuda:0', weights_only=False)
        assert saved['report']['covered_segments'] == len(config['contact_bounds'])-1
        assert not saved['report']['teacher_actions_in_student_segments']
        for row in saved['data'].values():
            assert row['condition'].shape[-1] == 39
            with torch.no_grad():
                label = teacher(row['teacher_history'], motion.goal(row['frame'].long(), goal_frames))
            # Batched bf16 GEMMs can differ slightly from rollout batch shape.
            torch.testing.assert_close(label, row['action'], atol=.03, rtol=.03)
            if len(row['action']) > 1:
                continuous=(row['episode_step'][1:]>row['episode_step'][:-1] if 'episode_step' in row
                            else torch.ones(len(row['action'])-1,dtype=torch.bool,device=row['action'].device))
                torch.testing.assert_close(row['history'][1:, -1, 64:93][continuous], row['student_action'][:-1][continuous])
            checked += len(row['action'])
    assert checked > 0
    print(f'PASS: {checked} native teacher labels, executed student-action histories, 39D timeless commands, equal segment sampling')


if __name__ == '__main__':
    main()
