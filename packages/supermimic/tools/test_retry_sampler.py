"""Regression tests for SomaForge window parity and statistics-only probes."""
from supermimic.paths import runtime_path, runtime_root, workspace_root, module_root
import ast
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

pass
import torch
from supermimic.retry_sampler import Recent30FailureSampler, failure_window, training_transition


def make_sampler(n=4):
    motion = SimpleNamespace(clip_start=torch.tensor([0]), clip_end=torch.tensor([200]))
    return Recent30FailureSampler(torch.zeros(n+3, dtype=torch.long),
                                  torch.tensor([True]*n+[False]*3), motion)


class SamplerTests(unittest.TestCase):
    def test_exact_somaforge_window_parity(self):
        source = module_root('holosoma') / 'managers/command/terms/wbt.py'
        tree = ast.parse(source.read_text())
        function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and
                        n.name == '_sample_around_failure_frames')
        namespace = {'torch': torch}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(source), 'exec'), namespace)
        start = torch.tensor([0, 0, 100, 100])
        end = torch.tensor([201, 201, 302, 302])  # SomaForge uses exclusive end
        target = torch.tensor([0, 199, 101, 301])
        owner = SimpleNamespace(device='cpu', motion=SimpleNamespace(motion_start_idx=start, motion_end_idx=end),
                                motion_cfg=SimpleNamespace(failure_window_pre_frames=50,
                                                           failure_window_post_frames=20,
                                                           failure_window_before_prob=.7))
        for seed in range(20):
            torch.manual_seed(seed)
            expected = namespace['_sample_around_failure_frames'](owner, torch.arange(4), target)
            torch.manual_seed(seed)
            actual = failure_window(start, end-2, target)
            torch.testing.assert_close(actual, expected)

    def test_recent30_individual_probe_episodes(self):
        s = make_sampler()
        values = []
        for step in range(15):
            f = torch.tensor([10]*4 + [10+step, 20+step, 30+step])
            term = torch.tensor([False]*4+[True]*3)
            s.transition(f, term, torch.zeros(7,dtype=torch.bool))
            values += [float(x)/200 for x in f[-3:]]
        self.assertEqual(s.total_probes.item(),45)
        self.assertEqual(s.count.item(),30)
        self.assertAlmostEqual(float(s.recent[0,:,0].mean()),sum(values[-30:])/30,places=6)
        clone=make_sampler();clone.load_state_dict(s.state_dict())
        torch.testing.assert_close(clone.recent,s.recent)
        self.assertFalse(clone.retry.any())

    def test_probes_from_zero_and_uncensored(self):
        s=make_sampler();s.failed[:]=True;s.failed_frame[:]=120
        frames=s.sample(torch.arange(7))
        self.assertTrue((frames[-3:]==0).all())
        self.assertTrue(((frames[:4]>=70)&(frames[:4]<=140)).all())
        timeout=s.transition(torch.full((7,),100),torch.zeros(7,dtype=torch.bool),torch.ones(7,dtype=torch.bool))
        self.assertFalse(timeout[-3:].any())
        self.assertEqual(s.total_probes.item(),0)

    def test_completion_mix_and_success_horizon(self):
        s=make_sampler(10000);s.count[:]=30;s.recent[:,:,0]=.5
        torch.manual_seed(0);frames=s.sample(torch.arange(10000))
        fraction=float(s.retry[:10000].float().mean())
        self.assertTrue(.68 < fraction < .72)
        self.assertTrue(((frames[s.retry[:10000]]>=50)&(frames[s.retry[:10000]]<=120)).all())
        s.retry.zero_();s.retry[0]=True;s.target[0]=100
        term=torch.zeros(10003,dtype=torch.bool);timeout=term.clone()
        result=s.transition(torch.full((10003,),150),term,timeout)
        self.assertTrue(result[0]);self.assertFalse(result[1:].any())

    def test_no_probe_transitions_or_episode_metrics(self):
        ids=torch.tensor([0,2,4])
        out={'s':torch.arange(6)[:,None].float(),'r':torch.arange(6).float(),
             'episode_env_ids':torch.tensor([1,2,5]),'ep_len':torch.tensor([999.,3.,999.]),
             'ep_ret_norm':torch.tensor([999.,.5,999.])}
        selected=training_transition(out,ids)
        self.assertEqual(selected['s'].flatten().tolist(),[0,2,4])
        self.assertEqual(selected['ep_len'].tolist(),[3])
        self.assertEqual(selected['episode_env_ids'].tolist(),[2])

    def test_long_and_short_both_update_shared_completion(self):
        motion=SimpleNamespace(clip_start=torch.tensor([100]),clip_end=torch.tensor([1100]))
        s=Recent30FailureSampler(torch.zeros(7,dtype=torch.long),
                                torch.tensor([True]*4+[False]*3),motion,probe_mode='long_short')
        s.count[:]=30;s.recent[:,:,0]=.5
        frames=s.sample(torch.arange(7))
        self.assertEqual(frames[4].item(),100)
        self.assertTrue((frames[5:]==500).all())
        self.assertTrue((s.target[5:]==600).all())
        # Long probe fails at 20%; short probes reach 55%, ending at target+50.
        f=torch.tensor([100]*4+[300,650,650]);term=torch.tensor([False]*4+[True,False,False])
        timeout=s.transition(f,term,torch.zeros(7,dtype=torch.bool))
        self.assertTrue(timeout[5:].all())
        self.assertEqual(s.total_probes.item(),3)
        torch.testing.assert_close(s.recent[0,:3,0],torch.tensor([.2,.55,.55]))
        self.assertAlmostEqual(float(s.recent[0,:,0].mean()),(27*.5+.2+.55+.55)/30,places=6)
        # A short probe failure resets near its actual failure, not the old mean.
        s.failed[5]=True;s.failed_frame[5]=400
        sampled=s.sample(torch.tensor([5]))
        self.assertEqual(sampled.item(),300)
        self.assertEqual(s.target[5].item(),400)


if __name__=='__main__':
    unittest.main()
