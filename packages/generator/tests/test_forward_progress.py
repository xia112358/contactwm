from generator.forward_progress import ForwardProgressGate


def test_backtrack_does_not_recount_previous_progress():
    gate = ForwardProgressGate((0,0),(1,0),window=3,minimum_m=.03)
    for x in (.1,0,.1):
        assert gate.update((x,0))['forward_progress_valid']
    assert not gate.update((.1,0))['forward_progress_valid']


def test_sideways_motion_does_not_count_and_small_steps_accumulate():
    gate = ForwardProgressGate((0,0),(0,2),window=4,minimum_m=.035)
    for i in range(1,5):
        result=gate.update((i,.01*i))
    assert result['forward_progress_valid']
    for i in range(4):
        result=gate.update((10+i,.04))
    assert not result['forward_progress_valid']


def test_pool_gate_matches_scalar_until_rejection_then_preserves_history():
    import torch
    from generator.parallel_rollout_pool import ParallelRolloutPool
    reference={'current_q':torch.zeros(1,2)}
    pool=ParallelRolloutPool(reference,torch.tensor([0]),torch.tensor([-1]),1,
        global_directions=torch.tensor([[1.,0.]]),progress_window=3,progress_minimum_m=.03)
    scalar=ForwardProgressGate((0,0),(1,0),window=3,minimum_m=.03)
    for x in (.1,0,.1,.1):
        reasons={}
        result=pool.update(torch.tensor([0]),torch.tensor([[x,0.]]),{},
            torch.tensor([True]),torch.tensor([True]),lambda q,*args: {'current_q':q},reset_reasons=reasons)
        expected=scalar.update((x,0))['forward_progress_valid']
        assert bool(result['no_forward_progress_rejections']) == (not expected)
    assert pool.age.item()==3
    assert not pool.supervised.item()
    old_history=pool.progress_history.clone()
    old_q=pool.observation['current_q'].clone()
    for x in (float('nan'), 10.):
        result=pool.update(torch.tensor([0]),torch.tensor([[x,0.]]),{},torch.tensor([False]),
            torch.tensor([False]),lambda *args: (_ for _ in ()).throw(AssertionError('rejected')))
        assert result['rollbacks']==1 and result['resets']==0
        assert torch.equal(pool.progress_history,old_history)
        assert torch.equal(pool.observation['current_q'],old_q)
        assert pool.age.item()==3
    # A later valid prediction can advance from the same retained A.
    result=pool.update(torch.tensor([0]),torch.tensor([[.14,0.]]),{},torch.tensor([True]),
        torch.tensor([True]),lambda q,*args:{'current_q':q})
    assert result['accepted_transitions']==1
    assert pool.age.item()==4 and pool.retries.item()==0
