import torch
from generator.parallel_rollout_pool import ParallelRolloutPool
from generator.rollout_termination import physical_reset_masks, demonstration_loss


def reference(count):
    return {'current_q':torch.arange(count).float()[:,None],
            'current_contact':torch.zeros(count,1,dtype=torch.bool)}


def observed(q,indices,raw,chosen):
    return {'current_q':q,'current_contact':raw['contact'][chosen]}


def test_success_continues_but_failed_attempts_retain_input_within_window():
    pool=ParallelRolloutPool(reference(3),torch.arange(3),torch.tensor([1,2,-1]),2,max_episode_steps=80,max_retries=1)
    pool.indices[:]=torch.tensor([2,0])
    original=pool.observation['current_q'].clone()
    for step in range(70):
        slots,ids,inputs=pool.batch(2)
        q=inputs['current_q']+1
        result=pool.update(slots,q,{'contact':torch.ones(2,1,dtype=torch.bool)},
            torch.tensor([True,False]),torch.ones(2,dtype=torch.bool),observed)
        assert pool.age.tolist()==[step+1,0]
        assert torch.equal(pool.observation['current_q'][0],q[0])
        assert torch.equal(pool.observation['current_q'][1],original[1])
        assert pool.indices.tolist()==[2,0]
        assert pool.supervised.tolist()==[False,True]
        assert result['resets']==0 and result['timeout_resets']==0
    assert pool.retries.tolist()==[0,70]


def test_rejection_preserves_generated_input_and_scene_label_state():
    bank=reference(10);successor=torch.arange(10)+1;successor[-1]=-1
    pool=ParallelRolloutPool(bank,torch.arange(10),successor,3)
    pool.indices[:]=torch.tensor([1,3,9])
    pool.observation['current_q'][:]=torch.tensor([[1.],[43.],[9.]])
    pool.age[1]=5
    q=torch.tensor([[51.],[53.],[59.]],requires_grad=True)
    result=pool.update(torch.arange(3),q,{'contact':torch.ones(3,1,dtype=torch.bool)},
        torch.tensor([True,False,True]),torch.tensor([True,False,True]),observed)
    assert pool.indices.tolist()==[2,3,9]
    assert pool.observation['current_q'].flatten().tolist()==[51,43,59]
    assert pool.observation['current_contact'].flatten().tolist()==[True,False,True]
    assert not pool.observation['current_q'].requires_grad
    assert pool.age.tolist()==[1,5,1]
    assert pool.supervised.tolist()==[True,True,False]
    assert result['terminal_resets']==0 and result['resets']==0 and result['rollbacks']==1
    assert result['demo_exhausted']==1
    # Rejected autonomous output retains the accepted autonomous state.
    pool.update(torch.tensor([2]),q[:1],{},torch.tensor([True]),torch.tensor([False]),observed)
    assert not pool.supervised[2] and pool.age[2]==1
    assert pool.observation['current_q'][2].item()==59


def test_successor_never_enters_validation_split():
    pool=ParallelRolloutPool(reference(4),torch.tensor([0,1]),torch.tensor([1,2,3,-1]),1)
    pool.indices[:]=1
    pool.update(torch.tensor([0]),torch.tensor([[8.]]),{'contact':torch.ones(1,1,dtype=torch.bool)},
        torch.tensor([True]),torch.tensor([True]),observed)
    assert pool.indices.item()==1 and not pool.supervised.item()


def test_severe_or_contactless_output_has_gradient_before_rollback():
    pool=ParallelRolloutPool(reference(10),torch.arange(10),torch.arange(10),2)
    original={k:v.clone() for k,v in pool.observation.items()}
    old_indices=pool.indices.clone()
    output=torch.tensor([[51.],[53.]],requires_grad=True)
    output.square().mean().backward()
    assert output.grad.abs().sum()>0
    def forbidden(*args):raise AssertionError('Physically terminated output must not feed back')
    result=pool.update(torch.arange(2),output,{},torch.zeros(2,dtype=torch.bool),
                      torch.zeros(2,dtype=torch.bool),forbidden)
    assert result['resets']==0 and result['rollbacks']==2
    assert torch.equal(pool.indices,old_indices)
    for key in original: assert torch.equal(pool.observation[key],original[key])


def test_reset_gate_is_contact_and_physical_failure_not_endpoint_success():
    q=torch.zeros(6,36);q[:,3]=1;q[5,0]=float('nan')
    contacts=torch.zeros(6,6,dtype=torch.bool);contacts[:3,2]=True
    keep,reasons=physical_reset_masks(q,torch.tensor([0.,2.9,3.01,0.,4.,0.]),
        torch.zeros(6),torch.zeros(6),contacts)
    assert keep.tolist()==[True,True,False,False,False,False]
    assert reasons['severe_penetration'].tolist()==[False,False,True,False,True,False]
    assert reasons['no_contact'].tolist()==[False,False,False,True,False,False]
    assert reasons['invalid'].tolist()==[False,False,False,False,False,True]
    import pytest
    with pytest.raises(ValueError,match='actual Newton'):
        physical_reset_masks(q,torch.zeros(6),torch.zeros(6),torch.zeros(6),None)


def test_post_demo_labels_have_zero_gradient_but_geometry_still_trains():
    pose=torch.tensor([2.,3.],requires_grad=True)
    demo=(pose-100).square();geometry=pose.square()
    loss=demonstration_loss(demo,torch.tensor([True,False]))+geometry
    loss.sum().backward()
    assert pose.grad.tolist()==[-192.,6.]


def test_default_window_resamples_success_and_failure_on_exactly_attempt_64():
    bank=reference(5)
    pool=ParallelRolloutPool(bank,torch.tensor([0,1,2]),torch.full((5,),-1),2)
    pool.draw_references=lambda n: torch.tensor([1,2])[:n]
    original={k:v.clone() for k,v in pool.observation.items()}
    for attempt in range(1,65):
        slots,_,inputs=pool.batch(2)
        output=(inputs['current_q']+1).requires_grad_()
        output.square().sum().backward()
        assert output.grad.abs().sum()>0  # Also on the resampling boundary.
        result=pool.update(slots,output,{'contact':torch.ones(2,1,dtype=torch.bool)},
            torch.tensor([True,False]),torch.tensor([True,False]),observed)
        if attempt<64:
            assert pool.attempts.tolist()==[attempt,attempt]
            assert pool.age.tolist()==[attempt,0]
            assert pool.supervised.tolist()==[False,True]
            assert torch.equal(pool.observation['current_q'][1],original['current_q'][1])
            assert result['periodic_resamples']==0
    assert result['periodic_resamples']==result['resets']==result['timeout_resets']==2
    assert result['retries']==1 and result['rollbacks']==0
    assert pool.indices.tolist()==[1,2]
    for k in bank: assert torch.equal(pool.observation[k],bank[k][[1,2]])
    assert pool.age.tolist()==pool.attempts.tolist()==pool.retries.tolist()==[0,0]
    assert pool.supervised.all()
    state=pool.state_dict()
    assert state['schema']=='bounded_attempt_rollout_v5'
    assert state['resample_interval']==64 and not state['attempts'].any()


def test_resampling_is_per_slot_and_restores_progress_and_all_observations():
    bank={'current_q':torch.tensor([[0.,0.],[7.,8.],[90.,90.]]),
          'current_contact':torch.tensor([[True],[False],[True]]),
          'heightmap':torch.arange(12.).reshape(3,2,2)}
    pool=ParallelRolloutPool(bank,torch.tensor([0,1]),torch.full((3,),-1),2,
        max_episode_steps=2,global_directions=torch.tensor([[1.,0.],[0.,1.],[1.,0.]]))
    pool.draw_references=lambda n:torch.ones(n,dtype=torch.long)
    pool.attempts[:]=torch.tensor([1,0])
    pool.age[0]=7;pool.retries[0]=1;pool.supervised[0]=False
    pool.progress_history[0]=99
    untouched={k:v[1].clone() for k,v in pool.observation.items()}
    history=pool.progress_history[1].clone()
    def forbidden(*args):raise AssertionError('Rejected output must not be queried')
    result=pool.update(torch.tensor([0]),torch.tensor([[float('nan'),0.]]),{},
        torch.tensor([False]),torch.tensor([False]),forbidden)
    assert result['periodic_resamples']==1
    assert pool.indices[0]==1 and pool.supervised[0]
    assert pool.attempts.tolist()==[0,0]
    assert (pool.progress_history[0]==8).all()
    for k in bank:
        assert torch.equal(pool.observation[k][0],bank[k][1])
        assert torch.equal(pool.observation[k][1],untouched[k])
    assert torch.equal(pool.progress_history[1],history)
    pool.update(torch.tensor([1]),torch.zeros(1,2),{},torch.tensor([False]),
        torch.tensor([False]),forbidden)
    assert pool.attempts.tolist()==[0,1]


def test_reference_sampling_never_uses_validation_rows():
    pool=ParallelRolloutPool(reference(5),torch.tensor([1,3]),torch.full((5,),-1),2)
    assert torch.isin(pool.draw_references(100),torch.tensor([1,3])).all()


def test_disabled_post_demo_resamples_only_after_accepted_final_target():
    bank=reference(4)
    pool=ParallelRolloutPool(bank,torch.tensor([0,1]),torch.tensor([1,2,3,-1]),1,
        allow_post_demo=False,max_episode_steps=3)
    pool.indices[:]=1  # Successor is outside the training split: chain ends here.
    pool.draw_references=lambda n:torch.zeros(n,dtype=torch.long)
    for complete,physical in ((False,True),(True,False),(True,True)):
        result=pool.update(torch.tensor([0]),torch.tensor([[9.]]),
            {'contact':torch.ones(1,1,dtype=torch.bool)},torch.tensor([complete]),
            torch.tensor([physical]),observed)
        assert pool.supervised.all() and result['autonomous_states']==0
        if not (complete and physical):
            assert pool.indices.item()==1 and result['resets']==0
    assert result['terminal_resets']==result['resets']==1
    assert result['periodic_resamples']==0  # Boundary has one reset reason.
    assert pool.indices.item()==0 and pool.attempts.item()==pool.age.item()==0
    assert torch.equal(pool.observation['current_q'],bank['current_q'][:1])
    assert pool.state_dict()['allow_post_demo'] is False


def test_trainer_counts_autonomous_inputs_before_feedback_without_reading_pool_field():
    """Exercise the trainer's accounting with real pool commits and rollbacks."""
    import ast
    from pathlib import Path
    from generator import parallel_rollout_pool

    path = Path(parallel_rollout_pool.__file__).with_name('train_full1000_position.py')
    tree = ast.parse(path.read_text())
    accumulation = next(node for node in ast.walk(tree)
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Name)
        and node.iter.id == 'pool_counts')
    code = compile(ast.Module(body=[accumulation], type_ignores=[]), str(path), 'exec')
    pool = ParallelRolloutPool(reference(2), torch.arange(2), torch.full((2,), -1), 2)
    pool.indices[:] = torch.arange(2)
    pool.observation = {k:v.clone() for k,v in pool.reference.items()}
    counts = {'generated_inputs': torch.tensor(0), 'autonomous_inputs': torch.tensor(0)}
    for completed in (torch.tensor([True, False]), torch.tensor([False, False])):
        slots, ids, inputs = pool.batch(2)
        counts['generated_inputs'] += (pool.age[slots] > 0).sum()
        counts['autonomous_inputs'] += (~pool.supervised[slots]).sum()
        feedback = pool.update(slots, inputs['current_q'] + 1,
            {'contact':torch.ones(2,1,dtype=torch.bool)}, completed,
            torch.ones(2,dtype=torch.bool), observed)
        for key in feedback:
            counts.setdefault(key, torch.tensor(0))
        exec(code, {'pool_counts': counts, 'feedback': feedback})
    assert counts['autonomous_inputs'] == 1  # The exhausting input was still supervised.
    assert counts['generated_inputs'] == 1
    assert counts['autonomous_states'] == 2  # Post-commit states, a different measure.
    assert counts['accepted_transitions'] == 1
    assert counts['rollbacks'] == 3
