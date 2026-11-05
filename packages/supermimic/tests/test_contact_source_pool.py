import json,torch,pytest
from supermimic.contact_source_pool import successful_source_pool


def capture(root,seed,success,term=False):
    p=root/f'capture-seed{seed}';p.mkdir();item=dict(motion=0,name='motion',successful_rollout=success)
    data={k:torch.zeros(5,2) for k in ('s','s1','a','a_prev','xi','next_xi')};data.update(f=torch.arange(5),ep_step=torch.arange(5),term=torch.tensor([0,0,0,0,term],dtype=torch.bool))
    torch.save(dict(data=data,report=item),p/'motion-000.pt')
    (p/'report.json').write_text(json.dumps(dict(motions=[item],checkpoint_sha256='a',task_manifest_sha256='b',stable_support_contract={},substep_support_contract={})))


def test_failed_trial_cannot_replace_a_successful_whole_source(tmp_path):
    capture(tmp_path,0,False);capture(tmp_path,1,True)
    (tmp_path/'capture-seed1.log').write_text('capture output')
    p=successful_source_pool(tmp_path)
    assert len(p['selected'])==1 and 'seed1' in p['selected'][0]['source']
    assert not p['attempts'][0]['success']


def test_missing_success_is_a_coverage_error_not_partial_training(tmp_path):
    capture(tmp_path,0,False)
    with pytest.raises(ValueError,match='Missing successful'):successful_source_pool(tmp_path)


def test_report_cannot_hide_native_termination(tmp_path):
    capture(tmp_path,0,True,term=True)
    with pytest.raises(ValueError,match='Missing successful'):successful_source_pool(tmp_path)



def test_every_successful_execution_enters_pool(tmp_path):
    capture(tmp_path,0,True);capture(tmp_path,1,True);capture(tmp_path,2,False);capture(tmp_path,3,True)
    pool=successful_source_pool(tmp_path)
    assert len(pool['selected'])==3
    assert all(e['success'] for e in pool['selected'])


def test_existing_consensus_rejects_incidental_support_and_matches_delays():
    import numpy as np
    from somaforge_core.observed_event_consensus import consensus_from_sequences
    sequences=[];masks=[]
    for index in range(3):
        sequence=[[] for _ in range(6)];mask=np.zeros((40,6),bool)
        for part,start,stop in [(0,10+index,25+index)]+([(1,2,5)] if index==0 else []):
            sequence[part].append(dict(part=part,surface=3,kind='establish',interval=[start,stop],stable_evidence=[start,start+1]))
            mask[start:stop,part]=True
        sequences.append(sequence);masks.append(mask)
    result=consensus_from_sequences(sequences,masks,stable_frames=1,max_dropout_frames=0)
    assert len(result['events'])==1 and result['events'][0]['votes']==3
    assert result['events'][0]['interval']==[11.,26.]
    assert result['minority_episodes'][0]['votes']==1
    assert len(result['events'][0]['observations'])==3


def test_existing_cloud_aggregation_weights_trials_not_sample_count():
    import numpy as np
    from somaforge_core.contact_aggregation import equal_demo_patch_cloud
    clouds=[np.zeros((1000,3)),np.ones((2,3)),np.full((5,3),1.1)]
    result=equal_demo_patch_cloud(clouds)
    assert result['representative_index']==1
    weights=result['point_weights'];offset=0
    for cloud in clouds:
        assert np.isclose(weights[offset:offset+len(cloud)].sum(),1/3)
        offset+=len(cloud)
    assert np.array_equal(result['representative_cloud'],clouds[1])


def test_duplicate_execution_cannot_receive_two_votes(tmp_path):
    capture(tmp_path,0,True)
    (tmp_path/'capture-seed1').symlink_to(tmp_path/'capture-seed0',target_is_directory=True)
    with pytest.raises(ValueError,match='Duplicate execution'):successful_source_pool(tmp_path)
