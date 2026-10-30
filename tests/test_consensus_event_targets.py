import numpy as np

from somaforge_core.consensus_event_targets import event_targets
from somaforge_core.observed_event_consensus import build_consensus, audit_consensus


def test_population_episode_survives_without_a_single_frame_majority():
    masks = np.zeros((5, 40, 6), bool)
    surfaces = np.full(masks.shape, -1)
    depth = np.full(masks.shape, np.inf)
    for i, start in enumerate((10, 14, 18)):
        masks[i, start:start+5, 2] = True
        surfaces[i, start:start+5, 2] = 1
        depth[i, start:start+5, 2] = .017+i*.001
    consensus = build_consensus(masks, surfaces)
    assert len(consensus['events']) == 1
    assert not (masks.sum(0) > len(masks)/2).any()
    reports = [audit_consensus(consensus,m,s) for m,s in zip(masks,surfaces)]
    guide = event_targets(consensus,reports,masks,surfaces,depth)
    assert guide['wanted'][14:19,2].all()
    assert not guide['release'][:,2].any()
    np.testing.assert_allclose(guide['target_depth'][14:19,2],.018)
    assert not guide['contact_truth']


def test_shared_release_is_not_lost_to_split_frame_votes():
    masks=np.zeros((3,30,6),bool);surfaces=np.full(masks.shape,-1)
    masks[:,:8,0]=True;masks[:,13:,0]=True;surfaces[masks]=0
    consensus=build_consensus(masks,surfaces)
    reports=[audit_consensus(consensus,m,s) for m,s in zip(masks,surfaces)]
    guide=event_targets(consensus,reports,masks,surfaces)
    assert guide['release'][8:13,0].all()
    assert not np.any(guide['wanted'] & guide['release'])
    assert guide['wanted'][:8,0].all() and guide['wanted'][13:,0].all()
    # Population noncontact does not establish a required shared release.
    consensus['events'][1]['release_votes']=2
    guide=event_targets(consensus,reports,masks,surfaces)
    assert not guide['release'].any()


def test_each_execution_has_one_depth_vote_despite_episode_duration():
    masks=np.zeros((3,30,6),bool);surfaces=np.full(masks.shape,-1)
    depth=np.full(masks.shape,np.inf)
    for i,(end,value) in enumerate(((29,.010),(15,.018),(14,.019))):
        masks[i,3:end,2]=True;surfaces[i,3:end,2]=1;depth[i,3:end,2]=value
    consensus=build_consensus(masks,surfaces)
    reports=[audit_consensus(consensus,m,s) for m,s in zip(masks,surfaces)]
    guide=event_targets(consensus,reports,masks,surfaces,depth)
    np.testing.assert_allclose(guide['target_depth'][guide['wanted']],.018)
