import numpy as np
import pytest
from somaforge_core.event_acceptance import build_contract, audit_events, optimization_events
from somaforge_core.event_acceptance import observed_source_contract


def action(kind, start=0, end=19):
    before=[-1]*6;after=[-1]*6;touch=[]
    if kind in ('keep','release'): before[0]=1
    if kind in ('keep','establish'): after[0]=1
    if kind=='establish': touch=[dict(part_index=0,frame=end,surface=1)]
    return dict(start_frame=start,end_frame=end,source_surfaces=before,target_surfaces=after,touchdown_events=touch,
                persistent_parts=['left_foot'] if kind=='keep' else [],
                release_events=[dict(part_index=0,scope='whole_endpoint')] if kind=='release' else [])


def evidence(n=24):
    mask=np.zeros((n,6),bool);mask[:,0]=True
    return mask,np.where(mask,1,-1)


def test_keep_tolerates_short_bounded_hole_not_true_departure():
    mask,faces=evidence();contract=build_contract(mask,faces,[action('keep',end=23)],50)
    candidate=mask.copy();candidate[8:11,0]=False
    report,holes=audit_events(contract,candidate,np.where(candidate,1,-1))
    assert report['passed'] and holes[8:11,0].all()
    assert not candidate[8:11,0].any()
    candidate[11,0]=False
    assert not audit_events(contract,candidate,np.where(candidate,1,-1))[0]['passed']


def test_source_only_calibrates_tolerance_not_phase_template():
    mask,faces=evidence();mask[6:12,0]=False;faces=np.where(mask,1,-1)
    contract=build_contract(mask,faces,[action('keep',end=23)],50)
    assert contract['policy']['max_dropout_frames']==6
    assert 'phases' not in contract
    candidate,_=evidence();candidate[14:18,0]=False
    assert audit_events(contract,candidate,np.where(candidate,1,-1))[0]['passed']


def test_wrong_surface_is_not_execution_noise():
    mask,faces=evidence();contract=build_contract(mask,faces,[action('keep',end=23)],50)
    faces[10:12,0]=2
    assert not audit_events(contract,mask,faces)[0]['passed']


def test_establish_requires_stable_contact_near_completion_not_exact_frame():
    mask,faces=evidence();contract=build_contract(mask,faces,[action('establish',end=19),action('keep',start=19,end=23)],50)
    # Evaluate establishment independently of the subsequent support requirement.
    contract['actions']=contract['actions'][:1]
    candidate=np.zeros_like(mask);candidate[17:20,0]=True
    assert audit_events(contract,candidate,np.where(candidate,1,-1))[0]['passed']
    candidate[:]=False;candidate[19,0]=True
    assert not audit_events(contract,candidate,np.where(candidate,1,-1))[0]['passed']
    candidate[:]=False;candidate[2:5,0]=True
    assert not audit_events(contract,candidate,np.where(candidate,1,-1))[0]['passed']


def test_release_requires_actual_absence():
    mask,faces=evidence();contract=build_contract(mask,faces,[action('release',end=23)],50)
    assert not audit_events(contract,mask,faces)[0]['passed']
    mask[20:,0]=False
    assert audit_events(contract,mask,np.where(mask,1,-1))[0]['passed']
    assert optimization_events(contract)[0]['release_events']


def test_unrequested_extra_contact_does_not_fail_phase_count():
    mask,faces=evidence();contract=build_contract(mask,faces,[action('keep',end=23)],50)
    mask[4:12,1]=True;faces[4:12,1]=0
    assert audit_events(contract,mask,faces)[0]['passed']


def test_cropped_gap_cannot_supply_boundary_evidence():
    mask,faces=evidence();events=[action('keep',end=8),action('keep',start=12,end=23)]
    contract=build_contract(mask,faces,events,50);mask[8:13,0]=False
    assert not audit_events(contract,mask,np.where(mask,1,-1))[0]['passed']


def test_old_phase_contract_is_rejected():
    mask,faces=evidence();contract=build_contract(mask,faces,[action('keep',end=23)],50)
    contract['schema']='source_contact_episode_acceptance_v1'
    with pytest.raises(ValueError,match='mismatch'): audit_events(contract,mask,faces)


def test_same_surface_without_explicit_support_never_becomes_keep():
    from somaforge_core.event_acceptance import action_requirements
    e=action('keep');e['persistent_parts']=[]
    assert action_requirements(e)==[]
    del e['persistent_parts']
    with pytest.raises(ValueError,match='Explicit'):action_requirements(e)


def test_source_binding_ignores_wrong_authored_roles_and_preserves_clock():
    mask, faces = evidence(24)
    mask[:, 0] = False
    mask[8:, 1] = True
    faces = np.where(mask, 0, -1)
    clock = [dict(action('keep', end=12), segment_id='edit_a'),
             dict(action('establish', start=12, end=23), segment_id='edit_b')]
    events, contract = observed_source_contract(dict(contact_part_mask=mask, contact_surface=faces, fps=50), clock)
    assert [(e['segment_id'], e['start_frame'], e['end_frame']) for e in events] == [
        ('edit_a', 0, 12), ('edit_b', 12, 23)]
    assert events[0]['touchdown_events'] == [dict(frame=8, part_index=1, part='right_foot', surface=0)]
    assert events[1]['persistent_parts'] == ['right_foot']
    assert contract['source_self_audit']['passed']
    candidate = mask.copy(); candidate[12:, 1] = False
    assert not audit_events(contract, candidate, np.where(candidate, 0, -1))[0]['passed']


def test_source_binding_preserves_removed_blocks_and_absolute_arrival_frames():
    mask, faces = evidence(24)
    mask[13:16, 0] = False; faces = np.where(mask, 1, -1)
    clock = [dict(action('keep', end=8), segment_id='before_cut'),
             dict(action('keep', start=13, end=23), segment_id='after_cut')]
    events, contract = observed_source_contract(dict(contact_part_mask=mask, contact_surface=faces, fps=50), clock)
    assert contract['retained_blocks'] == [(0, 9), (13, 24)]
    assert events[1]['touchdown_events'][0]['frame'] == 16
    assert events[1]['persistent_parts'] == []
    assert len(events) == len(clock)


@pytest.mark.parametrize('clock', [[], [dict(start_frame=0, end_frame=12), dict(start_frame=8, end_frame=23)],
    [dict(start_frame=0, end_frame=12, segment_id='same'), dict(start_frame=12, end_frame=23, segment_id='same')]])
def test_source_binding_rejects_invalid_clocks(clock):
    mask, faces = evidence()
    with pytest.raises(ValueError, match='clock'):
        observed_source_contract(dict(contact_part_mask=mask, contact_surface=faces, fps=50), clock)


def test_source_binding_reports_insufficient_evidence_without_dropping_action():
    mask, faces = evidence(24); mask[:, 0] = False; mask[-1, 0] = True
    with pytest.raises(ValueError, match='Source observations'):
        observed_source_contract(dict(contact_part_mask=mask, contact_surface=np.where(mask, 1, -1), fps=50),
                                 [dict(start_frame=0, end_frame=23)])
