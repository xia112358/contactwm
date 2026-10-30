from somaforge_core.contact_consensus_selection import filter_short_unrepeatable


def timeline(active, part=0, surface=8, n=20):
    return [[dict(part=part,surface=surface,position_w=[1,2,3])] if f in active else [] for f in range(n)]


def test_rejects_minority_graze_but_not_repeatable_short_contact():
    raw=timeline([8])
    kept,bad,report=filter_short_unrepeatable(raw,[timeline([8]),timeline([]),timeline([])],fps=50)
    assert kept[8]==[] and bad[8]==raw[8] and report[0]['votes']==1
    kept,bad,_=filter_short_unrepeatable(raw,[timeline([7]),timeline([9]),timeline([])],fps=50)
    assert kept==raw and not any(bad)  # tolerant of timing, no invented frames


def test_different_face_or_part_cannot_vote():
    raw=timeline([8])
    kept,_,report=filter_short_unrepeatable(raw,[timeline([8],surface=2),timeline([8],part=1)],fps=50)
    assert not any(kept) and report[0]['votes']==0


def test_keep_long_runs_and_repeatable_truncated_start():
    raw=timeline(list(range(4))+list(range(9,20)))
    kept,bad,_=filter_short_unrepeatable(raw,[timeline(range(4))]*3,fps=50)
    assert kept==raw and not any(bad)


def test_multiple_samples_do_not_multiply_rollout_votes():
    raw=timeline([8]); source=timeline([8]); source[8]*=100
    kept,_,report=filter_short_unrepeatable(raw,[source,timeline([]),timeline([])],fps=50)
    assert not any(kept) and report[0]['votes']==1
