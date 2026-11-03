from supermimic.repeated_evaluation import summarize
import pytest


def result(passed, ratio):
    return dict(n_failed=not passed, norm_return=ratio, norm_return_framew=ratio, mean_length_ratio=ratio,
        per_clip_lengths={'step':dict(steps=100*ratio, seconds=2*ratio, reference_steps=100,
            reference_seconds=2, length_ratio=ratio, norm_return=ratio, failed=not passed)})


def test_repeats_preserve_variation_and_per_motion_success_count():
    s=summarize([result(True,1.),result(False,.4),result(True,1.)],[0,1,2])
    assert s['passed_stats']['values']==[1,0,1]
    assert s['passed_stats']['mean']==pytest.approx(2/3)
    assert s['n_failed']==pytest.approx(1/3)
    assert s['mean_length_ratio']==pytest.approx(.8)
    assert s['per_clip_lengths']['step']['successes']==2
    assert s['per_clip_lengths']['step']['length_ratio_stats']['minimum']==.4


def test_mismatched_coverage_rejected():
    bad=result(True,1.);bad['per_clip_lengths']={}
    with pytest.raises(ValueError):summarize([result(True,1.),bad],[0,1])
