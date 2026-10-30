import torch


def test_region_motion_allows_rotation_near_support_but_detects_translation():
    from somaforge_core.contact_motion import contact_region_statistics
    # Three near-pivot samples, two distant moving samples. A median of a
    # whole rigid body's arbitrary cloud is not a support displacement.
    group=torch.zeros(5,dtype=torch.long)
    rotating=torch.tensor([0., .0001, .0002, .01, .02],requires_grad=True)
    stats=contact_region_statistics(rotating,group,2)
    torch.testing.assert_close(stats['lower_quartile'][0],torch.tensor(.0001))
    assert stats['pivot'][0]==0
    assert stats['maximum'][0]>.019 and torch.isnan(stats['pivot'][1])
    translated=contact_region_statistics(rotating+.008,group,2)
    assert translated['pivot'][0]>=.008
    translated['pivot'][0].backward()
    assert rotating.grad.isfinite().all() and rotating.grad.abs().sum()>0


def test_isolated_stationary_sample_does_not_erase_region_translation():
    from somaforge_core.contact_motion import contact_region_statistics
    stats=contact_region_statistics(torch.tensor([0., .01, .01, .01, .01]),
                                    torch.zeros(5,dtype=torch.long),1)
    assert stats['pivot'][0]==0 and stats['lower_quartile'][0]>.009
