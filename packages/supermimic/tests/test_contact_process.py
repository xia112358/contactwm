import torch
from supermimic.contact_process import native_patch,material_patch,process_loss


def run(q,active=None,initial=False):
    q=torch.tensor(q,dtype=torch.float64)[:,None]
    a=torch.ones_like(q,dtype=torch.bool) if active is None else torch.tensor(active)[:,None]
    return process_loss(q,a,torch.tensor([initial]))


def test_graze_is_not_persistent_and_sliding_history_not_erased():
    assert run([1.])['score']<run([1.]*30)['score']
    assert run([.05]*10+[1.]*20)['score']<run([1.]*30)['score']
    assert run([1.]*10+[.05]*20)['score']<run([1.]*30)['score']


def test_reset_support_excluded_and_recontact_does_not_clear_history():
    assert run([1.]*4,initial=True)['score']==0
    assert run([.1,0,1],[True,False,True])['score']<run([1,1,1])['score']
    assert run([1,0,0],[True,False,False])['score']<run([1])['score']
    assert run([0,1],[False,True],initial=True)['score']>0


def test_native_patch_duplicate_invariance_and_no_point_average():
    pos=torch.tensor([[-.2,0,0],[.2,0,0]],requires_grad=True);g=torch.zeros(2,dtype=torch.long)
    target=torch.zeros(1,1,3);radius=torch.full((1,1),.05)
    q=native_patch(pos,g,target,radius)
    assert q<.1 # averaging the positions would incorrectly give perfect coverage
    assert torch.equal(q,native_patch(pos.repeat(3,1),g.repeat(3),target,radius))
    q.sum().backward();assert torch.isfinite(pos.grad).all() and pos.grad.abs().sum()>0
    assert native_patch(pos[:0],g[:0],target,radius)==0


def test_sample_change_bound_and_gradients():
    torch.manual_seed(3)
    q=torch.rand(100,2,dtype=torch.float64,requires_grad=True)
    out=process_loss(q,torch.ones_like(q,dtype=torch.bool),torch.zeros(2,dtype=torch.bool))
    delta=torch.diff(torch.cat((torch.ones(1,2),out['loss'])),dim=0)
    assert delta.abs().max()<=.002/.04
    out['loss'][-1].sum().backward();assert torch.isfinite(q.grad).all() and (q.grad<0).all()


def test_fixed_nodes_permutation_duplicate_and_missing():
    nodes=torch.tensor([[[-.1,0,0],[.1,0,0]]]);local=nodes[0];g=torch.zeros(2,dtype=torch.long)
    patch=material_patch(local,g,nodes,1)
    assert torch.equal(patch,material_patch(local.flip(0).repeat(2,1),g.repeat(2),nodes,1))
    assert material_patch(local[:0],g[:0],nodes,1).sum()==0
