"""Training successes supply successor entries without a second producer pass."""
import ast
from pathlib import Path
from types import SimpleNamespace
import torch

ROOT=Path(__file__).resolve().parents[1]
HANDOFF=ROOT/'packages/supermimic/supermimic/handoff'


def test_training_step_preserves_successful_terminal_payload_before_reset():
    source=ast.parse((HANDOFF/'task_base.py').read_text())
    cls=next(n for n in source.body if isinstance(n,ast.ClassDef) and n.name=='PoolTask')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='step')
    history=torch.randn(4,5,7);active=torch.ones(4,6,dtype=torch.bool);f=torch.arange(4)
    transition=dict(goal_index=torch.arange(4),done=torch.tensor([True,True,False,True]),term=torch.tensor([False,True,False,False]),s1=torch.randn(4,10),a=torch.randn(4,29),next_xi=torch.randn(4,8))
    class Base:
        def step(self):
            assert self.in_step
            self.captured=(torch.arange(4),history.clone(),active.clone(),f.clone(),torch.tensor([0,0,1,2]),torch.tensor([0,1,1,1]))
            return transition
    test_cls=ast.ClassDef(name='Task',bases=[ast.Name(id='Base',ctx=ast.Load())],keywords=[],body=[method],decorator_list=[])
    scope=dict(Base=Base,torch=torch);exec(compile(ast.fix_missing_locations(ast.Module(body=[test_cls],type_ignores=[])),str(HANDOFF/'task_base.py'),'exec'),scope)
    task=scope['Task']();task.evaluating=False;task.successor=torch.tensor([1,2,3,-1]);task.iteration=9
    calls=[];task.pool=SimpleNamespace(add=lambda *args:calls.append(args))
    result=task.step();assert result is transition and not task.in_step
    assert len(calls)==1
    dest,payload,iteration=calls[0];assert dest==1 and iteration==9
    for name,expected in dict(s=transition['s1'][:1],a_prev=transition['a'][:1],xi=transition['next_xi'][:1],history=history[:1],active=active[:1],f=f[:1],origin=torch.tensor([0]),depth=torch.tensor([1])).items():
        torch.testing.assert_close(payload[name],expected,rtol=0,atol=0)


def test_training_loop_has_no_extra_reference_rollout():
    tree=ast.parse((HANDOFF/'train.py').read_text())
    loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='iteration')
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='populate_reference' for n in ast.walk(loop))
    # The startup bootstrap remains available to seed a new run.
    assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='populate_reference' for n in ast.walk(tree))
