"""Exact per-segment parameter gradients from one independent-sample backward."""
import torch
from torch import nn

class GroupedActorGradients:
    def __init__(self,actor,segments,*,embedding=None):
        self.actor=actor;self.segments=segments;self.embedding=embedding
        self.ids,self.inverse,self.counts=segments.unique(sorted=True,return_inverse=True,return_counts=True)
        self.batch=len(segments);self.groups=len(self.ids)
        width=int(self.counts.max())
        self.indices=torch.zeros((self.groups,width),dtype=torch.long,device=segments.device)
        self.valid=torch.zeros((self.groups,width),dtype=torch.bool,device=segments.device)
        for g in range(self.groups):
            slots=(self.inverse==g).nonzero().flatten();self.indices[g,:len(slots)]=slots;self.valid[g,:len(slots)]=True
        self.gradients={p:torch.zeros((self.groups,*p.shape),device=p.device,dtype=p.dtype) for p in actor.parameters()}
        self.handles=[]
    def __enter__(self):
        for module in self.actor.modules():
            if isinstance(module,nn.Linear):self.handles.append(module.register_forward_hook(self._linear))
        if self.embedding is not None:
            self.handles.append(self.embedding[0].register_forward_hook(self._embedding))
        return self
    def _linear(self,module,args,output):
        if not torch.is_grad_enabled() or not output.requires_grad:return
        x=args[0].detach()
        def collect(dy):
            with torch.no_grad():
                xx=x[self.indices].flatten(1,-2)
                yy=dy[self.indices]*self.valid.reshape(*self.valid.shape,*([1]*(dy.ndim-1)))
                yy=yy.flatten(1,-2)
                self.gradients[module.weight].add_(torch.bmm(yy.transpose(1,2),xx))
                if module.bias is not None:self.gradients[module.bias].add_(yy.sum(1))
        output.register_hook(collect)
    def _embedding(self,module,args,output):
        if not torch.is_grad_enabled() or not output.requires_grad:return
        def collect(dy):
            with torch.no_grad():
                value=torch.zeros_like(self.gradients[self.embedding[1]][:,0])
                value.index_add_(0,self.inverse,dy)
                self.gradients[self.embedding[1]].add_(value[:,None])
        output.register_hook(collect)
    def matrix(self):
        # The combined objective is a batch mean; recover each group's mean.
        scale=self.batch/self.counts.to(next(iter(self.gradients.values())).dtype)
        return torch.cat([v.flatten(1) for v in self.gradients.values()],1)*scale[:,None]
    def __exit__(self,*exc):
        for handle in self.handles:handle.remove()
