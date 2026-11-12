"""Exact per-segment parameter gradients from one independent-sample backward."""
import torch
from torch import nn

class GroupedActorGradients:
    def __init__(self,actor,segments,*,embedding=None,contiguous_group_size=None):
        self.actor=actor;self.segments=segments;self.embedding=embedding
        self.batch=len(segments)
        self.contiguous_width=None
        if contiguous_group_size is not None:
            # The caller constructs segments as arange(batch) // group_size.
            # Build the padded gather directly, without unique/nonzero or GPU
            # synchronization for every group.
            width=contiguous_group_size
            if width < 1 or not self.batch:
                raise ValueError('A nonempty batch and positive group size are required')
            if self.batch%width==0:self.contiguous_width=width
            self.groups=(self.batch+width-1)//width
            self.ids=torch.arange(self.groups,device=segments.device)
            self.inverse=torch.arange(self.batch,device=segments.device)//width
            self.counts=torch.full((self.groups,),width,dtype=torch.long,device=segments.device)
            self.counts[-1]=self.batch-(self.groups-1)*width
            positions=torch.arange(self.groups*width,device=segments.device).reshape(self.groups,width)
            self.valid=positions<self.batch
            self.indices=positions.clamp_max(self.batch-1)
        else:
            self.ids,self.inverse,self.counts=segments.unique(sorted=True,return_inverse=True,return_counts=True)
            self.groups=len(self.ids)
            width=int(self.counts.max())
            self.indices=torch.zeros((self.groups,width),dtype=torch.long,device=segments.device)
            self.valid=torch.zeros((self.groups,width),dtype=torch.bool,device=segments.device)
            for g in range(self.groups):
                slots=(self.inverse==g).nonzero().flatten();self.indices[g,:len(slots)]=slots;self.valid[g,:len(slots)]=True
        self.gradients={p:torch.zeros((self.groups,*p.shape),device=p.device,dtype=p.dtype) for p in actor.parameters()}
        self.handles=[]
        self.outputs=[]
    def __enter__(self):
        for module in self.actor.modules():
            if isinstance(module,nn.Linear):self.handles.append(module.register_forward_hook(self._linear))
        if self.embedding is not None:
            self.handles.append(self.embedding[0].register_forward_hook(self._embedding))
        return self
    def _linear(self,module,args,output):
        if not torch.is_grad_enabled() or not output.requires_grad:return
        self.outputs.append(output)
        x=args[0].detach()
        def collect(dy):
            with torch.no_grad():
                if self.contiguous_width is not None:
                    # Full consecutive groups are already in gather order.
                    # Reshape gives the same per-group matrix without copying
                    # the activations and adjoints at every BPTT step.
                    xx=x.reshape(self.groups,self.contiguous_width,*x.shape[1:]).flatten(1,-2)
                    yy=dy.reshape(self.groups,self.contiguous_width,*dy.shape[1:]).flatten(1,-2)
                else:
                    xx=x[self.indices].flatten(1,-2)
                    yy=dy[self.indices]*self.valid.reshape(*self.valid.shape,*([1]*(dy.ndim-1)))
                    yy=yy.flatten(1,-2)
                self.gradients[module.weight].baddbmm_(yy.transpose(1,2),xx)
                if module.bias is not None:self.gradients[module.bias].add_(yy.sum(1))
        output.register_hook(collect)
    def _embedding(self,module,args,output):
        if not torch.is_grad_enabled() or not output.requires_grad:return
        self.outputs.append(output)
        def collect(dy):
            with torch.no_grad():
                if self.contiguous_width is not None:
                    value=dy.reshape(self.groups,self.contiguous_width,*dy.shape[1:]).sum(1)
                else:
                    value=torch.zeros_like(self.gradients[self.embedding[1]][:,0])
                    value.index_add_(0,self.inverse,dy)
                self.gradients[self.embedding[1]].add_(value[:,None])
        output.register_hook(collect)
    def matrix(self):
        # The combined objective is a batch mean; recover each group's mean.
        scale=self.batch/self.counts.to(next(iter(self.gradients.values())).dtype)
        return torch.cat([v.flatten(1) for v in self.gradients.values()],1)*scale[:,None]
    def backward(self,loss):
        # Hooks already form the parameter gradients per group. Asking autograd
        # for parameter gradients as well repeats every Linear weight reduction
        # and then discards its result. Request the observed activations instead;
        # their adjoints still traverse recurrent connections and fire the hooks.
        torch.autograd.grad(loss,tuple(dict.fromkeys(self.outputs)),allow_unused=True)
    def __exit__(self,*exc):
        for handle in self.handles:handle.remove()
        self.outputs.clear()
