"""Actor LR reductions driven only by recent native training attempts."""
from collections import deque
import torch


class NativeSuccessLR:
    def __init__(self, optimizer, *, window=3, patience=3, factor=.5, minimum=3e-7):
        self.recent=deque(maxlen=window)
        self.scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,mode='max',factor=factor,patience=patience-1,
            threshold=.01,threshold_mode='abs',min_lr=minimum)

    def observe(self, success_rate):
        self.recent.append(float(success_rate))
        score=sum(self.recent)/len(self.recent)
        before=self.scheduler.optimizer.param_groups[0]['lr']
        if len(self.recent)==self.recent.maxlen:
            self.scheduler.step(score)
        after=self.scheduler.optimizer.param_groups[0]['lr']
        return dict(recent_success_rate=score,window_samples=len(self.recent),
                    learning_rate=after,reduced=after<before)

    def state_dict(self):
        return dict(recent=list(self.recent),scheduler=self.scheduler.state_dict())

    def load_state_dict(self, state):
        self.recent.extend(state['recent'])
        self.scheduler.load_state_dict(state['scheduler'])
