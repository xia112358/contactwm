"""Hard progress gate in a fixed task direction, independent of robot yaw."""
import math


class ForwardProgressGate:
    def __init__(self, initial_xy, direction_xy, *, window, minimum_m):
        norm = math.hypot(*direction_xy)
        if not math.isfinite(norm) or norm < 1e-8:
            raise ValueError('A finite nonzero global task direction is required')
        if window < 1 or not math.isfinite(minimum_m) or minimum_m <= 0:
            raise ValueError('Positive progress window and minimum required')
        self.origin = tuple(float(v) for v in initial_xy)
        self.direction = tuple(float(v)/norm for v in direction_xy)
        self.window, self.minimum = window, minimum_m
        self.best = [0.0]

    def update(self, xy):
        projection = sum((float(x)-o)*d for x,o,d in zip(xy,self.origin,self.direction,strict=True))
        if not math.isfinite(projection):
            raise ValueError('Nonfinite progress observation')
        self.best.append(max(self.best[-1], projection))
        ready = len(self.best) > self.window
        gain = self.best[-1]-self.best[-1-self.window] if ready else None
        return dict(global_progress_m=projection, furthest_progress_m=self.best[-1],
                    progress_window_gain_m=gain, progress_window_steps=self.window,
                    progress_minimum_m=self.minimum,
                    forward_progress_valid=not ready or gain >= self.minimum)
