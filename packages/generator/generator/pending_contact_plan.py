"""Store issued intent across execution repairs; never synthesize contact truth."""
import torch


def planner_weights(repair_mask):
    fresh = (~repair_mask).float()
    # Preserve planner scale on the fresh subset; all-repair batches yield zero.
    return fresh / fresh.mean().clamp_min(1 / max(1, len(fresh)))


class PendingContactPlans:
    def __init__(self, size, device):
        self.mask = torch.zeros(size, dtype=torch.bool, device=device)
        self.role = torch.zeros(size, 6, dtype=torch.long, device=device)
        self.points = torch.zeros(size, 6, 3, device=device)
        self.regions = torch.zeros(size, 6, 4, dtype=torch.bool, device=device)
        self.surface = torch.full((size, 6), -1, dtype=torch.long, device=device)
        self.point_id = torch.full((size, 6), -1, dtype=torch.long, device=device)

    def batch(self, slots):
        return dict(repair_mask=self.mask[slots], repair_role=self.role[slots],
                    repair_points_world=self.points[slots], repair_regions=self.regions[slots],
                    repair_surface=self.surface[slots], repair_point_id=self.point_id[slots])

    @torch.no_grad()
    def update(self, slots, prediction, points_world, retry):
        self.mask[slots] = retry
        self.role[slots] = prediction.conditioned_role.detach()
        self.points[slots] = points_world.detach()
        self.regions[slots] = prediction.planned_regions.detach()
        if getattr(prediction, 'conditioned_surface', None) is not None:
            self.surface[slots] = prediction.conditioned_surface.detach()
            self.point_id[slots] = prediction.conditioned_point_id.detach()

    def state_dict(self):
        return {k: v.detach().cpu() for k, v in vars(self).items()}
