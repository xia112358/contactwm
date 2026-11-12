"""Shared analytic static-box distance field; optimization geometry only."""
import torch


def box_distance_local(local, half):
    """Exact signed Euclidean distance in box-local coordinates."""
    distance=local.abs()-half
    return distance.clamp_min(0).norm(dim=-1)+distance.amax(-1).clamp_max(0)


def box_sdf(points, center, basis, half):
    """Exact OBB SDF, preserving the recovery_field query contract."""
    return box_distance_local((points-center)@basis,half)
