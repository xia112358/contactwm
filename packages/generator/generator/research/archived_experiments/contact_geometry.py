"""Preserved scene geometry used by contact replay diagnostics."""
import torch

COLLISION_TOLERANCE_M = 0.002

def scene_faces(scene):
    (center, rotation, half, ground) = (scene[k] for k in ('box_center', 'box_rotation', 'box_half_extents', 'ground_height'))
    top = torch.cat((center + rotation[:, :, 2] * half[:, 2:3], rotation[:, :, 2], rotation[:, :, 0], rotation[:, :, 1], half[:, :2], torch.zeros_like(ground[:, None])), -1)
    floor = torch.zeros_like(top)
    floor[:, 2] = ground
    floor[:, 5] = 1
    floor[:, 6] = 1
    floor[:, 10] = 1
    floor[:, 14] = 1
    return torch.stack((floor, top), 1)
