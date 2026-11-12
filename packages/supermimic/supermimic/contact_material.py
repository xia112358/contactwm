"""Fixed material correspondences; geometry is an objective, never contact detection."""
import torch

CONTRACT = dict(version='stable_hand_contact_points_v6', points_per_part=4, material_parts=[0,1,4,5], point_parts=[2,3],
    error='feet: four fixed-point RMS; knees: one fixed local material point; hands: stable native contact point',
    loss='mean squared point distance / 0.05**2',
    sampling='latest qualified native substep per part; endpoint guidance if missing',
    target='population goal; hand native stable point; foot/knee fixed body-local material anchors')

def world_points(position, rotation, local):
    return position[..., None, :] + (rotation[..., None, :, :] @ local[..., None]).squeeze(-1)

def point_error_sq(actual, target):
    return (actual-target).square().sum(-1).mean(-1)

def mixed_error_sq(actual, target, native_point, point_target):
    """Use stable native contact points for hands and fixed material for feet/knees."""
    material=point_error_sq(actual,target)
    point=(native_point-point_target).square().sum(-1)
    hand=torch.tensor([False,False,True,True,False,False],device=material.device)
    return torch.where(hand,point,material)


def update_witness_error(previous, qualified, actual, target):
    return torch.where(qualified, point_error_sq(actual, target), previous)

def mesh_anchors(vertices):
    """Four fixed surface vertices spanning the part, anchored on its lower surface.

    Feet start at the lowest vertex in each XY quadrant. The actual mesh vertices
    are retained, not a synthetic plane or a geometry-based contact test.
    """
    v=torch.as_tensor(vertices,dtype=torch.float32)
    middle=(v.amin(0)+v.amax(0))/2
    chosen=[]
    for x,y in ((False,False),(False,True),(True,False),(True,True)):
        candidates=v[((v[:,0]>=middle[0])==x)&((v[:,1]>=middle[1])==y)]
        if not len(candidates):raise ValueError('Collision mesh has an empty anchor quadrant')
        chosen.append(candidates[candidates[:,2].argmin()])
    points=torch.stack(chosen)
    if torch.linalg.matrix_rank(points-points.mean(0))<2:
        raise ValueError('Material anchors must contain three non-collinear surface points')
    return points
