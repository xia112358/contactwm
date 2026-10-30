from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F


HEIGHTMAP_FORWARD_MIN_M = -0.40


HEIGHTMAP_FORWARD_MAX_M = 1.00


HEIGHTMAP_LATERAL_MIN_M = -0.60


HEIGHTMAP_LATERAL_MAX_M = 0.60


HEIGHTMAP_RESOLUTION_M = 0.02


HEIGHTMAP_ROWS = 71


HEIGHTMAP_COLS = 61


HEIGHTMAP_CLIP_M = 1.50


def heightmap_grid() -> np.ndarray:
    """Return the fixed 2 cm root-yaw observation grid as [R,C,2]."""
    forward = np.linspace(
        HEIGHTMAP_FORWARD_MIN_M, HEIGHTMAP_FORWARD_MAX_M, HEIGHTMAP_ROWS, dtype=np.float32
    )
    lateral = np.linspace(
        HEIGHTMAP_LATERAL_MIN_M, HEIGHTMAP_LATERAL_MAX_M, HEIGHTMAP_COLS, dtype=np.float32
    )
    x, y = np.meshgrid(forward, lateral, indexing="ij")
    return np.stack((x, y), axis=-1)


def render_box_heightmaps(
    center: np.ndarray,
    rotation: np.ndarray,
    half_extents: np.ndarray,
    ground_height: np.ndarray,
) -> np.ndarray:
    """Render exact offline terrain geometry into the deployable observation.

    The returned tensor contains only sampled heights.  OBB parameters are not
    retained by the model and cannot be recovered as privileged inputs.
    """
    center = np.asarray(center, np.float32)
    rotation = np.asarray(rotation, np.float32)
    half_extents = np.asarray(half_extents, np.float32)
    ground_height = np.asarray(ground_height, np.float32).reshape(-1)
    if center.shape != (len(center), 3) or rotation.shape != (len(center), 3, 3):
        raise ValueError("invalid box pose for height-map rendering")
    if half_extents.shape != (len(center), 3) or ground_height.shape != (len(center),):
        raise ValueError("invalid box size/ground for height-map rendering")
    xy = heightmap_grid().reshape(-1, 2)
    points = np.zeros((len(center), len(xy), 3), np.float32)
    points[:, :, :2] = xy[None]
    local = np.einsum("npi,nij->npj", points - center[:, None], rotation)
    inside = (np.abs(local[..., :2]) <= half_extents[:, None, :2] + 1.0e-6).all(-1)
    top = center[:, 2] + rotation[:, 2, 2] * half_extents[:, 2]
    height = np.where(inside, top[:, None], ground_height[:, None])
    return np.clip(height, -HEIGHTMAP_CLIP_M, HEIGHTMAP_CLIP_M).reshape(
        -1, HEIGHTMAP_ROWS, HEIGHTMAP_COLS
    ).astype(np.float32)



def _numpy_root_yaw_basis(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, np.float32)
    w, x, y, z = np.moveaxis(q[:, 3:7], -1, 0)
    norm = np.sqrt(w*w+x*x+y*y+z*z)
    w, x, y, z = (value/norm for value in (w, x, y, z))
    cosine, sine = 1-2*(y*y+z*z), 2*(w*z+x*y)
    planar=np.sqrt(cosine*cosine+sine*sine);cosine,sine=cosine/planar,sine/planar
    basis = np.zeros((len(q), 3, 3), np.float32)
    basis[:,0,0],basis[:,1,0]=cosine,sine
    basis[:,0,1],basis[:,1,1]=-sine,cosine
    basis[:,2,2]=1
    return basis


def render_root_yaw_box_heightmaps(center, rotation, half_extents, ground_height, current_q,
                                   *, supersample: int = 1):
    """Render root-yaw height, optionally area-filtering each output cell."""
    center,rotation,half_extents,current_q=[np.asarray(x,np.float32) for x in
                                           (center,rotation,half_extents,current_q)]
    ground_height=np.asarray(ground_height,np.float32).reshape(-1);batch=len(center)
    if current_q.shape!=(batch,36):raise ValueError('current_q must be [B,36]')
    if center.shape!=(batch,3) or rotation.shape!=(batch,3,3):raise ValueError('invalid box pose')
    if supersample < 1:raise ValueError('supersample must be positive')
    grid=heightmap_grid().reshape(-1,2);basis=_numpy_root_yaw_basis(current_q)
    offsets=(np.arange(supersample,dtype=np.float32)+.5)/supersample-.5
    offsets=np.stack(np.meshgrid(offsets,offsets,indexing='ij'),-1).reshape(-1,2)*HEIGHTMAP_RESOLUTION_M
    top=center[:,2]+rotation[:,2,2]*half_extents[:,2];samples=[]
    for offset in offsets:
        local=np.zeros((batch,len(grid),3),np.float32);local[:,:,:2]=grid[None]+offset
        world=np.einsum('nij,npj->npi',basis,local)+current_q[:,None,:3]
        box_local=np.einsum('npi,nij->npj',world-center[:,None],rotation)
        inside=(np.abs(box_local[...,:2])<=half_extents[:,None,:2]+1e-6).all(-1)
        samples.append(np.where(inside,top[:,None],ground_height[:,None])-current_q[:,2:3])
    height=np.mean(samples,axis=0)
    return np.clip(height,-HEIGHTMAP_CLIP_M,HEIGHTMAP_CLIP_M).reshape(
        batch,HEIGHTMAP_ROWS,HEIGHTMAP_COLS).astype(np.float32)


def heightmap_supersample_for_architecture(architecture: str) -> int:
    """Return the recorded scan contract for each height-map architecture.

    Bound V4/V5 models bind contacts to actual grid samples, so averaging
    multiple terrain heights into one cell is forbidden for them.
    """
    return 2 if architecture == "heightmap_v3_robust" else 1


_DEVICE_HEIGHTMAP_GRIDS = {}


def render_root_yaw_box_heightmaps_device(center, rotation, half_extents, ground_height, current_q,
                                         *, supersample=1):
    """Same observation raster on device; this does not classify contact."""
    if current_q.ndim != 2 or current_q.shape[1] != 36 or supersample < 1:
        raise ValueError('Expected canonical qpos and positive supersampling')
    key = (current_q.device, current_q.dtype)
    if key not in _DEVICE_HEIGHTMAP_GRIDS:
        # Upload the identical static numpy linspace grid once, so replacing
        # observation arithmetic cannot change the raster cell definition.
        _DEVICE_HEIGHTMAP_GRIDS[key] = torch.as_tensor(heightmap_grid().reshape(-1, 2),
            device=current_q.device, dtype=current_q.dtype)
    grid = _DEVICE_HEIGHTMAP_GRIDS[key]
    basis, _ = _root_yaw_basis(current_q)
    offsets = (torch.arange(supersample, device=current_q.device, dtype=current_q.dtype)+.5)/supersample-.5
    offsets = torch.stack(torch.meshgrid(offsets, offsets, indexing='ij'), -1).reshape(-1, 2)*HEIGHTMAP_RESOLUTION_M
    top = center[:, 2]+rotation[:, 2, 2]*half_extents[:, 2]
    samples = []
    for offset in offsets.unbind():
        local = torch.cat((grid+offset, torch.zeros_like(grid[:, :1])), -1)
        world = torch.einsum('bij,pj->bpi', basis, local)+current_q[:, None, :3]
        box_local = torch.einsum('bpi,bij->bpj', world-center[:, None], rotation)
        inside = (box_local[..., :2].abs() <= half_extents[:, None, :2]+1e-6).all(-1)
        samples.append(torch.where(inside, top[:, None], ground_height[:, None])-current_q[:, 2:3])
    return torch.stack(samples).mean(0).clamp(-HEIGHTMAP_CLIP_M, HEIGHTMAP_CLIP_M).reshape(
        len(current_q), HEIGHTMAP_ROWS, HEIGHTMAP_COLS)


def _root_yaw_basis(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    qn=F.normalize(q[:,3:7],dim=-1);w,x,y,z=qn.unbind(-1)
    cosine,sine=1-2*(y*y+z*z),2*(w*z+x*y)
    planar=torch.sqrt(cosine.square()+sine.square()).clamp_min(1e-8);cosine,sine=cosine/planar,sine/planar
    zero=torch.zeros_like(cosine);one=torch.ones_like(cosine)
    basis=torch.stack((cosine,-sine,zero,sine,cosine,zero,zero,zero,one),-1).reshape(-1,3,3)
    yaw=torch.atan2(sine,cosine)
    yaw_quaternion=torch.stack((torch.cos(yaw/2),zero,zero,torch.sin(yaw/2)),-1)
    return basis,yaw_quaternion

