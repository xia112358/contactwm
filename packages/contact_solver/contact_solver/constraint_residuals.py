"""Geometry-to-residual adapters, with no robot, task, or contact-truth defaults.

Callers supply differentiable observations from their FK/collision backend and
stable identities. SDFs and distances guide optimization; they do not establish
Newton/MJWarp contact. Missing collision evidence must be raised by the adapter.
"""
from dataclasses import dataclass
from typing import Literal, Protocol
import torch
from .constraint_learning import Residual


class SignedDistanceField(Protocol):
    def distance(self,world_points: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class OrientedBoxField:
    center: torch.Tensor
    basis: torch.Tensor
    half_extents: torch.Tensor

    def __post_init__(self):
        if self.center.shape!=(3,) or self.basis.shape!=(3,3) or self.half_extents.shape!=(3,):raise ValueError('Invalid box field shapes')
        if not bool(torch.isfinite(self.center).all() and torch.isfinite(self.half_extents).all() and (self.half_extents>0).all()):raise ValueError('Invalid box field dimensions')
        if not torch.allclose(self.basis.T@self.basis,torch.eye(3,device=self.basis.device,dtype=self.basis.dtype),atol=1e-6,rtol=0):raise ValueError('Box basis must be orthonormal')

    def distance(self,points):
        d=((points-self.center)@self.basis).abs()-self.half_extents
        return d.clamp_min(0).norm(dim=-1)+d.amax(-1).clamp_max(0)


@dataclass(frozen=True)
class SphereField:
    center: torch.Tensor
    radius: float

    def __post_init__(self):
        import math
        if self.center.shape!=(3,) or not bool(torch.isfinite(self.center).all()) or not math.isfinite(self.radius) or self.radius<=0:raise ValueError('Invalid sphere field')

    def distance(self,points):return (points-self.center).norm(dim=-1)-self.radius


@dataclass(frozen=True)
class PlaneSurface:
    origin: torch.Tensor
    normal: torch.Tensor
    tangents: torch.Tensor  # [2,3], orthonormal world directions
    boundary_normals: torch.Tensor | None = None  # optional inward-region: n*x <= offset
    boundary_offsets: torch.Tensor | None = None

    def __post_init__(self):
        if self.origin.shape!=(3,) or self.normal.shape!=(3,) or self.tangents.shape!=(2,3):raise ValueError('Invalid plane frame shapes')
        frame=torch.cat((self.tangents,self.normal[None]))
        if not torch.allclose(frame@frame.T,torch.eye(3,device=frame.device,dtype=frame.dtype),atol=1e-6,rtol=0):raise ValueError('Plane frame must be orthonormal')
        if not bool(torch.isfinite(self.origin).all()):raise ValueError('Nonfinite plane origin')
        if (self.boundary_normals is None)!=(self.boundary_offsets is None):raise ValueError('Incomplete surface boundaries')
        if self.boundary_normals is not None:
            n=self.boundary_normals;b=self.boundary_offsets
            if n.ndim!=2 or n.shape[1]!=3 or b.shape!=(len(n),) or not len(n):raise ValueError('Invalid boundary shapes')
            if not bool(torch.isfinite(n).all() and torch.isfinite(b).all()) or not torch.allclose(n.norm(dim=1),torch.ones_like(b),atol=1e-6):raise ValueError('Boundaries require finite unit normals')
            if not torch.allclose(n@self.normal,torch.zeros_like(b),atol=1e-6):raise ValueError('Boundary normals must lie in the surface plane')

    def distance(self,points):return (points-self.origin)@self.normal
    def tangent(self,points):return points@self.tangents.T
    def coverage(self,points):
        if self.boundary_normals is None:raise ValueError('Coverage requested without finite surface boundaries')
        return points@self.boundary_normals.T-self.boundary_offsets


@dataclass(frozen=True)
class ContactTask:
    task_id: str
    mode: Literal['establish','keep','release']
    distance_scale: float
    min_gap: float
    max_gap: float
    normal_scale: float = 1.
    coverage: bool = False

    def __post_init__(self):
        import math
        if not self.task_id or self.mode not in ('establish','keep','release'):raise ValueError('Invalid contact task')
        if not all(math.isfinite(x) for x in (self.distance_scale,self.min_gap,self.max_gap,self.normal_scale)) or self.distance_scale<=0 or self.normal_scale<=0 or self.max_gap<self.min_gap:raise ValueError('Invalid contact scales or gap interval')


def contact_residuals(sample_id,context_id,task,surface,*,support_gap,
                      anchor_world=None,anchor_target=None,normal_world=None,
                      normal_target=None,coverage_points=None,realized=None,
                      approach_gap=None):
    """Exact/approximate support supplied by robot geometry, not a link-name rule.

    Scalar support_gap is the closest shape/patch separation to this surface.
    Alignment and tangential locking are independently optional. Releases only
    request a minimum separation, never a new contact or an anchor lock.
    """
    def row(suffix,kind,value,scale):return Residual.scaled(sample_id,context_id,f'contact/{task.task_id}/{suffix}',kind,value,scale)
    if support_gap.numel()!=1:raise ValueError('One support gap per contact task required')
    if (anchor_world is None)!=(anchor_target is None) or (normal_world is None)!=(normal_target is None):raise ValueError('Incomplete contact evidence')
    if task.mode=='release':
        if any(x is not None for x in (anchor_world,normal_world,coverage_points)) or task.coverage:raise ValueError('Release cannot carry alignment, coverage or locking')
        return [row('release','le',task.min_gap-support_gap,task.distance_scale)]
    # Solver evidence is supplied by the adapter, never inferred from geometry.
    # Keep the stable row identity when activation switches off attraction.
    upper=(support_gap if approach_gap is None else approach_gap)-task.max_gap
    if realized is not None:
        if not isinstance(realized,bool):raise ValueError('realized must be audited solver evidence')
        if realized:upper=support_gap*0
    rows=[row('lower_gap','le',task.min_gap-support_gap,task.distance_scale),row('upper_gap','le',upper,task.distance_scale)]
    if anchor_world is not None:
        if anchor_world.shape!=anchor_target.shape or anchor_world.shape[-1]!=3:raise ValueError('Invalid material anchors')
        rows.append(row('tangent','eq',surface.tangent(anchor_world-anchor_target),task.distance_scale))
    if normal_world is not None:
        if normal_world.shape!=(3,) or normal_target.shape!=(3,):raise ValueError('One normal vector required')
        if not bool(torch.isfinite(normal_world).all() and torch.isfinite(normal_target).all()) or min(float(normal_world.detach().norm()),float(normal_target.detach().norm()))<1e-8:raise ValueError('Invalid contact normal')
        rows.append(row('alignment','eq',normal_world/normal_world.norm()-normal_target/normal_target.norm(),task.normal_scale))
    if task.coverage:
        if coverage_points is None:raise ValueError('Missing contact patch for coverage')
        rows.append(row('coverage','le',surface.coverage(coverage_points),task.distance_scale))
    elif coverage_points is not None:raise ValueError('Unexpected coverage points')
    return rows


def collision_residual(sample_id,context_id,constraint_id,distances,*,distance_scale,component_ids,clearance=0.):
    """Signed distances, including self-collision rows, with stable identities."""
    return Residual.scaled(sample_id,context_id,f'collision/{constraint_id}','le',clearance-distances,distance_scale,component_ids=component_ids)


def joint_limit_residuals(sample_id,context_id,joints,lower,upper,*,angle_scale,joint_names):
    if joints.shape!=lower.shape or joints.shape!=upper.shape or not bool((lower<=upper).all()):raise ValueError('Invalid joint limits')
    return [Residual.scaled(sample_id,context_id,'joint_limits/lower','le',lower-joints,angle_scale,component_ids=joint_names),
            Residual.scaled(sample_id,context_id,'joint_limits/upper','le',joints-upper,angle_scale,component_ids=joint_names)]


def quadratic_prior(displacement,weights):
    """Caller-defined normalized pose difference; no teacher is constructed here."""
    weights=torch.as_tensor(weights,device=displacement.device,dtype=displacement.dtype)
    if weights.shape!=displacement.shape or not bool(torch.isfinite(weights).all()) or not bool((weights>=0).all()):raise ValueError('Finite nonnegative per-coordinate prior weights required')
    return .5*(weights*displacement.square()).sum()
