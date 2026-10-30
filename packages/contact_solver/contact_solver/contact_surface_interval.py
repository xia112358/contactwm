"""Loss geometry of the native-width interval above a finite contact face.

This distance never establishes contact. Activation and allocation are still
read from Newton; ``margin`` is provided by the actual solver configuration.
"""
from dataclasses import dataclass
import torch

CONTACT_SURFACE_INTERVAL_SCHEMA = 'native_margin_finite_upward_face_material_band_v1'


@dataclass(frozen=True)
class SurfaceIntervalQuery:
    """Selected FK material query, in the caller's link and pose chart.

    A unique, interior planar query may duplicate an affine solid witness.
    These identities never define solver activation or allocation.
    """
    link_names: tuple[str, ...]
    link: torch.Tensor
    point_local: torch.Tensor
    normal: torch.Tensor
    signed_gap: torch.Tensor
    unique: torch.Tensor
    interior: torch.Tensor


@dataclass(frozen=True)
class SurfaceIntervalBounds:
    """Squared lower/upper violations from the SAME selected geometry query.

    Anatomical regions and selected target surfaces retain their caller's
    identity. These are optimization residuals, never native contact labels.
    """
    lower: torch.Tensor
    upper: torch.Tensor
    # [solid_pair, side] target-component candidates, not activation or
    # sufficient proof of duplicate queries. Material identity is checked too.
    solid_target_mask: torch.Tensor | None = None
    query: SurfaceIntervalQuery | None = None
    # Preserve the provider's ordinary min scalar/derivative, including ties.
    # Decomposing it must not change floating-point reduction order.
    scalar: torch.Tensor | None = None

    @property
    def cost(self):
        return self.lower+self.upper if self.scalar is None else self.scalar


def same_interval_query(solid, query, link_names, part, region, side, *, bounds=None):
    """Identify duplicate affine material constraints, not nearby contacts.

    Entity/region equality alone is insufficient. Compare link, local point,
    signed distance and its normal. Tolerances cover numerical identity only.
    The caller also verifies static component identity and query selection.
    """
    if query is None:return torch.zeros_like(part,dtype=torch.bool)
    if tuple(query.link_names)!=tuple(link_names):
        raise ValueError('Interval query and solid link identities differ')
    batch=len(solid.q);shape=(batch,6,4)
    scalar_fields=(query.link,query.signed_gap,query.unique,query.interior)
    if any(v.shape!=shape for v in scalar_fields) or query.point_local.shape!=shape+(3,) or query.normal.shape!=shape+(3,):
        raise ValueError('Selected interval query must retain sample/part/region identity')
    if query.unique.dtype!=torch.bool or query.interior.dtype!=torch.bool or query.link.dtype!=torch.long:
        raise ValueError('Invalid selected interval query identity types')
    if not bool(torch.isfinite(query.signed_gap).all() & torch.isfinite(query.point_local).all() & torch.isfinite(query.normal).all()):
        raise ValueError('Undefined selected interval query geometry')
    selected=query.unique & query.interior
    if bool((selected & ((query.link<0)|(query.link>=len(link_names)))).any()):
        raise ValueError('Selected interval query link is outside the scene')
    if not hasattr(solid,'material_local') or not hasattr(solid,'normal_local'):
        raise ValueError('Same-query binding needs actual solid material coordinates and normals')
    sample=solid.sample;safe=part.clamp_min(0)
    gap=query.signed_gap[sample,safe,region].detach()
    point=query.point_local[sample,safe,region].detach()
    normal=query.normal[sample,safe,region].detach()
    physical_normal=solid.normal_local if side==1 else -solid.normal_local
    material=solid.material_local[:,side]
    epsilon=64*torch.finfo(solid.q.dtype).eps
    point_scale=torch.maximum(point.abs().amax(-1),material.abs().amax(-1)).clamp_min(1)
    gap_scale=torch.maximum(gap.abs(),solid.distances.detach().abs()).clamp_min(1)
    result=((part>=0) & selected[sample,safe,region]
        & (query.link[sample,safe,region]==solid.pair[f'body_link{side}'])
        & (gap<0) & (solid.distances.detach()<0)
        & ((point-material).abs().amax(-1)<=epsilon*point_scale)
        & ((normal-physical_normal).abs().amax(-1)<=epsilon)
        & ((gap-solid.distances.detach()).abs()<=epsilon*gap_scale))
    if bounds is not None:
        expected=(-gap).relu().square()
        supplied=bounds.lower[sample,safe,region].detach()
        result &= torch.isclose(supplied,expected,rtol=epsilon,atol=64*torch.finfo(solid.q.dtype).tiny)
        result &= bounds.upper[sample,safe,region].detach()==0
    return result


def convex_face_tangent_distance(points, normal, vertices, outward, offsets):
    """Exact distance of plane-projected points to a convex finite polygon.

    Inside points have zero tangent distance. Outside points use the nearest
    point on all polygon edges; summing half-space violations would overcount
    oblique corners. Geometry validation belongs to the actual face adapter.
    """
    gap = (points-vertices[0])@normal
    projected = points-gap[..., None]*normal
    edge = vertices.roll(-1, 0)-vertices
    delta = projected[..., None, :]-vertices
    parameter = (delta*edge).sum(-1)/edge.square().sum(-1)
    nearest = delta-parameter.clamp(0, 1)[..., None]*edge
    distance = torch.linalg.vector_norm(nearest, dim=-1).amin(-1)
    inside = (projected@outward.T <= offsets).all(-1)
    return torch.where(inside, 0, distance)[..., None]


def finite_face_band_cost(normal_gap, tangent_excess, margin):
    """Squared distance to the upward half of the face's margin neighborhood.

    A point may project outside a face and still lie within its native margin
    around an edge or corner. Penalizing tangent excess on its own would
    incorrectly require an exact footprint. Below the face, the normal
    residual supplies the intended upward recovery direction.
    """
    distance = torch.linalg.vector_norm(torch.cat((normal_gap.relu()[..., None], tangent_excess), -1), dim=-1)
    return (distance-margin).relu().square()+(-normal_gap).relu().square()
