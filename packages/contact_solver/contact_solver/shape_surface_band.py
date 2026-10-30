"""Continuous collision-shape distance to a finite target contact band.

For a fixed orientation, translations that touch a convex obstacle face are
the upper cap of its Minkowski difference with the robot shape. A band swept
outward from that cap is collision free for that shape. Distance to this band
supplies approach and recovery in one scalar, without a pose solve, a teacher,
an EPA/task-depth gate, or redefining actual Newton contact.

Meshes use their initialized convex vertices. Spheres retain analytic rounded
edge/corner caps, including anatomical material cuts. Other bodies, obstacles
and self collisions still need complete-solid checks outside this adapter.
"""
import numpy as np
import torch
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from contact_solver.shape_face_interval import NativeShapeFaceIntervals, sphere_prism_support
from contact_solver.native_contact_position import region_planes, point_triangle_squared, clip_triangles
from contact_solver.contact_surface_interval import SurfaceIntervalBounds
from somaforge_core import CONTACT_BODY_NAMES_BY_PART
from somaforge_core.motion_contracts import BODY_NAMES
from somaforge_core.g1_kinematics import _matrix_from_rotation6d, _rotation6d


SHAPE_SURFACE_BAND_SCHEMA = 'realized_shape_finite_face_configuration_band_v1'


def triangle_band_bounds(triangles, outward, point, margin):
    """Native-width neighborhood on the outside of a finite cap triangle.

    Margin is Euclidean clearance, including edges/corners. Restricting it to
    a normal extrusion would wrongly penalize valid near-edge contacts.
    """
    gap = ((point-triangles[..., 0, :])*outward).sum(-1)
    projected = point-gap[..., None]*outward
    tangent2 = point_triangle_squared(projected, triangles)
    distance=(gap.relu().square()+tangent2).clamp_min(torch.finfo(gap.dtype).tiny).sqrt()
    return torch.stack(((-gap).relu().square(),(distance-margin).relu().square()),-1)


def triangle_band_cost(triangles, outward, point, margin):
    return triangle_band_bounds(triangles,outward,point,margin).sum(-1)


def minimum_bounds(values, groups, size):
    """Select one shared geometry alternative for both interval bounds.

    Independent minima of lower and upper would fabricate a feasible point.
    Ties share the derivative, matching torch.amin on the original scalar.
    """
    total=values.sum(-1)
    best=total.new_full((size,),torch.inf).scatter_reduce(0,groups,total,reduce='amin')
    selected=torch.isfinite(total)&(total==best[groups])
    count=total.new_zeros(size).scatter_add(0,groups,selected.to(total))
    result=values.new_zeros(size,2).scatter_add(0,groups[:,None].expand(-1,2),torch.where(selected[:,None],values,0))
    return (result/count.clamp_min(1)[:,None]).masked_fill(count[:,None]==0,torch.inf)


def enabled_face_shape_ids(geometry, polygons, faces):
    """Bind loss faces to actual static shapes and their enabled pair list.

    An initialized collision shape is not necessarily enabled against terrain.
    This mapping uses the same realized allowed pairs as complete-solid queries.
    Ambiguous coincident static faces need an explicit source mapping, not a
    guessed distance threshold or permission to use a disabled body shape.
    """
    import trimesh
    static=[s for s in geometry['shapes'] if s['body'] is None]
    # The export schema indexes its compact shape-record list. Native shape
    # IDs are deliberately sparse and are not these pair indices.
    records=geometry['shapes']
    allowed=set()
    for a,b in geometry['allowed_pairs']:
        if not (0<=a<len(records) and 0<=b<len(records)):
            raise ValueError('Enabled pair index is outside realized shape records')
        allowed.add(tuple(sorted((records[a]['shape'],records[b]['shape']))))
    robot={s['shape'] for s in geometry['shapes'] if s['body'] is not None}
    result={}
    for surface,polygon_tensor in polygons.items():
        polygon=polygon_tensor.detach().cpu().numpy();normal=faces[surface][2].numpy()
        plane=faces[surface][3];owners=[]
        for shape in static:
            if shape['kind'] in ('mesh','convex_mesh'):
                vertices=np.asarray(shape['vertices'])*np.asarray(shape['scale'])
                triangles=vertices[np.asarray(shape['faces'])]
            elif shape['kind']=='box':
                mesh=trimesh.creation.box(extents=2*np.asarray(shape['scale']))
                triangles=mesh.triangles
            else:
                continue
            r=Rotation.from_quat(shape['transform'][3:]).as_matrix()
            triangles=triangles@r.T+np.asarray(shape['transform'][:3])
            normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
            normals=normals/np.linalg.norm(normals,axis=-1,keepdims=True)
            selected=np.isclose(normals,normal,atol=1e-7,rtol=0).all(-1)&np.isclose(triangles[:,0]@normal,plane,atol=1e-7,rtol=0)
            if not selected.any():continue
            points=triangles[selected].reshape(-1,3)
            axis=np.eye(3)[np.argmin(np.abs(normal))];u=np.cross(normal,axis);u/=np.linalg.norm(u);v=np.cross(normal,u)
            hull=ConvexHull(np.stack((points@u,points@v),-1))
            xy=np.stack((polygon@u,polygon@v),-1)
            if (xy@hull.equations[:,:2].T+hull.equations[:,2] <= 1e-7).all():owners.append(shape['shape'])
        if len(owners)!=1:
            raise ValueError('Finite face needs one unambiguous realized static shape owner')
        result[surface]={sid for sid in robot if tuple(sorted((owners[0],sid))) in allowed}
    return result


def rounded_cap_bounds(point, base, radius, margin, normals, offsets):
    """Distance to an analytic sphere cap swept over radii [r,r+margin].

    The cap's unit normals satisfy immutable feature and moving anatomical
    half-spaces. Nearest-normal selection is an exact support problem; its
    real scalar derivative retains both the shape pose and clipping planes.
    """
    delta = point-base
    zero = torch.zeros_like(delta)
    score = -sphere_prism_support(zero, 1., normals, offsets, -delta)
    valid = torch.isfinite(score)
    safe_score = torch.where(valid, score, 0)
    angular=(delta.square().sum(-1)-safe_score.square()).clamp_min(0)
    lower=(radius-safe_score).relu().square()
    upper=(safe_score-radius-margin).relu().square()+angular
    return torch.stack((lower,upper),-1).masked_fill(~valid[...,None],torch.inf)


def rounded_cap_cost(point, base, radius, margin, normals, offsets):
    return rounded_cap_bounds(point,base,radius,margin,normals,offsets).sum(-1)


class NativeShapeSurfaceBand:
    def __init__(self, metadata, fk, low, high):
        self.reference = NativeShapeFaceIntervals(metadata, fk, low, high)
        self.margin = metadata['configured_margin']
        self.enabled=enabled_face_shape_ids(metadata['solid_geometry'],self.reference.polygons,self.reference.faces)
        self.low, self.high = low.detach().cpu().numpy(), high.detach().cpu().numpy()
        names = metadata['link_names']
        q = low.new_zeros(1, 36).double(); q[:, 3] = 1
        with torch.no_grad():
            p, r = fk.link_poses(q, names)
            rp, rr = fk.link_poses(q, BODY_NAMES[1:7])
        p, r, rp, rr = (x[0].numpy() for x in (p, r, rp, rr))
        self.shapes = {}
        by_link = {name:part for part,key in enumerate(('left_foot','right_foot','left_hand','right_hand','left_knee','right_knee'))
                   for name in CONTACT_BODY_NAMES_BY_PART[key]}
        for shape in metadata['solid_geometry']['shapes']:
            if shape['body'] not in by_link:
                continue
            part = by_link[shape['body']]; bi = names.index(shape['body'])
            sr = Rotation.from_quat(shape['transform'][3:]).as_matrix()
            local_r = rr[part].T@r[bi]@sr
            offset = rr[part].T@(p[bi]+r[bi]@np.asarray(shape['transform'][:3])-rp[part])
            record = dict(part=part, kind=shape['kind'], shape=shape['shape'])
            surfaces={sid for sid,shapes in self.enabled.items() if shape['shape'] in shapes}
            if not surfaces:
                continue
            record['surfaces']=surfaces
            if shape['kind'] == 'sphere':
                record.update(center=offset, radius=float(shape['scale'][0]))
            elif shape['kind'] in ('mesh','convex_mesh'):
                vertices = (np.asarray(shape['vertices'])*np.asarray(shape['scale']))@local_r.T+offset
                patches = []
                for region in range((4,4,2,2,3,3)[part]):
                    n,b = region_planes(part,region,self.low[part],self.high[part])
                    patch = clip_triangles(vertices[np.asarray(shape['faces'])],n,b)
                    patches.append(np.unique(patch.reshape(-1,3),axis=0))
                record.update(vertices=vertices, patches=patches)
            else:
                raise ValueError('No exact initialized shape/face band for '+shape['kind'])
            self.shapes[int(shape['shape'])] = record

    def mesh_costs(self, shape, position, rotation, surface, *, bounds=False):
        polygon = self.reference.polygons[surface].to(position)
        normal = self.reference.faces[surface][2].to(position)
        patches = shape.get('patches')
        if patches is None:
            # Small standalone geometric fixtures use closed convex vertices.
            # Production patches are always clipped from initialized skin.
            hull = ConvexHull(shape['vertices'])
            patches = []
            for region in range((4,4,2,2,3,3)[shape['part']]):
                n,b = region_planes(shape['part'],region,self.low[shape['part']],self.high[shape['part']])
                tri = clip_triangles(shape['vertices'][hull.simplices],n,b)
                patches.append(np.unique(tri.reshape(-1,3),axis=0))
        return torch.stack([self._mesh_patch_cost(patch,position,rotation,polygon,normal,bounds=bounds) for patch in patches])

    def _mesh_patch_cost(self, patch, position, rotation, polygon, normal, *, bounds=False):
        if not len(patch):
            return position.new_full((2,) if bounds else (),torch.inf)
        local = torch.as_tensor(patch, dtype=position.dtype, device=position.device)
        vertices = position+local@rotation.T
        cap = (polygon[:, None]-vertices[None]).reshape(-1, 3)
        # Hull chooses actual closest features. The graph uses moving vertices
        # and recalculated normals, never frozen distances or custom gradients.
        hull = ConvexHull(cap.detach().cpu().numpy())
        faces = hull.simplices[(hull.equations[:, :3]@normal.detach().cpu().numpy()) > 128*np.finfo(float).eps]
        if not len(faces):
            raise ValueError('Missing target-facing Minkowski cap')
        all_tri = cap[torch.as_tensor(faces, device=cap.device)]
        normals = torch.linalg.cross(all_tri[:, 1]-all_tri[:, 0], all_tri[:, 2]-all_tri[:, 0])
        sign = torch.where((normals*normal).sum(-1) >= 0, 1., -1.)
        normals = normals*sign[:, None]/normals.norm(dim=-1, keepdim=True)
        zero = position.new_zeros(3)
        values=triangle_band_bounds(all_tri,normals,zero,self.margin)
        if bounds:
            return minimum_bounds(values,torch.zeros(len(values),dtype=torch.long,device=values.device),1)[0]
        return values.sum(-1).amin()

    def _sphere_features(self, shape, position, rotation, surface, *, bounds=False):
        polygon = self.reference.polygons[surface].to(position)
        outward = self.reference.faces[surface][0].to(position)
        up = self.reference.faces[surface][2].to(position)
        radius = shape['radius']
        center_local = torch.as_tensor(shape['center'], device=position.device, dtype=position.dtype)
        point = position+rotation@center_local
        initial, bases, constraints, constraint_offsets, regions = [], [], [], [], []
        def add(base, normals, offsets, region):
            # A unit sphere already satisfies n_z<=1. Padding with this
            # redundant plane permits one batched support query across parts.
            padding = 6-len(normals)
            normals = torch.cat((normals, up[None].expand(padding,3)))
            offsets = torch.cat((offsets, point.new_ones(padding)))
            bases.append(base); constraints.append(normals); constraint_offsets.append(offsets); regions.append(region)
        for region in range((4,4,2,2,3,3)[shape['part']]):
            nl, bl = region_planes(shape['part'], region, self.low[shape['part']], self.high[shape['part']])
            nl = torch.as_tensor(nl, device=position.device, dtype=position.dtype)
            bl = torch.as_tensor(bl, device=position.device, dtype=position.dtype)
            closest = center_local.clone()
            for n, b in zip(nl, bl):
                closest = closest-(n@closest-b).relu()*n
            if bool((closest-center_local).norm() > radius):
                initial.append(point.new_full((2,) if bounds else (),torch.inf))
                continue
            # Source material is c-r*R.T*n, independent of shell expansion.
            region_n = -(nl@rotation.T)
            region_b = (bl-nl@center_local)/radius
            value = point.new_full((2,) if bounds else (),torch.inf)
            if bool(((region_n@up) <= region_b).all()):
                triangles = torch.stack([torch.stack((polygon[0], polygon[j], polygon[j+1]))
                                         for j in range(1, len(polygon)-1)])+radius*up
                values=triangle_band_bounds(triangles,up,point,self.margin)
                value=(minimum_bounds(values,torch.zeros(len(values),dtype=torch.long,device=values.device),1)[0]
                    if bounds else values.sum(-1).amin())
            initial.append(value)
            edges = polygon.roll(-1, 0)-polygon
            for j, edge in enumerate(edges):
                direction = edge/edge.norm()
                fraction = ((point-polygon[j])*edge).sum()/edge.square().sum()
                base = polygon[j]+fraction.clamp(0,1)*edge
                normals = torch.cat((region_n, -up[None], -outward[j,None], direction[None], -direction[None]))
                offsets = torch.cat((region_b, point.new_zeros(4)))
                add(base, normals, offsets, region)
            for j, base in enumerate(polygon):
                # The corner normal cone is the nonnegative span of adjacent
                # outward normals, intersected with the upward hemisphere.
                e_prev = edges[j-1]/edges[j-1].norm()
                e_next = edges[j]/edges[j].norm()
                normals = torch.cat((region_n, -up[None], -e_prev[None], e_next[None]))
                offsets = torch.cat((region_b, point.new_zeros(3)))
                add(base, normals, offsets, region)
        return torch.stack(initial), point, bases, constraints, constraint_offsets, regions

    def sphere_costs(self, shape, position, rotation, surface):
        initial, point, bases, normals, offsets, regions = self._sphere_features(shape, position, rotation, surface)
        if not bases:
            return initial
        values = rounded_cap_cost(point.expand(len(bases),3), torch.stack(bases), shape['radius'], self.margin,
            torch.stack(normals), torch.stack(offsets))
        ids = torch.tensor(regions, device=position.device)
        best = initial.new_full(initial.shape, torch.inf).scatter_reduce(0, ids, values, reduce='amin')
        return torch.minimum(initial, best)

    def costs(self, fk, q, surfaces, *, world_frame=None, _bounds=False):
        position, rotation = fk.link_poses(q, BODY_NAMES[1:7])
        rotation = _matrix_from_rotation6d(_rotation6d(rotation))
        if world_frame is not None:
            origin, basis = world_frame
            basis = _matrix_from_rotation6d(_rotation6d(basis.to(q)))
            position = origin[:, None].to(q)+torch.einsum('bij,bpj->bpi', basis, position)
            rotation = basis[:, None]@rotation
        region_costs = q.new_full((len(q),6,4,2) if _bounds else (len(q),6,4), torch.inf)
        shape_costs = {}
        def merge(a,b):
            if not _bounds:return torch.minimum(a,b)
            values=torch.stack((a,b),1).reshape(-1,2)
            ids=torch.arange(len(a),device=a.device).repeat_interleave(2)
            return minimum_bounds(values,ids,len(a))
        sphere_keys, sphere_initial, points, bases, normals, offsets, radii, feature_ids = [], [], [], [], [], [], [], []
        for sample in range(len(q)):
            for sid, shape in self.shapes.items():
                part = shape['part']; surface = int(surfaces[sample, part])
                if surface < 0:
                    continue
                if surface not in self.reference.faces:
                    raise ValueError('Unknown or nonprimary target surface')
                if 'surfaces' in shape and surface not in shape['surfaces']:
                    continue
                if shape['kind'] == 'sphere':
                    initial, point, sb, sn, so, sr = self._sphere_features(shape, position[sample,part],rotation[sample,part],surface,bounds=_bounds)
                    sphere_keys.append((sample,sid))
                    pad=(0,0,0,4-len(initial)) if _bounds else (0,4-len(initial))
                    sphere_initial.append(torch.nn.functional.pad(initial,pad,value=torch.inf))
                    points.extend([point]*len(sb)); bases.extend(sb); normals.extend(sn); offsets.extend(so)
                    radii.extend([shape['radius']]*len(sb))
                    feature_ids.extend([(len(sphere_keys)-1)*4+r for r in sr])
                    continue
                cost = self.mesh_costs(shape, position[sample, part], rotation[sample, part], surface,bounds=_bounds)
                count = len(cost)
                region_costs[sample,part,:count] = merge(region_costs[sample,part,:count].clone(), cost)
                shape_costs[sample,sid] = cost
        if sphere_keys:
            initial = torch.stack(sphere_initial)
            best = initial.new_full((len(sphere_keys)*4,2) if _bounds else (initial.numel(),),torch.inf)
            if bases:
                args=(torch.stack(points),torch.stack(bases),q.new_tensor(radii),self.margin,torch.stack(normals),torch.stack(offsets))
                values=rounded_cap_bounds(*args) if _bounds else rounded_cap_cost(*args)
                ids=torch.tensor(feature_ids,device=q.device)
                best=(minimum_bounds(values,ids,len(sphere_keys)*4) if _bounds else best.scatter_reduce(0,ids,values,reduce='amin'))
            costs=merge(initial.reshape(-1,2),best).reshape_as(initial) if _bounds else torch.minimum(initial,best.reshape_as(initial))
            for key,cost in zip(sphere_keys,costs):
                sample,sid = key; part = self.shapes[sid]['part']
                count = (4,4,2,2,3,3)[part]
                cost = cost[:count]
                region_costs[sample,part,:count] = merge(region_costs[sample,part,:count].clone(),cost)
                shape_costs[key] = cost
        return region_costs, shape_costs

    def intervals(self,fk,q,surfaces,*,world_frame=None):
        """Shared-query interval bounds; raw full-body safety stays separate."""
        values,shapes=self.costs(fk,q,surfaces,world_frame=world_frame,_bounds=True)
        return SurfaceIntervalBounds(values[...,0],values[...,1]),shapes
