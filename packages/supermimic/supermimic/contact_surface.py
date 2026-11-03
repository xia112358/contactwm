"""Contact objective on the actual convex collision surfaces, before/after contact."""
import numpy as np
import torch
from torch import nn
from .contact_control import LINKS
from .rotations import quat_to_mat


def nearest_triangle(points, triangles):
    """Closest point to every triangle, then select the nearest; piecewise differentiable."""
    a,b,c=triangles.unbind(-2);ab=b-a;ac=c-a;ap=points[:,None]-a
    aa=(ab*ab).sum(-1);bb=(ab*ac).sum(-1);cc=(ac*ac).sum(-1)
    u=(ap*ab).sum(-1);v=(ap*ac).sum(-1);den=(aa*cc-bb*bb).clamp_min(1e-20)
    s=(u*cc-v*bb)/den;t=(v*aa-u*bb)/den
    plane=a+s[...,None]*ab+t[...,None]*ac
    candidates=[plane];valid=[(s>=0)&(t>=0)&(s+t<=1)]
    for x,y in [(a,b),(b,c),(c,a)]:
        edge=y-x;alpha=((points[:,None]-x)*edge).sum(-1)/edge.square().sum(-1).clamp_min(1e-20)
        candidates.append(x+alpha.clamp(0,1)[...,None]*edge);valid.append(torch.ones_like(s,dtype=torch.bool))
    candidates=torch.stack(candidates,2);distance=(candidates-points[:,None,None]).square().sum(-1).masked_fill(~torch.stack(valid,2),float('inf'))
    flat=candidates.flatten(1,2);best=distance.flatten(1).argmin(-1)
    return flat[torch.arange(len(points),device=points.device),best]


class CollisionSurface(nn.Module):
    """Use MuJoCo mesh convex hulls and helper spheres, in canonical part-link coordinates."""
    def __init__(self, model_path, geometry, device='cpu', *, collision_profile='legacy_v1'):
        super().__init__()
        import mujoco
        from scipy.spatial import ConvexHull
        from somaforge_core.collision_profiles import contact_body_names, audit_surface_mujoco, require_profile
        from somaforge_core.robot_assets import canonical_g1_asset_metadata, validate_g1_asset_metadata
        from pathlib import Path
        import json, hashlib
        body_names=contact_body_names(collision_profile)
        self.collision_profile=collision_profile
        self.robot_asset=canonical_g1_asset_metadata(collision_profile=collision_profile)
        self.model_sha256=hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
        sidecar=Path(model_path).with_suffix(Path(model_path).suffix+'.json')
        if collision_profile=='surface_v2' and not sidecar.is_file():
            raise ValueError('surface_v2 needs a fingerprinted current-solver model; omit --collision_model to export it automatically')
        if collision_profile=='surface_v2' or sidecar.exists():
            meta=json.loads(sidecar.read_text())
            if meta.get('schema')!='somaforge_solver_collision_model_v1' or meta.get('model_sha256')!=self.model_sha256:
                raise ValueError('Collision model fingerprint/schema mismatch')
            validate_g1_asset_metadata(meta.get('robot_asset'),context=str(model_path))
            require_profile(meta['robot_asset'],collision_profile,context=str(model_path))
        from .contact_control import PARTS
        m=mujoco.MjModel.from_binary_path(str(model_path));names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,i) or '' for i in range(m.nbody)]
        if collision_profile=='surface_v2':audit_surface_mujoco(m)
        self.geometry=geometry;self.catalog=[]
        def rotation(q):return quat_to_mat(torch.tensor(q,dtype=torch.float64)).numpy()
        for part,main in enumerate(LINKS):
            triangles=[];spheres=[]
            for g in range(m.ngeom):
                body=int(m.geom_bodyid[g]);label=names[body]
                if not any(label.endswith('_'+n) or label==n for n in body_names[PARTS[part]]):continue
                if not (m.geom_contype[g] or m.geom_conaffinity[g]):continue
                p=m.geom_pos[g].copy();R=rotation(m.geom_quat[g]);ancestor=body
                while not (names[ancestor].endswith('_'+main) or names[ancestor]==main):
                    if ancestor==0 or m.body_jntnum[ancestor]:raise ValueError('Collider is not rigidly attached to part link')
                    br=rotation(m.body_quat[ancestor]);p=m.body_pos[ancestor]+br@p;R=br@R;ancestor=int(m.body_parentid[ancestor])
                kind=int(m.geom_type[g])
                if kind==int(mujoco.mjtGeom.mjGEOM_MESH):
                    mesh=int(m.geom_dataid[g]);start=int(m.mesh_vertadr[mesh]);v=m.mesh_vert[start:start+int(m.mesh_vertnum[mesh])];v=v@R.T+p
                    hull=ConvexHull(v);faces=v[hull.simplices].copy()
                    # Stable outward normals are needed by the surface atlas.
                    flip=(np.cross(faces[:,1]-faces[:,0],faces[:,2]-faces[:,0])*hull.equations[:,:3]).sum(-1)<0
                    faces[flip]=faces[flip][:,[0,2,1]]
                    triangles.append(faces)
                elif kind==int(mujoco.mjtGeom.mjGEOM_SPHERE):spheres.append([*p,float(m.geom_size[g,0])])
                else:raise ValueError(f'Unsupported collision geom {kind}')
            if not triangles and not spheres:raise ValueError(f'No actual colliders for {main}')
            self.register_buffer(f'triangles_{part}',torch.tensor(np.concatenate(triangles) if triangles else np.empty((0,3,3)),dtype=torch.float32,device=device))
            self.register_buffer(f'spheres_{part}',torch.tensor(spheres,dtype=torch.float32,device=device).reshape(-1,4))
            self.catalog.append(dict(part=PARTS[part],triangles=sum(len(x) for x in triangles),spheres=len(spheres)))

    def contact_points(self,S,target,parts):
        positions,rotations=self.geometry.poses(S);answer=positions.new_zeros(target.shape)
        for part in parts:
            p=positions[:,part];R=rotations[:,part];local=(R.transpose(-1,-2)@(target[:,part]-p).unsqueeze(-1)).squeeze(-1)
            options=[];triangles=getattr(self,f'triangles_{part}');spheres=getattr(self,f'spheres_{part}')
            if len(triangles):options.append(nearest_triangle(local,triangles))
            if len(spheres):
                delta=local[:,None]-spheres[None,:,:3];unit=delta/delta.norm(dim=-1,keepdim=True).clamp_min(1e-12)
                candidates=spheres[None,:,:3]+unit*spheres[None,:,3:4];best=(candidates-local[:,None]).square().sum(-1).argmin(-1)
                options.append(candidates[torch.arange(len(local),device=local.device),best])
            candidates=torch.stack(options,1);best=(candidates-local[:,None]).square().sum(-1).argmin(-1);chosen=candidates[torch.arange(len(local),device=local.device),best]
            answer[:,part]=p+(R@chosen.unsqueeze(-1)).squeeze(-1)
        return answer


class SurfaceRegionObjective(CollisionSurface):
    """Area-weighted scalar region cost on fixed actual collision-surface nodes."""

    def __init__(self, model_path, geometry, device='cpu', *, samples_per_part=128,
                 temperature=.02, radius=0., target_normal=None, collision_profile='legacy_v1'):
        super().__init__(model_path,geometry,device,collision_profile=collision_profile)
        from somaforge_core.contact_system.atlas import SurfaceAtlas
        from .contact_control import PARTS
        import math
        if not math.isfinite(radius) or not math.isfinite(temperature) or radius < 0 or temperature <= 0 or (radius > 0 and target_normal is None):
            raise ValueError('A disk target needs an explicit world normal; radius/temperature must be valid')
        if target_normal is not None:
            normal=torch.as_tensor(target_normal,dtype=torch.float64)
            if normal.shape!=(3,) or not torch.isfinite(normal).all() or abs(float(normal.norm())-1)>1e-5:
                raise ValueError('Target normal must be a finite unit world vector')
        self.temperature,self.radius=temperature,radius
        self.target_normal=target_normal
        self.atlas=SurfaceAtlas.from_surfaces(
            [getattr(self,f'triangles_{p}').cpu().numpy() for p in range(6)],
            [getattr(self,f'spheres_{p}').cpu().numpy() for p in range(6)],
            parts=PARTS,samples_per_part=samples_per_part,
            provenance=dict(robot_asset=self.robot_asset,model_sha256=self.model_sha256,
                            geometry='realized_mujoco_convex_hulls_and_spheres')).to(device)

    @property
    def contract(self):
        return dict(schema='somaforge_surface_region_objective_v1',atlas=self.atlas.fingerprint,model_sha256=self.model_sha256,
                    collision_profile=self.collision_profile,
                    samples_per_part=self.atlas.areas.shape[1],temperature=self.temperature,
                    radius_m=self.radius,target_normal=self.target_normal,
                    activation='native_constraint_evidence_only',
                    reduction='area_normalized_scalar_logsumexp',scoring='native_event_or_terminal')

    def region_loss(self,S,target,parts,*,normals=None,normal_weight=1.,tangent_weight=1.):
        from somaforge_core.contact_system.objectives import smooth_region_loss
        p,R=self.geometry.poses(S)
        candidates=self.atlas.world(p,R,parts=parts)
        if normals is None:
            # For a zero-radius isotropic point target the normal cancels.
            if self.target_normal is None and (normal_weight!=tangent_weight or self.radius>0):
                raise ValueError('Anisotropic/region losses require explicit target normals')
            normal=target.new_tensor([0.,0.,1.] if self.target_normal is None else self.target_normal)
            normals=normal.expand_as(target)
        result=smooth_region_loss(candidates,target[:,parts],normals[:,parts],
            self.atlas.areas[parts],temperature=self.temperature,halfwidth=self.radius,
            normal_weight=normal_weight,tangent_weight=tangent_weight)
        # Keep the existing six-part result layout for replay/task masking.
        return {key:target.new_zeros(target.shape[:2]).index_copy(1,
                    torch.as_tensor(parts,device=target.device,dtype=torch.long),result[key])
                for key in ('loss','normal','tangent','entropy_cost','hard_loss')}


class ContinuousArrivalSurface:
    """Fixed-node geometry for a soft landing process on a shared finite terrain.

    ``triangles`` must contain ALL task-eligible native terrain faces, in the
    same environment-local frame as geometry. Never pass just the goal face
    or the infinite plane through the target. Worlds must share this terrain.
    Native contacts/success metrics remain separate from this surrogate.
    """
    def __init__(self, surface, triangles, *, dt, **arrival_options):
        self.surface = surface
        self.triangles = torch.as_tensor(triangles,
            device=surface.atlas.areas.device, dtype=surface.atlas.areas.dtype)
        if self.triangles.ndim != 3 or self.triangles.shape[1:] != (3, 3) or not len(self.triangles):
            raise ValueError('Expected nonempty finite terrain triangles [face,3,3]')
        self.areas = surface.atlas.areas
        self.dt, self.arrival_options = dt, arrival_options

    def samples(self, state):
        points = self.material_points(state)
        return points,self.gaps(points)

    def material_points(self, state):
        p, R = self.surface.geometry.poses(state)
        return self.surface.atlas.world(p, R)

    def gaps(self, points):
        flat = points.reshape(-1, 3)
        closest = nearest_triangle(flat, self.triangles)
        return (flat-closest).norm(dim=-1).reshape(points.shape[:-1])

    def begin(self, state, initial_supported):
        from .contact_soft_arrival import SoftLanding
        _, gap = self.samples(state)
        return SoftLanding(gap, self.areas, initial_supported=initial_supported, **self.arrival_options)
