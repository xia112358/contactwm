"""Native contact-part surface distance to an issued position's normal band.

Uses initialized Newton meshes/spheres, with canonical anatomical regions.
The target is a vertical segment spanning the actual normal gap interval.
Unlike XY witness error, this metric includes normal excess when the surface
is outside that band. Contact activation/coverage and complete-solid clearance
remain separate requirements. No pose solve, teacher, or custom backward.
"""
import itertools
import json
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from somaforge_core.contact_schema import CONTACT_BODY_NAMES_BY_PART
from somaforge_core.motion_contracts import BODY_NAMES
from somaforge_core.g1_kinematics import _matrix_from_rotation6d, _rotation6d
from somaforge_core.robot_assets import validate_g1_asset_metadata
from somaforge_core.solid_distance import SOLID_GEOMETRY_SCHEMA
from somaforge_core.contact_source_geometry import SOURCE_NORMAL_FAN_SCHEMA

CONTACT_POSITION_SCHEMA = 'native_part_surface_to_normal_interval_rms_v1'

PARTS = ('left_foot','right_foot','left_hand','right_hand','left_knee','right_knee')


def point_segment_squared(p, a, b):
    e=b-a
    t=((p-a)*e).sum(-1)/e.square().sum(-1).clamp_min(torch.finfo(p.dtype).tiny)
    return (p-a-t.clamp(0,1)[...,None]*e).square().sum(-1)


def point_triangle_squared(p, triangle):
    a,b,c=triangle.unbind(-2); e=b-a; f=c-a; v=p-a
    n=torch.linalg.cross(e,f); n2=n.square().sum(-1)
    projected=p-((v*n).sum(-1)/n2.clamp_min(torch.finfo(p.dtype).tiny))[...,None]*n
    pe=projected-a; ee=e.square().sum(-1); ff=f.square().sum(-1); ef=(e*f).sum(-1)
    det=(ee*ff-ef.square()).clamp_min(torch.finfo(p.dtype).tiny)
    u=((pe*e).sum(-1)*ff-(pe*f).sum(-1)*ef)/det
    w=((pe*f).sum(-1)*ee-(pe*e).sum(-1)*ef)/det
    inside=(u>=0)&(w>=0)&(u+w<=1)&(n2>0)
    edges=torch.stack([point_segment_squared(p,a,b),point_segment_squared(p,b,c),point_segment_squared(p,c,a)],-1).amin(-1)
    return torch.where(inside,(p-projected).square().sum(-1),edges)


def segment_segment_squared(a,b,c,d):
    e=b-a; f=d-c; v=a-c
    ee=e.square().sum(-1); ff=f.square().sum(-1); ef=(e*f).sum(-1)
    ev=(e*v).sum(-1); fv=(f*v).sum(-1); det=ee*ff-ef.square()
    tol=64*torch.finfo(a.dtype).eps
    nonparallel=det>tol*ee*ff
    denominator=torch.where(nonparallel,det,torch.ones_like(det))
    s=(ef*fv-ff*ev)/denominator; t=(ee*fv-ef*ev)/denominator
    interior=nonparallel&(s>=0)&(s<=1)&(t>=0)&(t<=1)
    cost=(v+s[...,None]*e-t[...,None]*f).square().sum(-1)
    boundaries=torch.stack([point_segment_squared(a,c,d),point_segment_squared(b,c,d),
        point_segment_squared(c,a,b),point_segment_squared(d,a,b)],-1).amin(-1)
    return torch.where(interior,torch.minimum(cost,boundaries),boundaries)


def _segment_triangle_costs(a,b,tri):
    a,b=a[:,None],b[:,None]
    v0,v1,v2=tri.unbind(-2);e=v1-v0;f=v2-v0
    n=torch.linalg.cross(e,f);n2=n.square().sum(-1)
    tiny=torch.finfo(tri.dtype).tiny
    ee=e.square().sum(-1);ff=f.square().sum(-1);ef=(e*f).sum(-1)
    det=(ee*ff-ef.square()).clamp_min(tiny)
    def barycentric(delta):
        de=(delta*e).sum(-1);df=(delta*f).sum(-1)
        u=(de*ff-df*ef)/det;w=(df*ee-de*ef)/det
        return (u>=0)&(w>=0)&(u+w<=1)
    # Edge/segment distances already contain every endpoint/edge distance;
    # endpoint face candidates need not evaluate those edges a second time.
    cost=segment_segment_squared(a[:,:,None],b[:,:,None],tri,tri.roll(-1,dims=-2)).amin(-1)
    for point in (a,b):
        delta=point-v0;dn=(delta*n).sum(-1)
        projection=point-(dn/n2.clamp_min(tiny))[...,None]*n
        inside=barycentric(projection-v0)&(n2>0)
        face=(point-projection).square().sum(-1).masked_fill(~inside,torch.inf)
        cost=torch.minimum(cost,face)
    denom=((b-a)*n).sum(-1);eps=64*torch.finfo(tri.dtype).eps
    valid=denom.abs()>eps*torch.linalg.vector_norm(b-a,dim=-1)*torch.linalg.vector_norm(n,dim=-1)
    t=((v0-a)*n).sum(-1)/torch.where(valid,denom,torch.ones_like(denom))
    intersect=valid&(t>=0)&(t<=1)&barycentric(a+t[...,None]*(b-a)-v0)
    return torch.where(intersect,0.,cost)


def segment_triangles_squared(a,b,tri):
    """Exact distance; build the backward graph only for all minimizing faces."""
    with torch.no_grad():
        costs=_segment_triangle_costs(a,b,tri)
        minimum=costs.amin(-1)
        selected=costs==minimum[:,None]
        sample,face=selected.nonzero(as_tuple=True)
    if not torch.is_grad_enabled() or not (a.requires_grad or b.requires_grad or tri.requires_grad):
        return minimum
    triangles=tri.expand(len(a),-1,-1,-1)[sample,face][:,None]
    values=_segment_triangle_costs(a[sample],b[sample],triangles)[:,0]
    # amin distributes its gradient equally across all tied minima.
    return values.new_zeros(len(a)).scatter_add(0,sample,values)/selected.sum(-1)


def region_planes(part,region,low,high):
    mid=(low+high)/2; n=[]; b=[]
    def bound(axis,value,positive):
        normal=np.zeros(3);normal[axis]=1 if positive else -1
        n.append(normal);b.append(value if positive else -value)
    if part<2:
        bound(0,mid[0],region//2==0);bound(1,mid[1],region%2==0)
    elif part<4: bound(0,mid[0],region==0)
    else:
        width=(high[2]-low[2])/3
        if region>0:bound(2,low[2]+region*width,False)
        if region<2:bound(2,low[2]+(region+1)*width,True)
    return np.asarray(n),np.asarray(b)


def clip_triangles(triangles,normals,offsets):
    out=[]
    for triangle in triangles:
        polygon=list(triangle)
        for n,b in zip(normals,offsets):
            clipped=[]
            for p,q in zip(polygon,polygon[1:]+polygon[:1]):
                dp,dq=n@p-b,n@q-b
                if dp<=0:clipped.append(p)
                if (dp<0 and dq>0) or (dp>0 and dq<0):clipped.append(p+(q-p)*dp/(dp-dq))
            polygon=clipped
            if not polygon:break
        for i in range(1,len(polygon)-1):
            t=np.array([polygon[0],polygon[i],polygon[i+1]])
            if np.linalg.norm(np.cross(t[1]-t[0],t[2]-t[0]))>0:out.append(t)
    return np.asarray(out).reshape(-1,3,3)


def sphere_patch_squared(center,radius,normals,offsets,a,b,position,rotation):
    """Exact sphere skin cut by regional planes to a vertical segment.

    Enumerate unconstrained sphere, clipping circles, and their intersections.
    Circle XY extrema use all real stationary roots. Roots only choose the
    closest feature: scalar differentiation uses the envelope theorem, with
    all moving geometry retained. This is not a frozen solver material point.
    """
    c=position+rotation@center; candidates=[]; eps=128*torch.finfo(a.dtype).eps
    slack=eps*(center.abs().sum()+radius+offsets.abs().sum()+1)
    def add(p):
        local=rotation.T@(p-position)
        if bool(torch.isfinite(p).all() & ((normals@local-offsets)<=slack).all()):
            candidates.append(point_segment_squared(p,a,b))
    for p in (a,b):
        delta=p-c
        if bool(delta.norm()>eps):add(c+radius*delta/delta.norm())
        else:
            for axis in torch.cat((torch.eye(3,dtype=a.dtype),-torch.eye(3,dtype=a.dtype))):add(c+radius*axis)
    xy=a[:2]-c[:2]
    if bool(xy.norm()>eps):add(torch.cat((c[:2]+radius*xy/xy.norm(),c[2:3])))
    horizontal=xy.square().sum()
    if bool(horizontal<=radius*radius):
        h=(radius*radius-horizontal).clamp_min(0).sqrt()
        for z in (c[2]-h,c[2]+h):
            if bool((z>=a[2])&(z<=b[2])):add(torch.cat((a[:2],z[None])))
    for nl,bl in zip(normals,offsets):
        separation=bl-nl@center; r2=radius*radius-separation.square()
        if bool(r2<0):continue
        cc=position+rotation@(center+separation*nl); normal=rotation@nl
        if bool(r2==0):add(cc);continue
        extent=r2.sqrt(); horizontal_length=normal[:2].norm()
        if bool(horizontal_length>eps):
            minor=normal[:2]/horizontal_length;major=torch.stack((-minor[1],minor[0]))
        else:major,minor=a.new_tensor([1.,0.]),a.new_tensor([0.,1.])
        u=torch.cat((major,a.new_zeros(1)));v=torch.cat((normal[2]*minor,(-horizontal_length)[None]))
        for p in (a,b):
            delta=p-cc; du,dv=delta@u,delta@v
            norm=(du.square()+dv.square()).sqrt()
            if bool(norm>eps):add(cc+extent*(du*u+dv*v)/norm)
            else:
                for direction in (u,-u,v,-v):add(cc+extent*direction)
        rhs1=(a[:2]-cc[:2])@major;rhs2=normal[2]*((a[:2]-cc[:2])@minor)
        k=extent*(normal[2].square()-1)
        polynomial=torch.stack((rhs2,2*rhs1-2*k,k*0,2*rhs1+2*k,-rhs2)).detach().numpy()
        angles=[0.,np.pi/2,np.pi,3*np.pi/2]
        for t in np.roots(np.trim_zeros(polynomial,'f')):
            if abs(t.imag)<=128*np.finfo(float).eps*(1+abs(t.real)):angles.append(2*np.arctan(t.real))
        for angle in angles:
            p=cc+extent*(np.cos(angle)*u+np.sin(angle)*v)
            if bool((p[2]>=a[2])&(p[2]<=b[2])):add(p)
    for i,j in itertools.combinations(range(len(normals)),2):
        n=normals[[i,j]];line=torch.linalg.cross(n[0],n[1])
        if bool(line.norm()<=eps):continue
        line=line/line.norm();shift=n.T@torch.linalg.solve(n@n.T,offsets[[i,j]]-n@center)
        r2=radius*radius-shift.square().sum()
        if bool(r2>=0):
            for sign in (-1,1):add(position+rotation@(center+shift+sign*r2.clamp_min(0).sqrt()*line))
    if not candidates:raise ValueError('Empty sphere surface region; geometry must be excluded at construction')
    return torch.stack(candidates).amin()


def sphere_patch_batch_squared(center,radius,normals,offsets,a,b,position,rotation):
    """Same sphere-skin candidates as the scalar oracle, batched over poses."""
    c=position+torch.einsum('bij,j->bi',rotation,center)
    eps=128*torch.finfo(a.dtype).eps
    # A whole sphere has a closed-form distance to a segment, including the
    # case where the segment lies entirely inside the sphere's skin.
    if bool((normals@center+radius<=offsets).all()):
        edge=b-a
        t=((c-a)*edge).sum(-1)/edge.square().sum(-1).clamp_min(torch.finfo(a.dtype).tiny)
        nearest=(c-a-t.clamp(0,1)[:,None]*edge).norm(dim=-1)
        farthest=torch.maximum((a-c).norm(dim=-1),(b-c).norm(dim=-1))
        return torch.maximum((nearest-radius).relu(),(radius-farthest).relu()).square()
    slack=eps*(center.abs().sum()+radius+offsets.abs().sum()+1)
    candidates=[]
    def add(p,valid=None):
        local=torch.einsum('bji,bj->bi',rotation,p-position)
        ok=torch.isfinite(p).all(-1)&((local@normals.T-offsets)<=slack).all(-1)
        if valid is not None:ok=ok&valid
        candidates.append(point_segment_squared(p,a,b).masked_fill(~ok,torch.inf))
    for point in (a,b):
        delta=point-c;norm=delta.norm(dim=-1)
        add(c+radius*delta/norm.clamp_min(eps)[:,None],norm>eps)
        for axis in torch.cat((torch.eye(3,dtype=a.dtype),-torch.eye(3,dtype=a.dtype))):
            add(c+radius*axis,norm<=eps)
    xy=a[:,:2]-c[:,:2];norm=xy.norm(dim=-1)
    add(torch.cat((c[:,:2]+radius*xy/norm.clamp_min(eps)[:,None],c[:,2:3]),-1),norm>eps)
    horizontal=xy.square().sum(-1)
    # Zero-radius intersections have zero distance; avoid sqrt'(0) NaNs.
    h=(radius*radius-horizontal).clamp_min(torch.finfo(a.dtype).tiny).sqrt()
    for z in (c[:,2]-h,c[:,2]+h):
        add(torch.cat((a[:,:2],z[:,None]),-1),(horizontal<=radius*radius)&(z>=a[:,2])&(z<=b[:,2]))
    for nl,bl in zip(normals,offsets):
        separation=bl-nl@center;r2=radius*radius-separation.square()
        if bool(r2<0):continue
        cc=position+torch.einsum('bij,j->bi',rotation,center+separation*nl)
        normal=torch.einsum('bij,j->bi',rotation,nl)
        if bool(r2==0):add(cc);continue
        extent=r2.sqrt();length=normal[:,:2].norm(dim=-1)
        minor=normal[:,:2]/length.clamp_min(eps)[:,None]
        minor=torch.where((length>eps)[:,None],minor,a.new_tensor([0.,1.]))
        major=torch.stack((-minor[:,1],minor[:,0]),-1)
        major=torch.where((length>eps)[:,None],major,a.new_tensor([1.,0.]))
        u=torch.cat((major,torch.zeros_like(length[:,None])),-1)
        v=torch.cat((normal[:,2,None]*minor,-length[:,None]),-1)
        for point in (a,b):
            delta=point-cc;du=(delta*u).sum(-1);dv=(delta*v).sum(-1)
            norm=torch.stack((du,dv),-1).norm(dim=-1)
            add(cc+extent*(du[:,None]*u+dv[:,None]*v)/norm.clamp_min(eps)[:,None],norm>eps)
            for direction in (u,-u,v,-v):add(cc+extent*direction,norm<=eps)
        rhs1=((a[:,:2]-cc[:,:2])*major).sum(-1)
        rhs2=normal[:,2]*((a[:,:2]-cc[:,:2])*minor).sum(-1)
        k=extent*(normal[:,2].square()-1)
        polynomial=torch.stack((rhs2,2*rhs1-2*k,k*0,2*rhs1+2*k,-rhs2),-1).detach().numpy()
        angles=np.zeros((len(a),8));valid=np.zeros((len(a),8),dtype=bool)
        angles[:,:4]=[0.,np.pi/2,np.pi,3*np.pi/2];valid[:,:4]=True
        for row,poly in enumerate(polynomial):
            roots=np.roots(np.trim_zeros(poly,'f'))
            real=[2*np.arctan(t.real) for t in roots if abs(t.imag)<=128*np.finfo(float).eps*(1+abs(t.real))]
            angles[row,4:4+len(real)]=real;valid[row,4:4+len(real)]=True
        for col in range(8):
            angle=a.new_tensor(angles[:,col]);ok=torch.as_tensor(valid[:,col])
            point=cc+extent*(angle.cos()[:,None]*u+angle.sin()[:,None]*v)
            add(point,ok&(point[:,2]>=a[:,2])&(point[:,2]<=b[:,2]))
    for i,j in itertools.combinations(range(len(normals)),2):
        n=normals[[i,j]];line=torch.linalg.cross(n[0],n[1])
        if bool(line.norm()<=eps):continue
        line=line/line.norm();shift=n.T@torch.linalg.solve(n@n.T,offsets[[i,j]]-n@center)
        r2=radius*radius-shift.square().sum()
        if bool(r2>=0):
            for sign in (-1,1):add(position+torch.einsum('bij,j->bi',rotation,center+shift+sign*r2.clamp_min(0).sqrt()*line))
    result=torch.stack(candidates).amin(0)
    if not bool(torch.isfinite(result).all()):raise ValueError('Empty sphere surface region')
    return result


class NativeContactRegionGeometry:
    def __init__(self,metadata,fk,low,high):
        native=metadata.get('solid_geometry',{})
        if native.get('schema')!=SOLID_GEOMETRY_SCHEMA or not native.get('model_fingerprint'):
            raise ValueError('Contact position requires initialized native geometry')
        validate_g1_asset_metadata(native.get('robot_asset'),context='native contact position')
        if metadata.get('surface_attribution_schema')!=SOURCE_NORMAL_FAN_SCHEMA:
            raise ValueError('Contact position requires current native face attribution')
        margin=metadata.get('configured_margin')
        if not isinstance(margin,(int,float)) or not np.isfinite(margin) or margin<=0:
            raise ValueError('Contact position requires the actual positive native margin')
        if low.shape!=(6,3) or high.shape!=(6,3) or not bool(torch.isfinite(low).all()&torch.isfinite(high).all()&(high>low).all()):
            raise ValueError('Invalid canonical contact-region boundaries')
        low,high=low.detach().cpu(),high.detach().cpu()
        self.metadata=metadata;self.fk=fk;self.patch={}
        parents={child:(parent,kind) for parent,child,kind,_ in fk.joint_records}
        names=tuple(metadata['link_names']);q=low.new_zeros(1,36).double();q[:,3]=1
        with torch.no_grad():
            p,r=fk.link_poses(q,names);rp,rr=fk.link_poses(q,BODY_NAMES[1:7])
            r=_matrix_from_rotation6d(_rotation6d(r));rr=_matrix_from_rotation6d(_rotation6d(rr))
        p,r,rp,rr=(x.numpy() for x in (p[0],r[0],rp[0],rr[0]))
        for part,part_name in enumerate(PARTS):
            local_tri=[];local_spheres=[]
            for s in metadata['solid_geometry']['shapes']:
                if s['body'] not in CONTACT_BODY_NAMES_BY_PART[part_name]:continue
                link=s['body'];reference=BODY_NAMES[part+1]
                while link!=reference:
                    if link not in parents or parents[link][1]!='fixed':
                        raise ValueError(f'Native contact body {s["body"]} is not rigid relative to {reference}')
                    link=parents[link][0]
                bi=names.index(s['body']);tr=rr[part].T@r[bi]@Rotation.from_quat(s['transform'][3:]).as_matrix()
                offset=rr[part].T@(p[bi]+r[bi]@np.array(s['transform'][:3])-rp[part])
                if s['kind']=='sphere':local_spheres.append((offset,s['scale'][0]))
                elif s['kind'] in ('mesh','convex_mesh'):
                    v=(np.array(s['vertices'])*np.array(s['scale']))@tr.T+offset
                    local_tri.extend(v[np.array(s['faces'])])
                else:raise ValueError('Unsupported actual contact geometry '+s['kind'])
            for region in range((4,4,2,2,3,3)[part]):
                n,b=region_planes(part,region,low[part].numpy(),high[part].numpy())
                tri=clip_triangles(local_tri,n,b);spheres=[]
                for c,rad in local_spheres:
                    if not np.isfinite(rad) or rad<=0:raise ValueError('Invalid native sphere radius')
                    # Region cuts are axis aligned in the part frame. Reject
                    # jointly empty cuts, not only individually distant planes.
                    closest=c.copy()
                    for normal,offset in zip(n,b):closest-=max(normal@closest-offset,0)*normal
                    if np.linalg.norm(closest-c)>rad:continue
                    spheres.append((torch.from_numpy(c),rad))
                if not len(tri) and not spheres:raise ValueError('Empty actual native region')
                self.patch[part,region]=(torch.from_numpy(tri),spheres,torch.from_numpy(n),torch.from_numpy(b))

    def distances(self,q,points,active,surfaces,regions,*,world_frame=None,
                  _normal_heights=None,_margin=None,per_region=False):
        # Actual task contract is horizontal primary faces. No contact inference.
        if q.device.type!='cpu' or q.dtype!=torch.float64:
            raise ValueError('Native contact position requires CPU float64 geometry; tensor transfers preserve autograd')
        if q.shape!=(len(q),36) or points.shape!=(len(q),6,3) or active.shape!=(len(q),6) or surfaces.shape!=active.shape:
            raise ValueError('Contact-position batch shape mismatch')
        valid=torch.arange(4)[None]<torch.tensor([4,4,2,2,3,3])[:,None]
        if regions is None:regions=valid[None].expand(len(q),-1,-1)
        if regions.shape!=(len(q),6,4) or bool((active&~regions.any(-1)).any()) or bool((regions&~valid[None]).any()):
            raise ValueError('Active contact position requires valid nonempty anatomical regions')
        if not bool(torch.isfinite(q).all()&torch.isfinite(points).all()):raise ValueError('Nonfinite contact position input')
        points=points.detach()
        heights=(self.surface_heights(active,surfaces,q) if _normal_heights is None else _normal_heights)
        margin=q.new_full((len(q),),self.metadata['configured_margin']) if _margin is None else _margin
        if heights.shape!=surfaces.shape or margin.shape!=(len(q),) or not bool(torch.isfinite(heights).all()&torch.isfinite(margin).all()&(margin>0).all()):
            raise ValueError('Invalid per-scene native normal interval')
        p,r=self.fk.link_poses(q,BODY_NAMES[1:7]);r=_matrix_from_rotation6d(_rotation6d(r))
        if world_frame is not None:
            origin,basis=world_frame
            basis=_matrix_from_rotation6d(_rotation6d(basis.to(q)))
            p=origin.to(q)[:,None]+torch.einsum('bij,bpj->bpi',basis,p)
            r=basis[:,None]@r
        result=q.sum(-1,keepdim=True).expand(-1,6)*0
        regional=q.sum(-1)[:,None,None].expand(-1,6,4)*0 if per_region else None
        for part in range(6):
            costs=[]
            for region in range((4,4,2,2,3,3)[part]):
                ids=(active[:,part]&regions[:,part,region]).nonzero().flatten()
                if not len(ids):continue
                a=points[ids,part].detach().clone();a[:,2]=heights[ids,part]
                b=a.clone();b[:,2]+=margin[ids]
                tri,spheres,n,offset=self.patch[part,region]
                values=[]
                if len(tri):
                    world=p[ids,part,None,None]+torch.einsum('bij,vtj->bvti',r[ids,part],tri.to(q))
                    values.append(segment_triangles_squared(a,b,world))
                for center,rad in spheres:
                    values.append(sphere_patch_batch_squared(center.to(q),rad,n.to(q),offset.to(q),
                        a,b,p[ids,part],r[ids,part]))
                distance=torch.stack(values).amin(0)
                cost=q.new_full((len(q),),torch.inf);cost=cost.index_put((ids,),distance)
                if per_region:
                    regional[:,part,region]=q.new_zeros(len(q)).index_put((ids,),distance)
                costs.append(cost)
            if costs:result[:,part]=torch.stack(costs).amin(0)
        if per_region:
            return regional
        return torch.where(active,result,0.)

    def surface_heights(self,active,surfaces,q):
        heights=torch.zeros_like(surfaces,dtype=q.dtype)
        known=torch.zeros_like(active)
        for face in self.metadata['surface_catalog']:
            selected=active&(surfaces==face['surface'])
            if bool(selected.any()):
                if not np.allclose(face['normal_w'],[0,0,1],atol=1e-7,rtol=0):
                    raise ValueError('Not a shared horizontal primary face')
                heights[selected]=face['plane_offset'];known|=selected
        if bool((active&~known).any()):raise ValueError('Missing actual primary surface')
        return heights


class NativeContactPositionRouter:
    """Scene routing for the differentiable scalar and its geometric audit.

    Native geometry runs in CPU float64. Ordinary Torch device conversions
    preserve the gradient to the original prediction tensor. Closest-feature
    selection supplies no pose correction or network target. The initialized
    geometry and canonical region bounds are immutable for this router.
    """
    def __init__(self,metadata_by_scene):
        self.metadata={int(key):value for key,value in metadata_by_scene.items()}
        if not self.metadata:raise ValueError('No native scenes for contact position')
        self.geometries={}
        self.cpu_fk=None
        self.bounds=None
        # Only identical initialized robot shapes may share one FK/geometry
        # batch. Scene planes, margins and contact truth remain independent.
        self.geometry_keys={key:json.dumps(dict(link_names=value['link_names'],
            shapes=[s for s in value['solid_geometry']['shapes'] if s['body'] is not None]),sort_keys=True)
            for key,value in self.metadata.items()}

    def __call__(self,fk,q,scene_ids,points,active,surfaces,regions,region_geometry,*,world_frame=None,
                 per_region=False):
        from somaforge_core.g1_kinematics import CanonicalG1ForwardKinematics
        low,high=region_geometry.low.detach().cpu(),region_geometry.high.detach().cpu()
        if self.cpu_fk is None:
            if not isinstance(fk,CanonicalG1ForwardKinematics) or any(p.requires_grad for p in fk.parameters()):
                raise ValueError('Native position requires fixed canonical robot kinematics')
            self.cpu_fk=CanonicalG1ForwardKinematics().double()
            self.cpu_fk.load_state_dict({k:v.detach().cpu().to(torch.float64) for k,v in fk.state_dict().items()},strict=True)
            self.bounds=(low.clone(),high.clone())
        elif not torch.equal(low,self.bounds[0]) or not torch.equal(high,self.bounds[1]):
            raise ValueError('Canonical region bounds changed during native geometry routing')
        pose=q.to(device='cpu',dtype=torch.float64)
        routes=scene_ids.detach().cpu()
        if routes.shape!=(len(q),):raise ValueError('Native position scene route shape mismatch')
        target=points.detach().to(device='cpu',dtype=torch.float64)
        active,surfaces=active.detach().cpu(),surfaces.detach().cpu()
        regions=None if regions is None else regions.detach().cpu()
        frame=None if world_frame is None else tuple(x.detach().to(device='cpu',dtype=torch.float64) for x in world_frame)
        distance=(pose.sum(-1)[:,None,None].expand(-1,6,4)*0 if per_region
                  else pose.sum(-1,keepdim=True).expand(-1,6)*0)
        heights=pose.new_zeros(len(pose),6);margins=pose.new_zeros(len(pose));groups={}
        for key in routes.unique().tolist():
            if key not in self.metadata:raise ValueError('Missing initialized contact-position scene')
            if key not in self.geometries:
                self.geometries[key]=NativeContactRegionGeometry(self.metadata[key],self.cpu_fk,low,high)
            selected=(routes==key).nonzero().flatten()
            heights[selected]=self.geometries[key].surface_heights(active[selected],surfaces[selected],pose)
            margins[selected]=self.metadata[key]['configured_margin']
            groups.setdefault(self.geometry_keys[key],[]).append((key,selected))
        for entries in groups.values():
            key=entries[0][0];selected=torch.cat([indices for _,indices in entries])
            value=self.geometries[key].distances(pose[selected],target[selected],active[selected],surfaces[selected],
                None if regions is None else regions[selected],
                world_frame=None if frame is None else tuple(x[selected] for x in frame),
                _normal_heights=heights[selected],_margin=margins[selected],per_region=per_region)
            distance=distance.index_copy(0,selected,value)
        return distance.to(q)

    def contract(self):
        return dict(schema=CONTACT_POSITION_SCHEMA,geometry='initialized Newton part mesh triangles and analytic sphere skins',
            normal_interval='zero to actual configured includemargin',
            position_metric='Euclidean distance to issued XY normal-gap segment; normal excess contributes outside interval',
            aggregation='mean squared distance over intended parts, then existing global RMS tolerance',
            contact_truth='separate actual Newton activation, allocation and primary-face/region checks',
            scene_fingerprints={str(key):value['solid_geometry']['model_fingerprint'] for key,value in self.metadata.items()})


def native_position_statistics(model,router,q,scene_ids,points_world,active,surfaces,observed,*,regions=None,world_frame=None):
    """One spatial definition for optimization and endpoint evaluation.

    Distance uses actual initialized shapes. Completeness separately requires
    allocated, primary-face-selected Newton contacts in the intended regions.
    Missing witnesses never suppress the geometric recovery gradient.
    """
    region_geometry=getattr(model,'region_geometry',None)
    if region_geometry is None:
        from contact_solver.contact_regions import ContactRegions
        region_geometry=getattr(model,'_native_position_regions',None)
        if region_geometry is None:
            region_geometry=ContactRegions(model.fk).to(q.device)
            model._native_position_regions=region_geometry
    distance=router(model.fk,q,scene_ids,points_world,active,surfaces,regions,region_geometry,world_frame=world_frame)
    error=distance.sum(-1)/active.sum(-1).clamp_min(1)
    with torch.no_grad():
        if observed.get('schema')=='newton_device_witness_batch_v1':
            pair=observed['pairs'];valid=pair['eligible'];sample=pair['sample'][valid]
            part=pair['part'][valid];face=pair['primary_surface'][valid]
            point=pair['geometry_point1_w'][valid].to(q)
        else:
            from somaforge_core.contact_face_selection import select_contact_pairs
            catalogs=observed.get('surface_catalog_by_sample')
            if catalogs is None:catalogs=[observed['surface_catalog']]*len(q)
            flat=[(i,p) for i,rows in enumerate(observed['pairs'])
                for p in select_contact_pairs([rows],catalogs[i])['contact_pairs'][0]]
            sample=torch.tensor([i for i,_ in flat],device=q.device,dtype=torch.long)
            part=torch.tensor([p['part'] for _,p in flat],device=q.device,dtype=torch.long)
            face=torch.tensor([p['surface'] for _,p in flat],device=q.device,dtype=torch.long)
            point=q.new_tensor([p['position_w'] for _,p in flat]).reshape(-1,3)
        matching=active[sample,part]&(surfaces[sample,part]==face)
        if regions is not None and len(sample):
            if world_frame is not None:
                origin,basis=world_frame
                point=torch.einsum('nji,nj->ni',basis[sample].to(q),point-origin[sample].to(q))
            region=region_geometry.witness_regions(model.fk,q,sample,part,point)
            matching&=regions[sample,part,region]
        present=torch.zeros_like(active)
        present[sample[matching],part[matching]]=True
        complete=active.any(-1)&(present|~active).all(-1)
    return error,complete,present.sum(-1)
