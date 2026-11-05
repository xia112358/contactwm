"""Associate measured support episodes; never infer or fill native contact.

Actual solver meshes describe the occupied placement region, not activation.
All geometry below is used only after native support on both ends is known.
The anchor stays fixed through a chain of resumptions, preventing drift.
"""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from shapely.geometry import Polygon
from .contact_temporal import TEMPORAL_CONTRACT,LIMB_PARTS
from .contact_source_motion import SOURCE_MOTION_CONTRACT
from .contact_substep_support import control_support_episodes

PLACEMENT_CONTRACT=dict(version='native_same_placement_v1',
    max_unload_s=SOURCE_MOTION_CONTRACT['quiet_frames']*SOURCE_MOTION_CONTRACT['dt'],
    max_clearance_change_m=TEMPORAL_CONTRACT['motion_excursion_m'],
    geometry='actual serialized surface_v2 solver mesh, shared canonical body frames',
    identity='same part and measured terrain face; shrinking common footprint throughout each gap, never move the anchor to follow successive resumptions; no large vertical excursion or intervening main-form change',
    evidence='episode association only; native activation, force, measured support and point-loss masks are not filled through gaps')


def solver_part_meshes(capture):
    import mujoco
    from somaforge_core.robot_assets import validate_g1_asset_metadata
    from .contact_control import LINKS
    path=Path(capture)/'solver-collision.mjb';meta=json.loads(path.with_suffix('.mjb.json').read_text())
    validate_g1_asset_metadata(meta['robot_asset'],context='placement continuity collision model')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=meta['model_sha256']:raise ValueError('Placement geometry differs from captured native solver')
    m=mujoco.MjModel.from_binary_path(str(path));meshes=[]
    for name in LINKS:
        bodies=[b for b in range(m.nbody) if (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,b) or '').endswith('_'+name)]
        if len(bodies)!=1:raise ValueError('Missing/ambiguous placement body: '+name)
        b=bodies[0];vertices=[]
        for g in range(m.body_geomadr[b],m.body_geomadr[b]+m.body_geomnum[b]):
            if not (m.geom_contype[g] or m.geom_conaffinity[g]):continue
            if m.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH:raise ValueError('Placement geometry requires the captured surface_v2 mesh contract')
            mesh=m.geom_dataid[g];v=m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh]+m.mesh_vertnum[mesh]].copy()
            vertices.append(Rotation.from_quat(m.geom_quat[g][[1,2,3,0]]).apply(v)+m.geom_pos[g])
        v=np.concatenate(vertices);meshes.append(v[ConvexHull(v).vertices])
    return meshes


def load_placement_evidence(capture):
    """Read same-solve poses and attributed terrain faces at every native tick."""
    capture=Path(capture);contract=json.loads((capture/'substeps/contract.json').read_text())
    samples=[]
    for file in sorted((capture/'substeps').glob('*.pt')):
        samples.extend(torch.load(file,map_location='cpu',weights_only=False)['substeps'])
    positions=torch.stack([r['part_position'] for r in samples]);rotations=torch.stack([r['part_rotation'] for r in samples])
    surfaces=[];known=[]
    for r in samples:
        shape=r['observation']['active'].shape;G=shape[0]*shape[1]
        use=r['eligible']&r['active']&r['allocated'];groups=r['groups'][use];faces=r['surfaces'][use]
        low=torch.full((G,),torch.iinfo(torch.long).max,dtype=torch.long);high=torch.full((G,),-1,dtype=torch.long)
        low.scatter_reduce_(0,groups,faces,reduce='amin',include_self=True);high.scatter_reduce_(0,groups,faces,reduce='amax',include_self=True)
        # -1: no observed face; -2: more than one observed face. Neither is a
        # measured single-surface identity and neither qualifies an association.
        face=torch.where(high<0,-1,torch.where(low==high,low,-2))
        surfaces.append(face.reshape(shape));known.append(r['observation']['known'])
    return dict(position=positions.numpy(),rotation=rotations.numpy(),surfaces=torch.stack(surfaces).numpy(),known=torch.stack(known).numpy(),dt=contract['dt'],meshes=solver_part_meshes(capture),
        points=torch.stack([r['observation']['point'] for r in samples]),
        qualified_points=torch.stack([r['observation']['known']&r['observation']['active']&r['features']['contact_stable']&r['support']['event_ready'] for r in samples]),
        world_task_ids=contract['world_task_ids'],substeps=contract['support_contract']['substeps_per_control'],
        native_samples=samples,world_origins=torch.tensor(contract['world_origins']))


def footprint(vertices):
    return Polygon(vertices[ConvexHull(vertices[:,:2]).vertices,:2])


def placement_link(anchor,anchor_floor,poses,mesh):
    """Retain a common occupied region, rather than following a moving center.

    Intersection can only shrink across resumes. A return after leaving the
    region is a new placement even when its final pose matches the old pose.
    """
    retained=anchor;clearance=0.
    for p,R in zip(*poses):
        vertices=mesh@R.T+p
        retained=retained.intersection(footprint(vertices))
        clearance=max(clearance,float(abs(vertices[:,2].min()-anchor_floor)))
    fraction=float(retained.area/anchor.area)
    return dict(retained_fraction=fraction,clearance_change_m=clearance,
        matched=retained.area>0 and clearance<=PLACEMENT_CONTRACT['max_clearance_change_m']),retained


def associate_support_placements(data,evidence,world):
    episodes=control_support_episodes(data,physics_bounds=True);T,S,_=data['substep_support_state'].shape
    resume=torch.zeros(T,6,dtype=torch.bool);audit=[]
    base=int(data['ep_step'][0])*S;pos=evidence['position'][base:base+T*S,world];rot=evidence['rotation'][base:base+T*S,world];faces=evidence['surfaces'][base:base+T*S,world];known=evidence['known'][base:base+T*S,world]
    for parts in LIMB_PARTS:
        previous=None;anchor=None
        for ep in (e for e in episodes if e['part'] in parts):
            p=ep['part'];begin=ep['physics_start'];end=ep['physics_stop'];mesh=evidence['meshes'][p]
            joined=False
            if previous is not None and previous['part']==p:
                a=previous['physics_stop'];b=begin
                duration=(b-a)*evidence['dt'];face0=int(faces[max(0,a-1),p]);face1=int(faces[b,p]);observed=faces[max(0,a-1):b+1,p]
                same_face=face0>=0 and face1==face0 and bool(((observed==-1)|(observed==face0)).all())
                qualified=0<=duration<=PLACEMENT_CONTRACT['max_unload_s'] and same_face and bool(known[max(0,a-1):b+1,p].all())
                item=dict(part=p,previous_start=previous['start'],previous_stop=previous['stop'],start=ep['start'],confirmed=ep['confirmed'],gap_s=duration,same_face=same_face,linked=False)
                if qualified:
                    metric,retained=placement_link(anchor,anchor_floor,(pos[max(0,a-1):b+1,p],rot[max(0,a-1):b+1,p]),mesh)
                    item.update(metric);joined=metric['matched'];item['linked']=joined
                audit.append(item)
                if joined:resume[ep['start'],p]=True;anchor=retained
            if not joined:
                vertices=mesh@rot[begin,p].T+pos[begin,p];anchor=footprint(vertices);anchor_floor=float(vertices[:,2].min())
            previous=ep
    return dict(support_placement_resume=resume),audit


def placement_episodes(data,episodes):
    """Group events without rewriting measured support spans or contact forms."""
    groups=[];resumptions=[]
    for parts in LIMB_PARTS:
        previous=None
        for ep in (e for e in episodes if e['part'] in parts):
            p=ep['part']
            resume=previous is not None and previous['part']==p and bool(data['support_placement_resume'][ep['start'],p])
            if resume:
                resumptions.append(dict(part=p,start=previous['stop'],stop=ep['start'],resume=ep['start']))
                previous['stop']=ep['stop']
            else:
                previous=dict(ep);groups.append(previous)
    return sorted(groups,key=lambda e:(e['start'],e['part'])),resumptions
