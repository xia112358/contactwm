"""Associate native support rows with the original goal's body-local material."""
import functools,hashlib,json
from pathlib import Path
import torch
from .contact_stability import CONTRACT as STABILITY
from .motion import CONTROL_DT

@functools.lru_cache(maxsize=64)
def source_context(source,sha):
    source=Path(source)
    if hashlib.sha256(source.read_bytes()).hexdigest()!=sha:raise ValueError('Material source hash mismatch')
    raw=torch.load(source,map_location='cpu',weights_only=False)
    if not raw['report']['successful_rollout'] or bool(raw['data']['term'].any()):raise ValueError('Material source is not a successful execution')
    contract=json.loads((source.parent/'substeps/contract.json').read_text())
    world=contract['world_task_ids'].index(int(raw['data']['xi'][0,33]))
    return world,torch.tensor(contract['world_origins'][world]),round(CONTROL_DT/contract['dt'])

@functools.lru_cache(maxsize=128)
def substeps(capture,frame):
    return torch.load(Path(capture)/'substeps'/f'{frame:06d}.pt',map_location='cpu',weights_only=False)['substeps']

def row_sample(source,sha,part,tick,row):
    w,origin,S=source_context(source,sha);r=substeps(str(Path(source).parent),tick//S)[tick%S]
    if not (r['eligible'][row] and r['active'][row] and r['allocated'][row] and r['groups'][row]==w*6+part and r['features']['contact_stable'][w,part]):
        raise ValueError('Material correspondence requires an allocated stable native row')
    p=r['part_position'][w,part]-origin;R=r['part_rotation'][w,part];q=r['rows']['position_w'][row]-origin
    force=r['rows']['force_on_body1_w'][row,2];load=float(force)*(1 if int(r['robot_side'][row])==1 else -1)
    return dict(source=source,source_sha256=sha,solver_tick=tick,native_row=row,point=q,local=R.T@(q-p),position=p,rotation=R,load=load)

def associated_material_target(event):
    """Keep original world goal and match distinct executions in body coordinates."""
    witness=row_sample(event['witness_source'],event['witness_source_sha256'],event['part'],event['solver_tick'],event['native_row'])
    if not torch.allclose(witness['point'],torch.tensor(event['point']),atol=1e-6,rtol=0):raise ValueError('Original target and native witness differ')
    anchor=witness['local'];nearest=[]
    for c in event['contributors']:
        best=None
        for tick,row,aligned in zip(c['ticks'],c['rows'],c['aligned_frames']):
            point=row_sample(c['source'],c['source_sha256'],event['part'],tick,row)
            if point['load']<=0:continue
            distance=float((point['local']-anchor).norm())
            if best is None or distance<best['distance_m']:best=dict(source=c['source'],source_sha256=c['source_sha256'],solver_tick=tick,native_row=row,aligned_frame=aligned,local=point['local'].tolist(),world=point['point'].tolist(),distance_m=distance,normal_load_n=point['load'])
            if distance<1e-5:break
        if best is not None:nearest.append(best)
    radius=STABILITY['row_match_distance_m'];matched=[c for c in nearest if c['distance_m']<=radius];selection='original_native_material'
    if len(matched)<2 and event['part'] in (2,3):
        # Two successful executions on the same hand surface can be less than
        # two row-match radii apart but neither exact row covers the other.
        # Project their midpoint to the actual convex collision surface.
        import numpy as np,trimesh
        from scipy.spatial import ConvexHull
        from .contact_placement import solver_part_meshes
        near=sorted(nearest,key=lambda x:x['distance_m'])[:2]
        if len(near)==2 and near[1]['distance_m']<=2*radius:
            vertices=np.asarray(solver_part_meshes(Path(near[0]['source']).parent)[event['part']]);tri=vertices[ConvexHull(vertices).simplices]
            midpoint=(np.asarray(near[0]['local'])+np.asarray(near[1]['local']))/2
            query=np.broadcast_to(midpoint,(len(tri),3));surface=trimesh.triangles.closest_point(tri,query);anchor=torch.as_tensor(surface[np.linalg.norm(surface-query,axis=1).argmin()],dtype=anchor.dtype)
            for c in nearest:c['distance_m']=float((torch.tensor(c['local'])-anchor).norm())
            matched=[c for c in nearest if c['distance_m']<=radius];selection='common_native_collision_surface'
    if len(matched)<2:raise ValueError(f'Insufficient material-matched successful executions for case {event["case"]} part {event["part"]}')
    target_witness=witness
    if witness['load']<=0:
        # Old aggregation may choose an allocated but unloaded row; retain
        # its goal/material identity, but anchor the target to a loaded match.
        chosen=min(matched,key=lambda c:float((torch.tensor(c['world'])-witness['point']).norm()))
        target_witness=row_sample(chosen['source'],chosen['source_sha256'],event['part'],chosen['solver_tick'],chosen['native_row'])
        selection='positive_load_matched_material'
    target=target_witness['position']+target_witness['rotation']@anchor
    return anchor,target,dict(selection=selection,original_goal=event['point'],goal_shift_m=float((target-witness['point']).norm()),witness_source=target_witness['source'],witness_tick=target_witness['solver_tick'],witness_row=target_witness['native_row'],original_witness_load_n=witness['load'],contributors=matched,excluded_contributors=[c for c in nearest if c not in matched])
