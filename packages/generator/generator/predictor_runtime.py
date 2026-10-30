"""Predictor endpoint audits independent of trajectory generation models."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import torch
from generator.scene_workers import Workers
from generator.planned_contact_predictor import plan_points_in_pose_frame, audit_intended_surfaces
from contact_solver.device_contact_objective import DeviceWitnessRows, acceptance
from contact_solver.native_contact_position import native_position_statistics
from contact_solver.interaction_acceptance import configured_acceptance
from somaforge_core.newton_contact_tensors import validate_observed
from somaforge_core.published_data import resolve_published_path

def evidence(path):
    path = resolve_published_path(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path.resolve()), sha256=digest)


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def cpu_tree(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu()
    if isinstance(value, dict): return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [cpu_tree(v) for v in value]
    return value


class NativeQueries:
    def __init__(self, workers, fk, batch_size):
        self.workers, self.fk, self.batch_size = workers, fk, batch_size
        self.names = tuple(workers.tensor_router.link_names)

    def observed(self, q, scene_ids):
        observed = self.workers.tensor_router(q.detach(), scene_ids)
        validate_observed(observed)
        return observed

    def depths(self, q, scene_ids, observed=None):
        if observed is None: observed = self.observed(q,scene_ids)
        rows = DeviceWitnessRows(self.fk, q, observed)
        p = observed['pairs']
        native = q.new_zeros(len(q),2)
        valid = p['full_kind'] >= 0
        group = p['sample'][valid]*2+p['full_kind'][valid]
        native.flatten().scatter_reduce_(0,group,(-p['dist'][valid]).relu(),reduce='amax',include_self=True)
        solid = self.workers.solid_router(self.fk,q,scene_ids)
        _, invalid = rows.full_body()
        return torch.maximum(native,solid.depths()), invalid, solid

    @torch.no_grad()
    def timeline(self, q, scene_id, audit_dir=None):
        masks, surfaces, depth, invalid = [], [], [], []
        for first in range(0,len(q),self.batch_size):
            chunk=q[first:first+self.batch_size]
            ids=torch.full((len(chunk),),scene_id,device=q.device,dtype=torch.long)
            observed=self.observed(chunk,ids)
            depths,bad,solid=self.depths(chunk,ids,observed)
            masks.append(observed['contact_part_mask'].cpu())
            surfaces.append(observed['contact_surface'].cpu())
            depth.append(depths.cpu()); invalid.append(bad.cpu())
            if audit_dir is not None:
                audit_dir.mkdir(parents=True,exist_ok=True)
                torch.save(dict(first=first,q=chunk.cpu(),observed=cpu_tree(observed),
                    solid_pairs=cpu_tree(solid.pair)),audit_dir/f'query_{first:06d}.pt')
        return dict(mask=torch.cat(masks),surface=torch.cat(surfaces),depth=torch.cat(depth),invalid=torch.cat(invalid))


@torch.no_grad()
def audit_prediction(model,query,prediction,q0,initial,scene_id,limits):
    q=prediction.qpos
    ids=torch.tensor([scene_id],device=q.device)
    observed=query.observed(q,ids)
    points=plan_points_in_pose_frame(q0[None],prediction.conditioned_points_local)
    active=prediction.conditioned_contact
    surface=(prediction.conditioned_surface if prediction.conditioned_surface is not None
        else audit_intended_surfaces(points,active,observed))
    audit=acceptance(observed,active,surface)
    depth,invalid,_=query.depths(q,ids,observed)
    error,complete,count=native_position_statistics(model,query.workers.position_router,q,ids,
        points,active,surface,observed,regions=prediction.planned_regions)
    reasons=[]
    if not bool(audit['contact_accepted'][0]): reasons.append('actual_contact_plan')
    if not bool(active.any()): reasons.append('empty_intent')
    rms=float(error[0].sqrt())
    if not bool(complete[0]) or not math.isfinite(rms) or rms>.04: reasons.append('actual_region_position')
    if not bool(torch.isfinite(depth).all()) or float(depth.max())>limits.shallow_penetration_m:
        reasons.append('terrain_or_self_penetration')
    if bool(invalid.any()): reasons.append('invalid_native_normal')
    keep=prediction.conditioned_role==2
    same_relation = acceptance(initial, keep, surface)['realized']
    if bool((keep & ~same_relation).any()):
        reasons.append('keep_without_current_same_surface_contact')
    if bool(((q[:,7:]<model.fk.joint_lower-1e-6)|(q[:,7:]>model.fk.joint_upper+1e-6)).any()): reasons.append('joint_limits')
    return observed,dict(accepted=not reasons,reasons=reasons,depth_m=depth[0].tolist(),
        position_rms_m=rms if math.isfinite(rms) else None,position_complete=bool(complete[0]),
        contact_accepted=bool(audit['contact_accepted'][0]),observed_parts=int(count[0]))
