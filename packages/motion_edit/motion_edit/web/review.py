"""Read-only motion/research playback in the shared Contact Editor.

Reports are observations, never a source of fresh contact labels. In particular
old ``actual_contact`` flags are displayed as historical diagnostics only.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable
import xml.etree.ElementTree as ET

import numpy as np
from somaforge_core.robot_assets import canonical_g1_urdf_path, decode_robot_asset_json

from .motion_data import load_motion_sequence


def _validate_pose(q, names):
    values = np.asarray(q, dtype=float)
    if values.shape != (7 + len(names),) or not np.isfinite(values).all():
        raise ValueError("review pose must contain finite root xyz, wxyz and named joints")
    if not np.isclose(np.linalg.norm(values[3:7]), 1., atol=1e-3):
        raise ValueError("review root quaternion must be normalized (wxyz)")
    return values.tolist()


def _validate_names(names):
    expected = {j.attrib['name'] for j in ET.parse(canonical_g1_urdf_path()).getroot().findall('joint')
                if j.attrib.get('type') != 'fixed'}
    if len(names) != len(set(names)) or set(names) != expected:
        raise ValueError("review requires the complete canonical G1 joint names")


def _base(title, qpos, names, terrain, url):
    return dict(
        motion_id=title, source_motion_id=title, provenance='diagnostic', read_only=True,
        capabilities=dict(playback=True, edit_contacts=False, generate=False),
        fps=50, qpos=qpos, joint_names=names,
        robot_urdf_url=url(canonical_g1_urdf_path()), terrain_obj_url=url(terrain) if terrain else None,
        graph=dict(anchors=[], transitions=[]), edit_handles=[], surfaces=[],
        contact_force=dict(part_order=[], forces=[], masks=[], positions=[], position_valid=[],
                           sample_points=[], sample_forces=[]),
        pending_edit_count=0, can_undo=False, can_redo=False, settings={}, plan=None,
        generation=dict(status='idle', stage=None, output_motion_path=None, output_motion_id=None,
                        warnings=[], error=None, started_at=None, finished_at=None),
    )


def _relative_path(value, report: Path):
    if not value:
        return None
    candidate = Path(value).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (report.parent / candidate).resolve()


def _case_review(path, data, *, url, terrain, case_index, reference_index):
    cases = data['cases']
    if not isinstance(cases, list) or not cases:
        raise ValueError('review collection requires a nonempty cases list')
    if not 0 <= case_index < len(cases):
        raise ValueError('review case index is out of range')
    entries = []
    for index, row in enumerate(cases):
        motion = _relative_path(row.get('motion') or row.get('path'), path)
        if motion is None or motion.suffix.lower() != '.npz':
            raise ValueError(f'review case {index} requires an NPZ motion')
        entries.append(dict(index=index, label=str(row.get('name') or motion.stem),
                            motion=motion, terrain=terrain or _relative_path(row.get('terrain') or data.get('terrain'), path)))
    selected = entries[case_index]
    payload = load_review(selected['motion'], url=url, terrain=selected['terrain'])
    row = cases[case_index]
    if reference_index is None:
        # Only pair records that explicitly identify the same corpus segments.
        reference_index = next((i for i, other in enumerate(cases)
                                if i != case_index and other.get('reference') is True
                                and row.get('corpus_indices')
                                and row['corpus_indices'] == other.get('corpus_indices')), -1)
    if reference_index < -1 or reference_index >= len(cases):
        raise ValueError('review reference index is out of range')
    if reference_index == case_index:
        reference_index = -1
    review = payload['review']
    review['collection'] = dict(path=str(path), case_index=case_index, reference_index=reference_index,
                                terrain_override=str(terrain) if terrain else None,
                                cases=[dict(index=e['index'], label=e['label']) for e in entries])
    review['segments'] = []
    for segment in row.get('segments', []):
        start, length = int(segment['first_frame']), int(segment['frames'])
        if start < 0 or length <= 0 or start + length > len(payload['qpos']):
            raise ValueError('review segment is outside the motion')
        review['segments'].append(dict(start=start, end=start + length - 1,
                                       label=str(segment.get('corpus_index', start))))
    depth = row.get('native_depth_mm')
    if depth is not None:
        values = np.asarray(depth, dtype=float)
        if values.shape != (len(payload['qpos']),) or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError('native_depth_mm must contain one finite nonnegative value per frame')
        review['depth_mm'] = values.tolist()
    # Aggregate booleans in older manifests lack raw solver evidence. Never turn
    # these into current contact masks or use them as an editing layer.
    review['notice'] += ' 案例清单中的接触布尔值不作为已验证接触；穿透值仅为报告诊断。'
    if reference_index >= 0:
        reference = entries[reference_index]
        loaded = load_review(reference['motion'], url=url, terrain=reference['terrain'])
        review['comparison'] = dict(label=reference['label'], qpos=loaded['qpos'], fps=loaded['fps'],
                                    joint_names=loaded['joint_names'], terrain_obj_url=loaded['terrain_obj_url'])
    payload['motion_id'] = selected['label']
    return payload


def load_review(path: Path, *, url: Callable, terrain: Path | None = None,
                case_index: int = 0, reference_index: int | None = None):
    """Validate a canonical NPZ sequence or a pose-record JSON report."""
    if path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as data:
            decode_robot_asset_json(data.get('robot_asset_json'), context=str(path))
        qpos, fps, names = load_motion_sequence(path)
        _validate_names(names)
        if not len(qpos) or not np.isfinite(fps) or fps <= 0:
            raise ValueError('review requires nonempty motion and positive fps')
        poses = [_validate_pose(q, names) for q in qpos]
        payload = _base(path.stem, poses, list(names), terrain, url)
        payload['fps'] = fps
        payload['review'] = dict(kind='motion', source=str(path),
                                notice='只读运动回放；未加载求解器接触证据，接触状态未知。')
        return payload
    if path.suffix.lower() != '.json':
        raise ValueError('review accepts canonical .npz motion or .json pose report')
    data = json.loads(path.read_text())
    if 'cases' in data:
        return _case_review(path, data, url=url, terrain=terrain,
                            case_index=case_index, reference_index=reference_index)
    if data.get('kind') == 'support_slip':
        from somaforge_core.contact_face_selection import select_contact_pairs
        from somaforge_core.newton_contact_data import load_raw_contact_labels
        from scipy.spatial.transform import Rotation
        motion=Path(data['motion']);contacts=Path(data['contacts'])
        if not motion.is_absolute():motion=(path.parent/motion).resolve()
        if not contacts.is_absolute():contacts=(path.parent/contacts).resolve()
        load_raw_contact_labels(motion,contacts)
        with np.load(motion,allow_pickle=False) as source:
            decode_robot_asset_json(source.get('robot_asset_json'),context=str(motion))
            if 'newton_direct_fk_metadata_json' not in source:
                raise ValueError('Support motion review requires certified Newton FK')
            body_names=source['body_names'].astype(str).tolist()
            positions=source['body_pos_w'].copy();quats=source['body_quat_w'].copy()
        rotations=Rotation.from_quat(quats[..., [1,2,3,0]].reshape(-1,4)).as_matrix().reshape(*quats.shape[:-1],3,3)
        qpos,fps,names=load_motion_sequence(motion);_validate_names(names)
        with np.load(contacts,allow_pickle=False) as labels:
            semantics=json.loads(labels['contact_semantics_json'].item())
            selected=select_contact_pairs(json.loads(labels['contact_pairs_json'].item()),semantics['scene']['surface_catalog'])
        frames=[]
        for t,row in enumerate(selected['contact_pairs']):
            points=[]
            if t<len(qpos)-1:
                for pair in row:
                    b=body_names.index(pair['body_name']);point=np.asarray(pair['position_w'],float)
                    local=rotations[t,b].T@(point-positions[t,b])
                    delta=positions[t+1,b]-positions[t,b]+(rotations[t+1,b]-rotations[t,b])@local
                    normal=np.asarray(pair['normal_w'],float);delta-=normal*np.dot(delta,normal)
                    points.append(dict(part=int(pair['part']),surface=int(pair['surface']),body=pair['body_name'],position=point.tolist(),
                                       delta=delta.tolist(),motion_cm=float(np.linalg.norm(delta)*100)))
            frames.append(points)
        mesh=terrain or _relative_path(data.get('terrain'), path)
        payload=_base(data.get('title',path.stem),[_validate_pose(q,names) for q in qpos],list(names),Path(mesh) if mesh else None,url)
        payload['fps']=fps
        payload['review']=dict(kind='support_slip',source=str(path),slip_frames=frames,
            notice='实际 Newton 接触样本的同材料点切向位移；绿色为各部位最低运动点，橙色为其他点。几何支点存在不等于整个区域无滑动；此轨迹无受力证据，承重滑移未知。')
        from somaforge_core.support_semantics import SUPPORT_ASSESSMENT_SCHEMA
        payload['review']['support_assessment_schema'] = SUPPORT_ASSESSMENT_SCHEMA
        payload['review']['actual_support_status'] = 'unknown_without_same_solve_loads'
        if data.get('support_observations'):
            from somaforge_core.support_evidence import load_support_observations
            evidence = Path(data['support_observations'])
            if not evidence.is_absolute(): evidence = path.parent / evidence
            assessed = load_support_observations(motion, evidence)
            if len(assessed['load_known']) != len(qpos):
                raise ValueError('Support review observation/motion clock mismatch')
            payload['review']['support_frames'] = [
                [dict(part=p,load_known=bool(assessed['load_known'][t,p]),
                      load_state=str(assessed['load_state'][t,p]),
                      normal_force_n=float(assessed['normal_force_n'][t,p]) if assessed['load_known'][t,p] else None,
                      loaded_speed_cm_s=float(assessed['loaded_tangent_speed_m_s'][t,p]*100)
                          if assessed['loaded_motion_known'][t,p] else None)
                 for p in range(6)] for t in range(len(qpos))]
            payload['review']['actual_support_status'] = 'observed_original_solve_samples'
            payload['review']['notice'] = ('箭头是位姿间的同材料点几何运动；下方载荷和速度来自原始 Newton 同次求解。'
                '两者采样时序不同。承重与滑移分开报告；没有标定速度预算，不能宣称无滑移。')
        return payload
    if data.get('kind') == 'support_motion':
        motion = Path(data['motion'])
        if not motion.is_absolute():
            motion = (path.parent / motion).resolve()
        with np.load(motion, allow_pickle=False) as source:
            decode_robot_asset_json(source.get('robot_asset_json'), context=str(motion))
        qpos, fps, names = load_motion_sequence(motion)
        _validate_names(names)
        if not len(qpos) or not np.isfinite(fps) or fps <= 0:
            raise ValueError('review requires nonempty motion and positive fps')
        poses = [_validate_pose(q, names) for q in qpos]
        phases = []
        for row in data['phases']:
            start, end = int(row['start']), int(row['end'])
            if start < 0 or end < start or end >= len(poses):
                raise ValueError('support phase is outside the motion')
            trajectory = np.asarray(row['aggregate'], dtype=float)
            raw = np.asarray(row['rollouts'], dtype=float)
            if (trajectory.shape != (end-start+1, 3) or raw.ndim != 3
                    or raw.shape[1:] != trajectory.shape or not len(raw)
                    or not np.isfinite(trajectory).all() or not np.isfinite(raw).all()):
                raise ValueError('support trajectories must be finite XYZ arrays for the phase')
            phases.append(dict(start=start, end=end, part=str(row['part']),
                               aggregate=trajectory.tolist(), rollouts=raw.tolist(),
                               aggregate_net_cm=float(row['aggregate_net_cm']),
                               rollout_median_net_cm=float(row['rollout_median_net_cm'])))
        mesh = terrain or _relative_path(data.get('terrain'), path)
        payload = _base(path.stem, poses, list(names), Path(mesh) if mesh else None, url)
        payload['fps'] = fps
        payload['review'] = dict(kind='support_motion', source=str(path), phases=phases,
                                 notice='固定材料点运动诊断；轨迹和位移不代表接触激活、承重或摩擦滑移。')
        return payload
    decode_robot_asset_json(data.get('robot_asset_json'), context=str(path))
    names = list(data['joint_names'])
    _validate_names(names)
    records = []
    seen = set()
    for row in data['records']:
        label = str(row['label'])
        if label in seen:
            raise ValueError(f'duplicate review record: {label}')
        seen.add(label)
        match = re.fullmatch(r'(.+)_(initial|final|step\d+)', label)
        method, iteration = match.groups() if match else (label, 'pose')
        pairs = row.get('penetrating_pairs', {})
        starts = np.asarray(pairs.get('geometry_point1_w', []), dtype=float).reshape(-1, 3)
        normals = np.asarray(pairs.get('normal_w', []), dtype=float).reshape(-1, 3)
        if starts.shape != normals.shape or not np.isfinite(starts).all() or not np.isfinite(normals).all():
            raise ValueError(f'invalid penetration vectors in {label}')
        # Keep the original record, including provenance/solver fields, for audit.
        records.append(dict(label=label, method=method, iteration=iteration,
                            q=_validate_pose(row['q'], names), diagnostics=row,
                            normal_starts=starts.tolist(), normals=normals.tolist()))
    if not records:
        raise ValueError('review report has no records')
    original = next((row for row in records if row['label'] == 'original'), records[0])
    mesh = terrain or _relative_path(data.get('terrain'), path)
    payload = _base(path.stem, [original['q']], names, mesh, url)
    payload['review'] = dict(kind='report', source=str(path), records=records,
                            reference=original['label'],
                            notice='历史诊断回放；报告接触标记未经当前主表面规则重新验证，不作为接触真值。')
    return payload
