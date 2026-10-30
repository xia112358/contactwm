"""Rebuild native contact labels, event evidence and caches as a new dataset.

Source motions and authored action timestamps stay immutable. Contact targets
are recomputed by current Newton/MJWarp and the shared primary face selection.
"""
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class Config:
    source_manifest: Path
    output: Path
    worker_output: Path
    query_checkpoint: Path
    query_source_manifest: Path
    query_source_model: Path
    query_worlds: int = 32
    batch_size: int = 128


def rebuild(cfg, device):
    from generator.scene_workers import Workers
    from generator.support_path_acceptance import file_evidence
    from somaforge_core.published_data import load_published_json
    from somaforge_core.newton_contact_query import install_contact_query, query_contacts
    from somaforge_core.newton_contact_data import require_current_newton_manifest
    from relabel_newton_motion import relabel
    from prepare_predictor_event_paths import prepare
    from prepare_predictor_training_data import build_cache

    cfg.output = cfg.output.resolve()
    cfg.worker_output = cfg.worker_output.resolve()
    cfg.output.mkdir(parents=True, exist_ok=False)
    source = load_published_json(cfg.source_manifest)
    rows = source['motion_files']
    heights = [round(float(load_published_json(row['edit_plan_file'])['metadata']['height_scale']), 8) for row in rows]
    rebuilt, declarations, audit = [], [], []
    with Workers(cfg.worker_output, sorted(set(heights)), device,
                 checkpoint=cfg.query_checkpoint, source_manifest=cfg.query_source_manifest,
                 source_model=cfg.query_source_model, query_worlds=cfg.query_worlds) as workers:
        runtime = None
        try:
            for index, (row, height) in enumerate(zip(rows, heights, strict=True)):
                folder = cfg.output / 'certificates' / f'{index:04d}'
                motion = Path(row['motion_file'])
                before = file_evidence(motion)
                worker = workers.entries[height]
                endpoint = worker['endpoint']
                install_contact_query(lambda q, scene, endpoint=endpoint: query_contacts(q, scene, endpoint=endpoint))
                labels = folder / 'contacts.npz'
                relabel(motion, worker['folder'] / 'binding.json', labels, cfg.batch_size)
                prepare(motion, labels, Path(row['event_segments_file']), folder)
                single = json.loads((folder / 'manifest.json').read_text())
                require_current_newton_manifest(single, context='rebuilt Predictor labels')
                if runtime is not None and runtime != single['newton_runtime']:
                    raise ValueError('Mixed Newton runtimes')
                runtime = single['newton_runtime']
                rebuilt.append(dict(row, **single['motion_files'][0]))
                declarations.append(dict(single['demonstration_event_evidence'][0], motion_id=index))
                if file_evidence(motion) != before:
                    raise ValueError('Source motion changed during relabeling')
                audit.append(dict(motion=before, labels=file_evidence(labels), height=height,
                                  model_fingerprint=worker['fp'], motion_unchanged=True))
                print(json.dumps(dict(completed=index+1, total=len(rows))), flush=True)
        finally:
            install_contact_query(None)
    manifest = dict(source, motion_files=rebuilt, demonstration_event_evidence=declarations,
                    newton_runtime=runtime, training_ready=False,
                    rematerialization=dict(source_manifest=file_evidence(cfg.source_manifest),
                        method='actual_current_newton_mjwarp_fixed_q_shared_primary_face_selection',
                        action_clock='source timestamps only; contact observations and targets rebuilt',
                        old_contact_targets_reused=False, source_motions_modified=False))
    path = cfg.output / 'manifest.json'
    path.write_text(json.dumps(manifest, indent=2)+'\n')
    summary = build_cache(path, cfg.output)
    (cfg.output / 'rematerialization_audit.json').write_text(json.dumps(dict(
        motions=audit, cache=summary, output_manifest=file_evidence(path)), indent=2)+'\n')
    return summary


def main():
    import tyro
    from somaforge_core.launcher import parse_isaaclab_launcher_args
    official = parse_isaaclab_launcher_args(__doc__)
    cfg = tyro.cli(Config)
    print(json.dumps(rebuild(cfg, official.device), indent=2))


if __name__ == '__main__':
    main()
