"""Publish a bounded training run's live status to the existing static server."""
import argparse
import json
from pathlib import Path
import shutil
import time


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def rows(path):
    if not path.exists():
        return []
    result = []
    for line in path.read_text().splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return result


def recent_native(current, baseline):
    """Recover completed-attempt window statistics from cumulative counters."""
    result = dict(current)
    previous = {s['segment']: s for s in baseline['segments']}
    segments = []
    for item in current['segments']:
        old = previous[item['segment']]
        n = item['attempts'] - old['attempts']
        wins = item['successful_attempts'] - old['successful_attempts']
        row = dict(item, attempts=n, successful_attempts=wins,
                   success_rate=wins/n if n else None)
        for key in ('final_root_error_m', 'final_contact_error_m'):
            total = (item[key] or 0)*item['attempts'] - (old[key] or 0)*old['attempts']
            row[key] = max(0., total/n) if n else None
        segments.append(row)
    covered = [s for s in segments if s['attempts']]
    n = sum(s['attempts'] for s in segments)
    wins = sum(s['successful_attempts'] for s in segments)
    result.update(segments=segments, attempts=n, successful_attempts=wins,
        success_rate=wins/n if n else None,
        macro_success_rate=sum(s['success_rate'] for s in covered)/len(covered) if covered else None,
        covered_goals=len(covered),
        mean_root_error_m=sum(s['final_root_error_m']*s['attempts'] for s in covered)/n if n else None,
        window_start_iteration=baseline['iteration'], window_end_iteration=current['iteration'],
        metric_scope='completed attempts within recent iteration window; warmup excluded')
    return result


def snapshot(run, record):
    config = read(run/'config.json', {})
    state = read(run/'status.json', {})
    process = read(record, {})
    proc = Path('/proc')/str(process.get('pid', -1))/'stat'
    try:
        fields = proc.read_text().split()
        alive = fields[21] == process.get('start_identity') and fields[2] != 'Z'
    except FileNotFoundError:
        alive = False
    if config.get('independent_segments'):
        evaluations=[]
        for path in sorted(run.glob('native_*.json')):
            report=read(path,{})
            if report.get('independent_segments'):
                evaluations.append(report)
        evaluations.sort(key=lambda r:r['iteration'])
        latest_native=state.get('native')
        if latest_native and evaluations and latest_native['iteration'] >= evaluations[-1]['iteration']:
            evaluations=[r for r in evaluations if r['iteration'] < latest_native['iteration']]+[latest_native]
        window_updates=5*config.get('log_every',5)
        recent=[]
        for report in evaluations:
            cutoff=report['iteration']-window_updates
            baseline=next((r for r in reversed(evaluations) if r['iteration']<=cutoff),evaluations[0])
            recent.append(recent_native(report,baseline))
        phase=state.get('phase','starting')
        kind='running'
        labels=dict(segment_evaluation='正在评估各单段',wm_calibration='正在校准 WM',training='正在学习目标接触',
                    collecting='正在采集训练数据',training_collection='正在采集训练数据',
                    complete='本轮已结束',task_success='全部单段已成功',interrupted='已停止',failed='运行出错')
        label=labels.get(phase,'正在启动')
        if config.get('freeze_actor') and phase in ('training','collecting'):
            label='正在验证固定策略（不更新权重）'
        if phase in ('complete','task_success'):kind='done'
        elif phase in ('interrupted','failed'):kind='error'
        elif not alive:label,kind='进程已退出，未记录正常结束','error'
        metrics=rows(run/'metrics.jsonl')[-100:]
        progress=state.get('evaluation_progress') if phase=='segment_evaluation' else None
        return dict(updated_at=time.time(),training_updated_at=state.get('updated_at'),mode='wm',continuous_random=config.get('continuous_random',False),sequence_length=config.get('sequence_length',1),policy_gradient=config.get('policy_gradient','imagination'),dmo_processed=state.get('dmo_processed'),dmo_total=state.get('dmo_total'),label=label,kind=kind,
            alive=alive,phase=phase,iteration=state.get('iteration',0),policy_updates=state.get('policy_updates',0),
            wm_updates=state.get('wm_updates',0),max_iterations=config.get('start_iteration',0)+config.get('iterations',0),
            contact_face_matching=config.get('contact_face_matching',True),completion_mode=config.get('completion_mode','onset'),root_tolerance=config.get('root_tolerance',.1),contact_tolerances_m=config.get('contact_tolerances_m'),stop_rule=config.get('stop_rule'),horizon=config.get('horizon'),wm_window=config.get('wm_window'),gate=state.get('gate'),
            wm_loss=state.get('wm_loss'),policy_loss=state.get('policy_loss'),progress=progress,
            prior_weight=config.get('prior_weight',1.),pose_weight=config.get('pose_weight',1.),segment_grace_steps=config.get('segment_grace_steps',10),
            native=recent[-1] if recent else None,native_cumulative=state.get('native'),window_updates=window_updates,best=read(run/'best-native.json'),
            curve=[dict(iteration=r['iteration'],success_rate=r.get('macro_success_rate',r.get('success_rate',r.get('successful_goals',0)/r['total_goals'])),root_error=r['mean_root_error_m'],covered_goals=r['covered_goals']) for r in recent],
            timing=state.get('timing') or (metrics[-1] if metrics else None),num_envs=config.get('num_envs'),sim_steps=state.get('sim_steps'),
            wm_frozen=config.get('freeze_wm',False),wm_frozen_unchanged=state.get('wm_frozen_unchanged'),
            latest=dict(iteration=state.get('iteration',0),updates=state.get('policy_updates',0),
                        evaluating_segment=progress.get('completed') if progress else None),
            error=state.get('error'),run_name=run.name,metrics=metrics)
    current = rows(run/'dagger/metrics.jsonl')
    history = []
    if config.get('dagger_resume'):
        history = rows(Path(config['dagger_resume']).parent/'metrics.jsonl')
    merged = {}
    for row in history + current:
        values = row['student_state_action_mse']
        merged[row['round']] = dict(round=row['round'], updates=row['updates'],
            mse=sum(values)/len(values), segments=row['covered_segments'])
    curve = [merged[k] for k in sorted(merged)]
    stale = 0
    reference = None
    for row in current:
        value = sum(row['student_state_action_mse'])/len(row['student_state_action_mse'])
        threshold = max(config.get('bc_min_improvement', .005),
                        config.get('bc_relative_improvement', .02)*reference) if reference is not None else 0
        if reference is None or value < reference-threshold:
            reference, stale = value, 0
        elif row['round'] >= config.get('dagger_min_rounds', 10):
            stale += 1
    summary = read(run/'dagger/summary.json')
    phase = state.get('phase', 'starting')
    if phase in ('complete', 'task_success'):
        label, kind = '已结束', 'done'
    elif phase in ('failed', 'interrupted'):
        label, kind = ('运行出错' if phase == 'failed' else '已中断'), 'error'
    elif not alive:
        label, kind = '进程已退出，未记录正常结束', 'error'
    elif summary:
        label, kind = '初始化已停止，正在最终评估', 'evaluating'
    else:
        label, kind = '训练中', 'running'
    return dict(updated_at=time.time(), training_updated_at=state.get('updated_at'),
        label=label, kind=kind, alive=alive, phase=phase, curve=curve,
        latest=curve[-1] if curve else None, plateau_count=stale,
        plateau_patience=config.get('bc_patience', 4), max_rounds=config.get('dagger_rounds'),
        min_improvement=config.get('bc_min_improvement', .005),
        relative_improvement=config.get('bc_relative_improvement', .02),
        summary=({k: summary.get(k) for k in ('stop_reason', 'rounds', 'updates', 'best_round', 'best_updates', 'plateau_metric')}
                 if summary else None), error=state.get('error'),
        native_evaluation=read(run/'native_dagger.json'),
        run_name=run.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--process-record', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--kind',choices=('dagger','wm'),default='dagger')
    parser.add_argument('--comparison-run',type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    template=Path(__file__).with_name('contact_wm_monitor.html') if args.kind=='wm' else Path(__file__).with_suffix('.html')
    shutil.copyfile(template, args.output/'index.html')
    last = None
    while True:
        data = snapshot(args.run, args.process_record)
        if args.comparison_run:
            comparison=read(args.comparison_run/'status.json',{})
            data['comparison']=dict(phase=comparison.get('phase'),iteration=comparison.get('iteration'),
                native=comparison.get('native'),actor_frozen_unchanged=comparison.get('actor_frozen_unchanged'))
        temp = args.output/'status.tmp'
        temp.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False))
        temp.replace(args.output/'status.json')
        marker = (data['kind'], data['latest'])
        if marker != last:
            print(json.dumps(dict(label=data['label'], latest=data['latest']), ensure_ascii=False), flush=True)
            last = marker
        # This exporter serves this finite run only; it is not a reboot daemon.
        comparison_finished=not args.comparison_run or data['comparison']['phase'] in ('complete','failed','interrupted')
        if args.once or (not data['alive'] and comparison_finished):
            break
        time.sleep(3)


if __name__ == '__main__':
    main()
