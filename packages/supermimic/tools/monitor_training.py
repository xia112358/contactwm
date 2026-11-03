"""Read-only, dependency-free training dashboard; bind to loopback for SSH forwarding."""
from supermimic.paths import runtime_path
import argparse
import csv
import io
import json
import math
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def rows(path):
    try:
        # Ignore a final line that the trainer has not finished writing.
        data = path.read_text()
        data = data[:data.rfind('\n') + 1]
    except OSError:
        return []
    result = []
    for row in csv.DictReader(io.StringIO(data)):
        converted = {}
        for key, value in row.items():
            if key is None:
                continue
            try:
                number = float(value)
                converted[key] = number if math.isfinite(number) else None
            except (TypeError, ValueError):
                converted[key] = value
        result.append(converted)
    return result


def process_alive(pid, expected):
    try:
        command = Path(f'/proc/{int(pid)}/cmdline').read_bytes().decode().replace('\0', ' ')
        return expected in command
    except (ValueError, TypeError, OSError):
        return False


def tail(path):
    try:
        with path.open('rb') as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 12000))
            return '\n'.join(f.read().decode(errors='replace').splitlines()[-20:])
    except OSError:
        return ''


def snapshot(run):
    meta = read_json(run / 'experiment.json')
    train = rows(run / 'train_log.csv')
    latest = train[-1] if train else {}
    alive = process_alive(meta.get('processes', {}).get('training'), 'supermimic/train.py')
    eval_alive = process_alive(meta.get('processes', {}).get('evaluation'), 'supermimic/evaluate.py')
    failed = (run / 'invalid_transition.pt').exists()
    stopped = (run / 'STOPPED.json').exists()
    iteration = latest.get('iteration', 0) or 0
    total = meta.get('iterations', 6000)
    state = ('尚未启动' if not run.exists() else '异常停机' if failed else '已停止' if stopped else '训练中' if alive
             else '训练更新已完成' if iteration >= total else '训练进程已退出')
    evaluations = []
    sources = [(run, '当前训练')]
    parent = meta.get('parent_run')
    if parent and Path(parent).name == parent:
        sources.insert(0, (runtime_path('logs') / parent, '恢复前基线'))
    for folder, label in sources:
        for row in rows(folder / 'eval.csv'):
            if label == '恢复前基线' and row.get('iteration', 0) > meta.get('resume_iteration', 0):
                continue
            number = int(row['iteration'])
            detail = read_json(folder / f'ckpt_{number:06d}_eval.json')
            count = len(detail.get('per_clip', {}))
            row.update(source=label, total=count or None, groups=detail.get('groups', {}),
                       passed=count - int(row['n_failed']) if count else None)
            evaluations.append(row)
    path = run / 'train_log.csv'
    sampler = read_json(run / 'sampler_stats.json')
    sampler.pop('per_clip', None)
    return dict(run=run.name, state=state, alive=alive, eval_alive=eval_alive,
                failure=failed, total=total, gpu=meta.get('physical_gpu'),
                group_counts=meta.get('group_counts', {}),
                sampler_stats=sampler,
                perception_stats=read_json(run / 'perception_stats.json'),
                updated=time.time(), log_age=time.time() - path.stat().st_mtime if path.exists() else None,
                train=train, latest=latest, evaluations=evaluations,
                pipeline=tail(run / 'pipeline.log'), evaluation_log=tail(run / 'auto_eval.log'))


def comparison(allowed):
    members = []
    common = None
    for name in allowed:
        run = runtime_path('logs') / name
        meta = read_json(run / 'experiment.json')
        if meta.get('comparison_family') != 'mixed145_v1':
            continue
        evaluations = {int(row['iteration']): row for row in rows(run / 'eval.csv')}
        iterations = {i for i in evaluations if i % 1000 == 0 and
                      (run / f'ckpt_{i:06d}_eval.json').exists()}
        common = iterations if common is None else common & iterations
        members.append({'run': name, 'label': meta.get('comparison_label', name)})
    iteration = max(common) if common else None
    for member in members:
        detail = read_json(runtime_path('logs') / member['run'] / f'ckpt_{iteration:06d}_eval.json') if iteration else {}
        member['groups'] = detail.get('groups', {})
    return {'iteration': iteration, 'members': members}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='climb_low_force6_resume1000')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--compare', action='append', default=[], help='Additional allowed run names')
    args = parser.parse_args()
    if any(Path(name).name != name for name in [args.run, *args.compare]):
        parser.error('--run must be a directory name inside logs/')
    run = runtime_path('logs') / args.run
    if not run.is_dir():
        parser.error(f'Run directory does not exist: {run}')
    allowed = [args.run, *args.compare]

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlparse(self.path)
            path = url.path
            if path in ('/', '/monitor', '/monitor.html'):
                content = (ROOT / 'tools' / 'training_monitor.html').read_bytes()
                content_type = 'text/html; charset=utf-8'
            elif path == '/api/status':
                selected = parse_qs(url.query).get('run', [args.run])[0]
                if selected not in allowed:
                    self.send_error(404)
                    return
                content = json.dumps(snapshot(runtime_path('logs') / selected), ensure_ascii=False, allow_nan=False).encode()
                content_type = 'application/json; charset=utf-8'
            elif path == '/api/runs':
                content = json.dumps(allowed).encode()
                content_type = 'application/json; charset=utf-8'
            elif path == '/api/comparison':
                content = json.dumps(comparison(allowed), ensure_ascii=False, allow_nan=False).encode()
                content_type = 'application/json; charset=utf-8'
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(content)

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'Monitoring {run.name}: http://127.0.0.1:{args.port}/', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
