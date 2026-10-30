"""Summarize the completed, single-seed low-climb comparison without rerunning it."""
from supermimic.paths import runtime_path, runtime_root, workspace_root, module_root
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUNS = [('Heightmap + force6', 'climb_low_force6_resume1000', '#087f8c'),
        ('Original SuperMimic', 'climb_low_original_main', '#ce6c37')]


def main():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    summary = {'scope': 'Same 40 low-climb tasks, 6000 updates, seed 0',
               'limitations': [
                   'Policy, critic and world-model inputs all differ; not an isolated force-head ablation.',
                   'One training seed per method; two final evaluations are repeated rollouts, not independent training seeds.',
                   'Force6 resumed at iteration 1000 after a simulator failure, with fresh replay/history.',
                   'Training time is cumulative time in training logs; excludes startup, failed continuation and final postprocessing.',
               ], 'runs': {}}
    table = []
    for label, name, color in RUNS:
        run = runtime_path('logs') / name
        train = list(csv.DictReader((run / 'train_log.csv').open()))
        evaluations = {int(r['iteration']): r for r in csv.DictReader((run / 'eval.csv').open())}
        if name == 'climb_low_force6_resume1000':
            for r in csv.DictReader((runtime_path('logs/climb_low_force6_main/eval.csv')).open()):
                if int(r['iteration']) <= 1000:
                    evaluations[int(r['iteration'])] = r
        steps = [i for i in sorted(evaluations) if i % 1000 == 0]
        returns = [float(evaluations[i]['norm_return']) for i in steps]
        passes = [(40 - int(evaluations[i]['n_failed'])) / 40 * 100 for i in steps]
        minutes = [float(evaluations[i]['train_wall_h']) * 60 for i in steps]
        final = json.loads((run / 'ckpt_006000_eval.json').read_text())
        repeat = json.loads((runtime_path('results') / (name + '_final_eval.json')).read_text())
        info = {'final_evaluation': final, 'recording_evaluation': repeat,
                'training_log_minutes': float(train[-1]['wall_h']) * 60,
                'sim_steps': int(train[-1]['sim_steps']),
                'last_1000_iterations_per_second': 1000 / (
                    (float(train[-1]['wall_h']) - float(next(r for r in train if r['iteration'] == '5000')['wall_h'])) * 3600),
                'video': 'results/' + name + '_final.mp4'}
        summary['runs'][name] = info
        for i, ret, passed, minute in zip(steps, returns, passes, minutes):
            table.append(dict(run=name, iteration=i, norm_return=ret, pass_rate_percent=passed, training_log_minutes=minute))
        for ax, xs, ys in [(axes[0], steps, returns), (axes[1], steps, passes), (axes[2], minutes, returns)]:
            ax.plot(xs, ys, marker='o', lw=2, color=color, label=label)
        # Independent recording rollouts are marked explicitly, not averaged into the primary curve.
        axes[0].scatter([6000], [repeat['norm_return']], marker='x', s=75, color=color, zorder=4)
        axes[1].scatter([6000], [(40-repeat['n_failed'])/40*100], marker='x', s=75, color=color, zorder=4)
        axes[2].scatter([minutes[-1]], [repeat['norm_return']], marker='x', s=75, color=color, zorder=4)
    for ax, title, xlabel, ylabel, upper in zip(axes,
            ['Full-clip return', 'Full-clip completion', 'Return vs. logged training time'],
            ['Training updates', 'Training updates', 'Cumulative training time (min)'],
            ['Mean normalized return', 'Passed clips (%)', 'Mean normalized return'], [1, 100, 1]):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel, ylim=(0, upper))
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(loc='upper left', frameon=False)
    fig.suptitle('Low climb: same 40 tasks, 6000 updates, one training seed per method', fontsize=14)
    fig.text(.5, .02, 'Circles: primary evaluation. Crosses: independent final recording rollout. '
             'Force6 resumed at update 1000; both policy and world-model inputs differ.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .95))
    out = runtime_path('results/climb_comparison_6000')
    fig.savefig(out.with_suffix('.png'), dpi=180)
    out.with_suffix('.json').write_text(json.dumps(summary, indent=2) + '\n')
    with out.with_suffix('.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(json.dumps({name: {k:v for k,v in info.items() if k not in ['final_evaluation','recording_evaluation']}
                      for name,info in summary['runs'].items()},indent=2))


if __name__ == '__main__':
    main()
