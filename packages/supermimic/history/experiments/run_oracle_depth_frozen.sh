#!/usr/bin/env bash
# Independent terrain-conditioned SuperMimic trial on verified SomaForge low-step tasks.
set -euo pipefail
# Parse the whole pipeline before launching training. This prevents a later
# on-disk edit from changing the tail of an already running shell script.
main() {
REPO="/home/zhouxia/workspace/supermimic_vibed"
RUN="${1:-heightcontact_boxes_main}"
ITERATIONS="${2:-6000}"
TASK="${3:-$REPO/data/lowstep_boxes/task.json}"
CONTACT_MODEL="${4:-binary2}"
RESUME="${5:-}"
RESET_SAMPLER="${6:-return_bins}"
PROBE_MODE="${7:-all_long}"
POLICY_FORCE_OBSERVATION="${8:-measured}"
PERCEPTION="${9:-heightmap}"
if [[ "$CONTACT_MODEL" != binary2 && "$CONTACT_MODEL" != force6 && "$CONTACT_MODEL" != original && "$CONTACT_MODEL" != height_only && "$CONTACT_MODEL" != force_only ]]; then
  echo "Contact model must be binary2, force6, original, height_only or force_only" >&2
  exit 1
fi
if [[ ! "$RUN" =~ ^[A-Za-z0-9_.-]+$ || "$RUN" == . || "$RUN" == .. ]]; then
  echo "Invalid run name" >&2
  exit 1
fi
if [[ ! "$ITERATIONS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Iterations must be a positive integer" >&2
  exit 1
fi
if [[ -e "$REPO/logs/$RUN/args.json" ]]; then
  echo "Existing run: choose a new name or explicitly resume with run_train_native.sh." >&2
  exit 1
fi
if [[ -n "$RESUME" && ! -f "$RESUME" ]]; then
  echo "Resume checkpoint does not exist: $RESUME" >&2
  exit 1
fi
cd "$REPO"
mkdir -p "logs/$RUN/source"
echo "$$" > "logs/$RUN/pipeline.pid"
trap 'code=$?; printf "%s\n" "$code" > "$REPO/logs/$RUN/pipeline.exitcode"' EXIT
cp "$TASK" "logs/$RUN/task_snapshot.json"
cp "$REPO/run_lowstep_native.sh" "$REPO/run_train_native.sh" "logs/$RUN/source/"
cp supermimic/lowstep.py supermimic/lowstep_kinematics.py supermimic/lowstep_observations.py supermimic/lowstep_colliders.py supermimic/physics_failure.py supermimic/networks.py supermimic/mdp.py "logs/$RUN/source/"
cp supermimic/retry_sampler.py supermimic/env.py "logs/$RUN/source/"
git diff -- supermimic/train.py supermimic/evaluate.py supermimic/motion.py supermimic/buffer.py tools/render.py \
  > "logs/$RUN/source/core_changes.patch"
TRAIN_ARGS=(--task_manifest "$TASK" --contact_model "$CONTACT_MODEL" --num_envs 4096 --iterations "$ITERATIONS" --ckpt_every 500 --capture_physics_failure)
TRAIN_ARGS+=(--reset_sampler "$RESET_SAMPLER")
TRAIN_ARGS+=(--probe_mode "$PROBE_MODE")
TRAIN_ARGS+=(--policy_force_observation "$POLICY_FORCE_OBSERVATION")
TRAIN_ARGS+=(--perception "$PERCEPTION")
if [[ "$PERCEPTION" == panorama_* ]]; then
  TRAIN_ARGS+=(--log_every 10)
  cp supermimic/panorama.py supermimic/rolling_panorama.py "logs/$RUN/source/"
fi
if [[ -n "$RESUME" ]]; then
  TRAIN_ARGS+=(--resume "$RESUME")
fi
"$REPO/run_train_native.sh" "$RUN" "${TRAIN_ARGS[@]}"
"$REPO/tools/run_native.sh" supermimic/evaluate.py --run_dir "logs/$RUN" \
  --iteration_interval 1000 \
  2>&1 | tee "logs/$RUN/evaluation.log"
printf -v CHECKPOINT 'logs/%s/ckpt_%06d.pt' "$RUN" "$ITERATIONS"
"$REPO/tools/run_native.sh" supermimic/evaluate.py --run_dir "logs/$RUN" --ckpt "$CHECKPOINT" \
  --record "results/${RUN}_final.npz" 2>&1 | tee "logs/$RUN/record.log"
mapfile -t PREVIEW_CLIPS < <(python3 - "$TASK" <<'PY'
import json, sys
spec = json.load(open(sys.argv[1]))
seen = set()
if spec.get('group_counts'):
    for task in spec['tasks']:
        if task['group'] not in seen:
            print(task['name'])
            seen.add(task['group'])
PY
)
RENDER_ARGS=(--max_clips 4)
if (( ${#PREVIEW_CLIPS[@]} )); then
  RENDER_ARGS=(--clips "${PREVIEW_CLIPS[@]}")
fi
"$REPO/tools/run_native.sh" tools/render.py "results/${RUN}_final.npz" "results/${RUN}_final.mp4" \
  --task_manifest "$TASK" "${RENDER_ARGS[@]}" --seconds 10 --title "SuperMimic $RUN | it $ITERATIONS" \
  2>&1 | tee "logs/$RUN/render.log"
"$REPO/tools/run_native.sh" tools/plot.py "logs/$RUN" --out "results/${RUN}_success_curve.png"
}
main "$@"
