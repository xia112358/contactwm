# Repository Rules

## Branching

- `main` must stay runnable and reproducible.
- Do not develop directly on `main`.
- Start new work from latest `main`:

```bash
git switch main
git pull origin main
git switch -c feat/name
```

- Use short-lived branches:
  - `feat/...` for features
  - `fix/...` for bugs
  - `exp/...` for experiments
  - `refactor/...` for cleanup
  - `docs/...` for docs
- Before merging, sync with latest main:

```bash
git fetch origin
git rebase origin/main
```

- After merge, delete finished branches.

## Commits

- Keep commits focused.
- Use concise messages:
  - `feat: ...`
  - `fix: ...`
  - `exp: ...`
  - `refactor: ...`
  - `docs: ...`

## Project Workflow

- The maintained browser entrypoint is `motion-edit review --input PATH`.
- The UI is the single-port `motion_edit/web/server.py` + `web/` Three.js app.
- Playback is read-only. Do not reintroduce trajectory editing/generation,
  `generate-ref`, Infiller, IK workers, Viser, iframe wrappers, a second UI port,
  or file-polling bridges.
- Keep next-contact Predictor in `packages/generator` independent and maintained.
- Preserve shared FK, contact evidence, motion storage and retained data utilities.
  Shared numerical helpers belong to `somaforge_core`, not a generation package.
- Historical design documents and generated artifacts retain their provenance;
  they are not current implementation requirements or evidence of task success.

Current product flow:

```text
Canonical motion / diagnostic report / case collection
-> Validate identity and input structure
-> Read-only browser playback
```

## Data

- `data/` is local runtime data and must not be tracked.
- Before pushing, verify:

```bash
git ls-files data | wc -l
```

Expected:

```text
0
```

- Do not commit checkpoints, logs, videos, `.npz`, `.npy`, or other large runtime artifacts unless explicitly requested.

## Tests

- Before merging to `main`, run:

```bash
source scripts/source_somaforge.sh
PYTHONDONTWRITEBYTECODE=1 python3 scripts/manage_environment.py run --profile main -- python \
  -m pytest -p no:cacheprovider packages/motion_edit/tests -q
npm --prefix packages/motion_edit/web run build
```

- Run these commands from the repository root. Check browser playback against
  an explicitly selected canonical motion or diagnostic report. Browser checks
  must exercise only the retained read-only flow. Root
  `tests/e2e/test_training.py` starts training and is not part of cleanup checks.

- Isaac Sim / CUDA failures inside Codex sandbox are expected; validate those on the local machine.
