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

- Browser playback, motion editing and trajectory generation are retired.
- Keep next-contact Predictor independent and maintained.
- Preserve shared FK, contact evidence, motion storage and retained data utilities.
- Historical design documents and artifacts retain their provenance.

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

Run the remaining package tests with the configured environment. Do not start
training as part of cleanup checks.
