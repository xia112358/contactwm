# original12: task_max76_with_walk.json

`original12` contains twelve original PHP reference clips: step and 58/76 cm climb
with their mirrors, plus three eight-second walking windows and their mirrors.
The complete task component is distributed as
[original12-20261031-v1](https://github.com/xia112358/contactwm/releases/tag/original12-20261031-v1),
not as an assumed local training directory.

After [installing ContactWM](quickstart.md), run from the checkout root:

```bash
python3 scripts/download_data.py fetch original12 --catalog configs/original12_release.json
python3 scripts/download_data.py verify original12 --catalog configs/original12_release.json
```

For a private repository, use an account with repository access and an authenticated
GitHub CLI (`gh auth login`). The downloader uses `gh` if GitHub rejects an anonymous
release request. It never stores access tokens in the catalog or download URLs.
If you already have the release assets, use `--local-archives /path/to/assets`.
No account is needed for that offline route.

The manifest is now located at:

```text
runtime/current/workflow/original12/task_max76_with_walk.json
```

Use this path as `--task_manifest` in the existing workflow commands. For example,
inspect the tracking launch without starting training:

```bash
python3 scripts/run_workflow.py tracking --dry-run -- \
  --task_manifest runtime/current/workflow/original12/task_max76_with_walk.json
```

Optional file-integrity and relocation check, using the installed interpreter:

```bash
runtime/environments/venv/workflow/bin/python -m supermimic.task_manifest \
  runtime/current/workflow/original12/task_max76_with_walk.json
```

The 54-file component includes the unchanged task JSON, all twelve canonical NPZ
references, both referenced/selected CSV directories, paired USD/OBJ terrain,
visualization XML, source catalogue and upstream license/notice. It contains no
tracking policy, world-model weights, contact recording or training checkpoint.
The catalogue is retained as provenance; unrelated clips in that inventory are
not included in this selected component.

The installer verifies archive and file hashes and writes host-local path aliases
under `runtime/current/manifests/published_data_paths.json`. Task readers apply
those aliases in memory. Original files, manifest SHA, robot identity, sampling
weights, mirror relationships, frame ranges, collision surfaces and contact
semantics are preserved. No directory with the author's username is needed.
Do not rewrite the JSON to make its old absolute paths look local: changing its
bytes also changes the identity used by recordings/checkpoints.

A task manifest supplies reference inputs; it does not choose compatible physics
or certify physical execution. Keep the intended experiment's explicit collision
profile/contact response and matching checkpoint contracts. Source contact hints
are not actual Newton contacts. Fresh native contact collection remains required.
The auxiliary renderer uses the destination checkout's canonical mesh directory;
it does not load a second robot asset from the historical machine.

Maintainers can rebuild a release without simulation using
`scripts/build_original12_release.py --help`. Supply the original manifest and
upstream LICENSE/NOTICE explicitly. The builder checks every declared input hash,
preserves original bytes and stream-verifies the resulting archive.
