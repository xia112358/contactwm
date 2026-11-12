# Retired Contact WM implementation

Archived on 2026-11-08. The maintained pipeline is SuperMimic tracking → native
collection → DAgger initialization → `supermimic.contact_train` (DMO or imagination).
See [current workflow](../../current-workflow.md) and
[current contact training](../../../packages/supermimic/README_CONTACT.md).

`source/` preserves the former `src/holosoma/holosoma/contact_wm/` implementation
(36 tracked files). `tests/` preserves its nine tests as `.py.txt` reference files,
so ordinary test discovery does not collect retired imports.
[design.md](design.md) preserves the former `docs/contact-wm-bptt.md`.

This snapshot includes the residual WM, inherited force-WM adapter, custom stream
trainer, activation head, panorama actor and world-fixed spatial contact field.
They are historical experiments and are outside installed source packages.
The current trainer uses the original SuperMimic WorldModel and force predictor;
it does not use this snapshot's activation head or custom replay/stream loop.

No checkpoints, datasets, logs or current SuperMimic source changes were moved.
To reproduce a historical experiment, restore this snapshot in a separate checkout
with its original runtime and artifact contracts; archived commands are not current recipes.
