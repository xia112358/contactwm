# Repository cleanup scope

Current scope and verification commands are maintained in
[repository-maintenance.md](repository-maintenance.md).

The earlier WBT-only cleanup has completed. The workspace now includes Motion
Edit, Generator, Contact Solver, shared core, WBT and the ACP branch.
Infiller and the standalone Retargeting package have been retired by explicit
user request. Current runtime data and independent third-party reference material
remain protected; module retirement does not authorize clearing other inputs.

All cleanup preserves existing source changes, baseline snapshots and referenced
runtime artifacts. File removals require an enumerated approval and use the
system recycle bin. Never use a blanket `git clean`, discard untracked research,
or clear `tmp/` based solely on its name.
