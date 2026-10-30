# SomaForge Core

Shared assets, kinematics, data contracts and native Newton contact evidence.
This package owns the contracts used by WBT, Motion Edit, Generator and Contact
Solver; it does not start their applications or import their training loops.

| Area | Modules |
| --- | --- |
| Canonical robot and asset identity | `robot_assets`, `asset_registry` |
| Motion and training inputs | `motion_schema`, `motion_contracts`, `training_manifest` |
| FK and pose/velocity consistency | `g1_kinematics`, `kinematics` |
| Actual contact activation and allocation | `newton_contacts`, `newton_contact_tensors` |
| Finite primary surfaces and contact reduction | `contact_face_selection`, `newton_support`, `native_surface_index` |
| Load, contact intent and material motion | `support_semantics`, `support_evidence`, `loaded_material_motion` |
| Runtime locations and subprocess environment | `runtime_paths` |

The authoritative G1 asset is
`src/holosoma/holosoma/data/robots/g1/g1_29dof_spherehand.urdf`.
Use `somaforge_core.robot_assets` and preserve `robot_asset_json`; missing or
mismatched identities are rejected. A Git checkout includes the URDF and its
meshes. Generated USD, motion data, checkpoints and logs require local setup or
separate restoration; they are not supplied by installing this Python package.

Actual contact requires the solver's `CONSTRAINT` type, `dist < includemargin`
and allocated `efc` rows. Effective task contacts additionally pass the shared
primary-face selector: ground or horizontal upward top surfaces, excluding
self/cross-robot contacts. Force, geometric proximity and predicted intent do
not define contact. Preserve raw evidence, part/shape mappings and sampling
times; last-solve contacts and post-integration poses are not strictly same-state
samples. See the [contact contract](../../docs/contact_label_contract.md) and
[support evidence](../../docs/support-semantics.md).

Canonical motion files use WXYZ root/body quaternions; simulator-facing WBT
states use XYZW. Conversion belongs at the declared interface, not in callers
that guess component order. See [runtime and rotation conventions](../../docs/environments.md).

From the repository root, `source scripts/source_somaforge.sh` exposes the
workspace packages. Use the [GA environment setup](../../docs/environments.md)
before simulator-dependent calls. Ordinary CPU contracts live in `tests/` beside
this package; a package import or CPU test does not validate GPU execution.

Related entrypoints: [Motion Edit](../motion_edit/README.md),
[Generator](../generator/README.md), [Contact Solver](../contact_solver/README.md),
[modular WBT expert/distillation](../../docs/wbt-modular-experiments.md), and the
[documentation index](../../docs/README.md).
