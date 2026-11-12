# ContactWM documentation

Start with the [repository overview](../README.md) and
[environment setup](environments.md). Run documented commands from the repository
root unless a recipe explicitly states otherwise. A code checkout contains source,
configuration and canonical robot meshes; datasets, checkpoints and generated USD
have separate preparation or restoration requirements.

| Task | Guide |
| --- | --- |
| Download current tracking/contact starting checkpoints | [Checkpoint downloads and purposes](checkpoints.md) |
| Run tracking → native collection → DAgger → contact training | [Current workflow](current-workflow.md), [Contact training](../packages/supermimic/README_CONTACT.md) |
| Create independent Miniforge + pip profiles without replacing an existing runtime | [Python environment profiles](python-environments.md) |
| Install the pinned Isaac Lab / Newton runtime; check quaternion conventions | [Environments](environments.md) |
| Download selected training inputs, reference models or extracted demonstrations | [Data downloads](data-downloads.md) |
| Find package ownership, entrypoints and verification scope | [Repository maintenance](repository-maintenance.md) |
| Configure dense, sparse or force experts and distillation independently | [Modular WBT experiments](wbt-modular-experiments.md) |
| Review motions, Predictor outputs or diagnostics in the browser | [Read-only playback](../packages/motion_edit/README.md) |
| Train or evaluate Predictor | [Generator](../packages/generator/README.md) |
| Query obstacle-aware distances for online top-contact goals | [Navigation distance](contact-navigation.md) |
| Use contact objectives, collision gradients or projection | [Contact Solver](../packages/contact_solver/README.md) |
| Use shared assets, FK and evidence contracts | [SomaForge Core](../packages/somaforge_core/README.md) |
| Diagnose whether contact targets are visible | [Contact observability](../packages/contact_observability/README.md) |
| Locate code versus external motion/model data | [Repository data](repository-data.md), [runtime layout](../runtime/README.md) |
| Read tracked-data and retention rules | [Data policy](data-policy.md), [cleanup scope](cleanup-scope.md) |

## Data and physical evidence

- [Contact-label contract](contact_label_contract.md): native activation,
  constraint allocation, source identity and evidence timing.
- [Support evidence](support-semantics.md): distinguish contact, measured load,
  slip diagnostics and keep intent; use the shared primary-face rules.
- [Policy reference format](policy-ref-v1.md): canonical reference arrays and provenance.
- [Trajectory refinement](trajectory_refinement.md): retained policy refinement and
  execution validation for explicitly supplied references. Old editing/generation
  recipes in historical reports are not current entrypoints.
- [Offline contact consensus](offline_contact_consensus.md): recorded-contact processing.

## Baselines and historical records

The registered Predictor comparison baseline is selected by
[`baselines/current.json`](../baselines/current.json). Its frozen recipe and
weights describe that experiment; current scratch defaults are documented in
the [Generator guide](../packages/generator/README.md). Neither designation
implies completed end-to-end physical acceptance.

See [historical experiments](history/README.md) and
[`baselines/archives.json`](../baselines/archives.json) for older recipes.
Their commands, verification counts and local artifact paths retain historical
meaning and must not be treated as current setup instructions or training inputs.

The old trajectory editor/generator and Infiller are retired. Their design notes,
experiment reports and release records are archival references, including older
files still located outside `history/`. They do not authorize restoring removed
entrypoints or treating historical contact labels as current evidence. Predictor,
WBT, contact-control work, shared FK and read-only playback remain maintained.
