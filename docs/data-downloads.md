# Download training inputs and demonstrations

Training data and fixed reference models are optional GitHub Release assets,
separate from the source checkout:
[data-20261022-v1](https://github.com/xia112358/somaforge/releases/tag/data-20261022-v1).
The tracked [`configs/data_releases.json`](../configs/data_releases.json) catalog
uses schema `somaforge_data_release_catalog_v1`.

The downloader needs only Python's standard library. It does not install the
simulator, select a new checkpoint, start training/evaluation, or connect training
stages into a workflow. Choose inputs explicitly for each command using the
[modular WBT guide](wbt-modular-experiments.md) or the relevant Generator/Motion
Edit recipe.

## Select and download

Run these commands from the checkout root:

```bash
python3 scripts/bootstrap_runtime.py
python3 scripts/download_data.py list

# Download just the baseline motion inputs.
python3 scripts/download_data.py fetch wbt_baseline29
python3 scripts/download_data.py verify wbt_baseline29

# Download PHP motions and one explicitly named existing expert.
python3 scripts/download_data.py fetch php-wbt-296-data php-wbt-teacher-locomotion
python3 scripts/download_data.py verify php-wbt-teacher-locomotion

# Optional: restore every published component and its data dependencies.
python3 scripts/download_data.py fetch all
python3 scripts/download_data.py verify all
```

`list` reports component IDs, purpose, byte counts and original status. `fetch`
and `verify` require one or more IDs or `all`; nothing is downloaded when no
component was selected. Declared dependencies are included: a teacher depends
on its motion inputs, for example. This resolves data dependencies only.

The catalog's `bytes` and `file_count` describe each component's own immutable
files after cross-component deduplication; dependency components are additional.
Transfer sizes and archive checksums are recorded under `archives`, while each
`*.files.json` index records individual installed file sizes and SHA256 hashes.
Uncompressed component bytes do not equal network transfer or total disk usage.
The catalog's `repository_files` are already distributed by Git. Their hashes
and historical aliases are checked without downloading or rewriting them.

For `data-20261022-v1`, the 25 components contain **10,564 unique payload files**,
totalling **16,155,241,949 uncompressed bytes (16.16 GB)**. Compressed archive
parts total **11,665,364,686 bytes (11.67 GB)**, excluding small file indexes and
the catalog. The 44 already tracked repository files are separate from these
payload counts. Grouped unique, uncompressed payload sizes are:

| Group | Bytes | GB (decimal) |
| --- | ---: | ---: |
| PHP 296-motion inputs and five experts | 1,118,665,774 | 1.12 |
| WBT baseline29 and full legacy eight-part recordings | 1,352,942,706 | 1.35 |
| Predictor inputs, reference checkpoint and materials | 987,213,144 | 0.99 |
| Three Infiller corpora and retained sources | 1,351,929,807 | 1.35 |
| Motion Edit height110 example | 6,583,245 | 0.007 |
| Eight extracted-demonstration variants | 4,575,273,443 | 4.58 |
| Source-prefix-standard 207 | 1,145,318,287 | 1.15 |
| Dynamics-corrected 207 | 5,615,483,104 | 5.62 |
| Demonstration catalog and migration ledger | 1,832,439 | 0.002 |

Shared files belong to one component; selecting a group may also require the
dependencies declared in the catalog. These figures describe the published
payload, not peak installation disk usage or optional retained download caches.

## Components and retained meaning

Infiller code and training entrypoints were retired on 2026-10-23. Infiller
components in this immutable release are historical data archives. Downloading
them does not restore a supported Infiller module.

| Component | Inputs and purpose |
| --- | --- |
| `php-wbt-296-data` | 296 canonical G1 motions, all corresponding original PHP motion/terrain inputs, generated terrains, manifests and upstream attribution. No observed contact/force labels. |
| `php-wbt-teacher-locomotion`, `php-wbt-teacher-low-step`, `php-wbt-teacher-high-step`, `php-wbt-teacher-low-climb-76`, `php-wbt-teacher-high-climb-76` | Five explicitly retained dense expert checkpoints and native configurations. Existing models, not a new automatic model selection. |
| `wbt_baseline29` | Canonical 29-motion baseline and original source evidence. |
| `wbt_legacy8part_full_demonstrations` | Older eight-part force references and complete original rollout recordings. Threshold/hysteresis masks retain their historical meaning. |
| `predictor_current_training_data` | Registered Predictor motion/contact/event/certificate/cache/query-scene closure. Original `training_ready=false` and recorded runtime remain unchanged. |
| `predictor_current_reference` | Registered analysis checkpoint, frozen comparison records and material geometry/model dependencies. |
| `infiller_demo_latest`, `infiller_demo_recursive` | Prepared full demonstration corpora and available retained source inputs for each Infiller experiment. |
| `infiller_historical_shared_demo` | Retained complete 207-motion / 4761-segment historical corpus and source closure. |
| `motion_edit_height110` | Editing source, plan, layers, terrain and retained example result. |
| `demonstrations_*` | Eight registered extracted-demonstration variants and referenced content-addressed objects. Exact IDs appear below. |
| `source_prefix_standard207` | Formal 207 source-prefix-standard kinematic references, plans and completion evidence. Kinematic training inputs do not establish successful physical execution. |
| `dynamics_corrected207` | Formal 207 dynamics-corrected demonstrations, native labels, events and provenance. Existing acceptance records are preserved; `training_ready=false` is unchanged. |
| `demonstration_catalog` | Source archive catalog and retained migration ledger; depends on its indexed variants. An inventory, not an admission certificate. |

The eight extracted-demonstration IDs are:

- `demonstrations_temporal207_training_v2_ready`
- `demonstrations_temporal207_newton_main_v1`
- `demonstrations_temporal207_newton_main_eventclean_v1`
- `demonstrations_temporal207_hold_compact_v4`
- `demonstrations_temporal207_newton_main_hold_compact_eventclean_v1`
- `demonstrations_full1000_fixed_timeline_newton_20260922_v3`
- `demonstrations_climb00_joint_consensus_retention_v5`
- `demonstrations_support_preserving_regeneration_20260926`

Names containing `ready`, `accepted` or `newton_main` describe the saved
experiment; they do not certify compatibility with a later runtime or contact
contract. Download verification establishes file integrity, not training
admission, convergence or physical task success. Preserve `status`,
`provenance_notes` and original acceptance fields. Incompatible native evidence
must still fail at its training loader; publication never regenerates labels or
relaxes those checks.

Six archived variants retain one unavailable historical surface `mesh_path`:
the prepared training/query terrain inputs are present, but the old LTE
regeneration process is not claimed complete. The dynamics archive also retains
nine unavailable original contact-label sources; its formal rematerialized
labels and events are included. These provenance gaps are recorded in component
notes. A different available file is never substituted for a missing original
whose hash cannot be established.

The release excludes training logs, redundant historical checkpoint sequences,
temporary verification downloads and disposable failed work directories. It
retains selected formal inputs and original extraction/provenance evidence;
it is not a workstation backup.

## Another checkout, external disk or offline assets

`--root` selects the destination checkout and defaults to the repository
containing the script. `--catalog` defaults to that script's
`configs/data_releases.json`. Specify both when invoking another script copy:

```bash
python3 scripts/download_data.py fetch wbt_baseline29 \
  --root /path/to/new-checkout \
  --catalog /path/to/new-checkout/configs/data_releases.json

# Supply both *.files.json indexes and every required *.tar.gz part.
python3 scripts/download_data.py fetch wbt_baseline29 \
  --root /path/to/new-checkout --local-archives /path/to/release-assets
python3 scripts/download_data.py verify wbt_baseline29 \
  --root /path/to/new-checkout --local-archives /path/to/release-assets
```

For an external disk, first initialize the destination checkout with
`python3 scripts/bootstrap_runtime.py --data-root /absolute/path/to/data`.
Downloads follow its `runtime/current` link. Matching installed files are
verified and kept; conflicting files are reported instead of overwritten or
deleted. Retrying preserves matching installed files. Interrupted downloads and
conflicting evidence are retained for inspection.

Use one installer at a time per physical data root. Concurrent writers to the
same `runtime/current` tree or its local alias file are not supported, even
when invoked from different checkout paths.

Assets are cached under `runtime/current/cache/data-releases/`. `verify` checks
installed payloads without installing missing data; it may download a missing
small file index. Use `--local-archives` with the required indexes and parts
for offline operation.

## Immutable files and local path aliases

Old manifests, certificates, NPZ provenance and checkpoints can contain absolute
source-workstation paths. Their original bytes and hashes are retained. The
downloader creates local installation metadata at:

```text
runtime/current/manifests/published_data_paths.json
```

Its schema is `somaforge_published_data_paths_v1`; `prefixes` and `files` contain
explicit absolute `source`/`target` pairs. Readers apply exact file aliases first,
then the most specific directory prefix. Declared relative references between
an old physical data root and its checkout use these aliases even when the new
checkout and data disk have different layouts. Undeclared paths are not guessed.

This ignored host-local mapping is not part of the source evidence or a replacement
manifest. Do not edit original JSON/NPZ/model files or restamp their hashes to
move them. Reader-side relocation preserves identity, certificate/cache checks
and checkpoint sampler metadata without changing model or optimizer tensors.
With no installed mapping, ordinary path handling is unchanged. Fetch the
selected components for the destination checkout rather than blindly copying
another host's local mapping.

Training still requires the canonical G1 asset and pinned environment. PHP's
clean canonical motions require separate actual-execution recording before use
as new six-part sparse/force references. Historical eight-part force masks,
archived demonstrations and new six-part references are not interchangeable.
See [installation](install.md), [data policy](data-policy.md),
[contact labels](contact_label_contract.md) and [support evidence](support-semantics.md).

## Attribution and publication scope

PHP inputs retain their upstream Apache-2.0 `LICENSE`, Amazon `NOTICE`, release
asset checksums and README under
`runtime/current/motions/php_wbt_v1_20261016/raw/upstream/`, with the individual
bundle README files alongside the raw inputs. Preserve these source notices
when copying or redistributing that component.

The extracted OmniRetarget robot-terrain demonstrations in this release are
the retained in-house motion-capture inputs. Retargeted LAFAN1 data restricted
from redistribution is not included. Preserve each source's own provenance and
terms; the repository's code license is not a blanket license for third-party
motion data, models or assets.

Publication verification reads the built archives as streams and checks their
declared members and hashes. It does not retain a second full verification
download/extracted dataset. This is separate from a user's requested installation
and its normal download cache.

Maintainers can repeat this check against existing local release assets without
extracting or copying the payload:

```bash
python3 scripts/build_data_release.py --verify-only \
  --output /path/to/release-assets --catalog configs/data_releases.json \
  --tag data-20261022-v1
```
