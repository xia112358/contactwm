# Core starting checkpoints

[Download the 2026-11-12 checkpoint release](https://github.com/xia112358/contactwm/releases/tag/core-starting-checkpoints-20261112-v1).
The six files total 742,752,351 bytes (743 MB). Original checkpoint bytes and saved
formats are retained. The [tracked manifest](core-starting-checkpoints-20261112.json)
records exact source paths, sizes and SHA256 values.

## Download and verify

Run from the repository root; choose an external storage directory:

```bash
gh release download core-starting-checkpoints-20261112-v1 \
  --repo xia112358/contactwm --dir /external/contactwm-checkpoints
cd /external/contactwm-checkpoints
sha256sum -c SHA256SUMS
```

Individual files can be selected with `gh release download ... --pattern FILENAME`.
The release also includes `manifest.json`, `SHA256SUMS` and a README.

## Files and purposes

| Download | Size (MB) | Purpose |
| --- | ---: | --- |
| [tracking-5k.pt](https://github.com/xia112358/contactwm/releases/download/core-starting-checkpoints-20261112-v1/tracking-5k.pt) | 84.88 | Current original12 iteration-5000 tracking teacher and WM source; use as `--checkpoint` for collection, DAgger initialization and contact training. |
| [tracking-5k-full-state.pt](https://github.com/xia112358/contactwm/releases/download/core-starting-checkpoints-20261112-v1/tracking-5k-full-state.pt) | 258.26 | Original `tracking/last.pt` from the same run; full saved tracking continuation state. |
| [contact-init.pt](https://github.com/xia112358/contactwm/releases/download/core-starting-checkpoints-20261112-v1/contact-init.pt) | 123.58 | Contact initialization retained by `bottleneck-handoff-retention-20261108`; also the start of navigation-chain100. Use as the contact `--resume` for that setup. |
| [contact-prior.pt](https://github.com/xia112358/contactwm/releases/download/core-starting-checkpoints-20261112-v1/contact-prior.pt) | 41.44 | Fixed actor prior paired with that initialization; use as `--prior_checkpoint`. |
| [contact-start-160.pt](https://github.com/xia112358/contactwm/releases/download/core-starting-checkpoints-20261112-v1/contact-start-160.pt) | 123.59 | Iteration-160 checkpoint from `bottleneck-chain100-prefix-fresh-20261110`; starting state of the retained single-scene navigation comparisons. |
| [navigation-reference-wm.pt](https://github.com/xia112358/contactwm/releases/download/core-starting-checkpoints-20261112-v1/navigation-reference-wm.pt) | 111.00 | Frozen locally refitted WM used by those navigation diagnostics. Distinct from the original tracking teacher WM. |

For contact training, `--checkpoint`, `--resume` and `--prior_checkpoint` have
separate roles as described in [the workflow](current-workflow.md). Use matching
task manifests, recordings, slices and physics settings. Historical absolute paths
can remain in the saved metadata; publication preserves the original files.

These are model assets, not a complete data/environment bundle. Source at upload
was based on `d54b71f` with local changes, published with this source sync. The older [original12 training release](original12.md) contains the separate
iteration-1000 initialization and iteration-1500 continuation, not this 5k run.
The historical `accepted-baseline.json` and its original DAgger file remain
missing; the published contact initialization has its own recorded provenance.

The navigation models are experiment-specific. The retained single-scene
comparison reached zero outside-region collisions but stable-contact success was
0/53 at update 60; see [navigation scope and results](contact-navigation.md).
