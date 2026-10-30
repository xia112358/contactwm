# Historical Infiller environment validation

The Infiller implementation and training entrypoints were retired on 2026-10-23.
These are preserved historical build records; commands below are not current entrypoints.

Infiller defaults use the same runtime input record and
`runtime/current/motions/infiller_main_20261022/corpus.pt`, registered as
`dataset.infiller.main_20261022`. The full build completed on October 23 with
3,105 demonstration segments, 2,660 accepted isolated predictions out of 3,105
attempts, and 1,506 accepted recursive predictions out of 1,651 attempts. Failed
endpoints remain in the audit and are excluded from predicted training records.
The 7,271 records retain the original height split: 4,111 train, 1,501 validation
and 1,659 test. This is full corpus materialization; the completed short training
check used the separately capped diagnostic corpus.

Build the corpus once, then explicitly select training with a fresh output
directory (skip the build command when restoring the registered corpus):

```bash
python3 scripts/manage_environment.py run --profile main -- \
  python -m generator.train_frozen_predictor_infiller --mode build \
  --report tmp/infiller_build_new --device cuda:0
python3 scripts/manage_environment.py run --profile main -- \
  python -m generator.train_frozen_predictor_infiller --mode train \
  --output runtime/current/models/generator/infiller_new_run \
  --report tmp/infiller_train_new --device cuda:0
```

The default `--mode all` builds before training and refuses an existing corpus.
A corpus from a different scene fingerprint must be rebuilt; changing its
metadata alone is not a migration. Sample-limited `--smoke` builds are diagnostic
artifacts, not evidence of complete dataset coverage.
