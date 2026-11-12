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

The retired build and training commands are preserved in the [original record](https://github.com/xia112358/contactwm/blob/d54b71f7708d4d82eb5f477b223464872c239f60/docs/history/infiller-main-validation-20261023.md).
They are not current entrypoints. A corpus from a different scene fingerprint must be rebuilt; changing its
metadata alone is not a migration. Sample-limited `--smoke` builds are diagnostic
artifacts, not evidence of complete dataset coverage.
