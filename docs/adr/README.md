# Architecture decision records

Significant decisions are recorded here, one file each, in the format of
Michael Nygard's ADRs: context, decision, consequences. A decision that is
revised gets a new record that supersedes the old one; old records are not
rewritten.

| # | Decision | Status |
|---|---|---|
| [0001](0001-record-architecture-decisions.md) | Record architecture decisions | Accepted |
| [0002](0002-harness-over-a-frozen-model.md) | An external harness over a frozen model | Accepted |
| [0003](0003-provider-neutral-structured-output.md) | Provider-neutral interface with native structured output | Accepted |
| [0004](0004-equal-budget-accounting.md) | Equal-budget accounting as a first-class concern | Accepted |
| [0005](0005-deterministic-rounds.md) | Deterministic rounds for parallel streams | Accepted |
| [0006](0006-brain-constants-as-defaults.md) | Brain constants, converted per thought, as defaults | Accepted |
| [0007](0007-benchmark-in-repository.md) | The benchmark lives in this repository for now | Accepted |
| [0008](0008-hashing-embedder-default.md) | A dependency-free hashing embedder as the default | Accepted |

To add one: copy the structure of an existing record, take the next number, and
link it in this table in the same pull request as the change it describes.
