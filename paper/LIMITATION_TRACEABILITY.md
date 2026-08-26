# Limitation traceability

This map is normative for Section 7 of `PAPER_DRAFT.md`. Each distinct
limitation must retain one row and must point either to a claim-ledger row that
supports the limitation or to an explicit `Unknown` evidence-absence record.
The map does not turn an absence of evidence into a positive result.

| ID | Exact Section 7 anchor | Claim-ledger support | Support kind |
|---|---|---|---|
| L1 | `one legacy checkpoint, one sampling seed, ten members and 40 development dates` | `C11`, `C35` | Direct scope limitation |
| L2 | `not an independent temporal generalization estimate` | `C3`, `C15` | Direct design limitation |
| L3 | `not direct satellite SIC retrievals` | `C2` | Direct observation-system limitation |
| L4 | `does not establish casewise or fieldwise coverage` | `C13` | Direct diagnostic limitation |
| L5 | `Clipping complicates bounded mean comparisons` | `C12`, `C19` | Direct mechanism limitation |
| L6 | `residual upper-tail undercoverage remains` | `C13`, `C27` | Direct reliability limitation |
| L7 | `No independent comparison with 3D-Var is claimed` | `C9`, `C17` | Explicit evidence absence (`C9` is `Unknown`) |
| L8 | `Generalization across checkpoints, seeds, ensemble sizes, regions or observation systems remains unverified` | `C10` | Explicit evidence absence (`C10` is `Unknown`) |

The eight rows cover the complete limitation inventory in Section 7. A new,
removed or materially changed limitation requires an atomic update to the
manuscript, this map and its claim-ledger support.
