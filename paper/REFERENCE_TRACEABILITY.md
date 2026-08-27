# Reference traceability audit

Audit date: 2026-08-27

This audit maps every bibliography item to the narrow statement it supports.
Bibliographic sources establish method definitions and scoring conventions; a
reference does not support project-specific empirical values, which remain
traceable only through the claim ledger and compact publication artifacts.

| Reference | Immutable identity | Supported manuscript statement | Scope limit |
|---:|---|---|---|
| 1 | `10.48550/arXiv.2006.11239` | diffusion probabilistic models generate samples with learned reverse/noise-removal dynamics | Does not establish this project's conditional architecture, resolution or sea-ice skill. |
| 2 | `10.48550/arXiv.2210.02747` | Flow Matching trains continuous normalizing flows by regressing vector fields along conditional probability paths | Does not establish this project's conditioning branches or empirical performance. |
| 3 | `10.1002/qj.2270` | fair ensemble scores correct finite-ensemble unfairness when members are interpreted as samples | Supports the fair-score convention, not any reported score value or the stronger phrase that it is universally required. |
| 4 | `10.1198/016214506000001437` | CRPS is a proper scoring rule for a predictive distribution | Does not imply that improvement in one proper score establishes calibration or permits compensation across gate families. |
| 5 | `10.1214/13-STS443` | ECC-Q reconstructs multivariate ensembles by coupling calibrated univariate quantiles to the raw ensemble rank template | Does not establish that the project's hurdle or ZOIB marginals are calibrated, spatially safe or eligible. |
| 6 | `10.1002/2015GL067232` | IIEE is the area where forecast and truth disagree on whether concentration exceeds 15% | Does not support this project's IIEE values, tolerances or causal interpretation. |
| 7 | `10.1080/01621459.2017.1307116` | split conformal inference supplies a finite-sample marginal-coverage construction with a statistical-efficiency/computational tradeoff | Does not establish validity under this project's temporally purged folds or support its coverage and width thresholds. |
| 8 | `10.1016/j.physd.2006.11.008` | LETKF is a local ensemble-transform construction for finite-ensemble data assimilation | Does not establish this project's background-ensemble identity, tuning fairness, numerical validity or comparative skill. |

## Independent metadata and statement audit

- Titles, author lists, venues, years and DOI/arXiv identities match the primary
  publisher or conference records reviewed on 2026-08-27; volume/page fields,
  where present in the manuscript, also match.
- Every item is cited in the manuscript, and every citation has one explicit
  support role above; no bibliography item is used as empirical evidence.
- The wording for reference 3 is deliberately narrower than “required”: the
  source establishes fairness for sampled ensembles, while the manuscript's
  choice of fair CRPS as its selection objective is a study design decision.
- References 1 and 2 support only the high-level generative-model motivation.
  Stochastic interpolants were mentioned in the same sentence but were not
  independently sourced; that separate family has therefore been removed from
  the citation statement.
- References 7 and 8 define the two still-missing baseline method families.
  They do not turn either frozen project-specific contract into an empirical
  result and cannot change its `MISSING` evidence state.

Audit decision: PASS_WITH_SCOPE_CORRECTION
