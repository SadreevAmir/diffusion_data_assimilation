# Block-conformal area comparator: exact controller handoff

## Scientific identity

The immutable scientific contract is
`paper/NEXT_CONFORMAL_BASELINE_CONTRACT.md`.  The implementation package used
for independent review consists of:

- `paper/conformal_area_reference.py`;
- `paper/conformal_area_runner_prototype.py`;
- `paper/test_conformal_area_runner_prototype.py`;
- `paper/validate_conformal_area_admission.py`;
- `paper/test_validate_conformal_area_admission.py`.

The prototype and its tests are parity evidence, not controller admission and
not a scientific result.  Independent review must bind the reviewed server
runner to the SHA-256 of the frozen contract without changing folds, purge,
quantile, thresholds, source experiment or compact outputs.

## Sole experiment parameter and server boundary

The admitted server-CPU mode may accept only:

```json
{"source_experiment":"joint_full_condition_validation_2022"}
```

Retrieval is `summary_only`.  All member-area and truth computation remains on
the server.  The result directory contains exactly
`conformal_summary.json` and `conformal_per_case.csv`; no raw ensemble, truth
field or member field may be returned.

## Exact admission record

The controller-visible JSON object has exactly these keys:

```json
{
  "reviewed_mode": "LITERAL_CONTROLLER_VISIBLE_MODE",
  "publication_commit": "LOWERCASE_40_HEX",
  "runner_sha256": "LOWERCASE_64_HEX",
  "contract_sha256": "LOWERCASE_64_HEX",
  "synthetic_result_sha256": "LOWERCASE_64_HEX",
  "test_command": "NONEMPTY_REVIEWED_COMMAND",
  "test_sentinel": "NONEMPTY_PASS_SENTINEL",
  "decision_bearing_validation": "PASS",
  "deviations": []
}
```

Capitalized values above describe required independently supplied values; they
are deliberately invalid placeholders.  A local author must not replace them
to manufacture review evidence.  `contract_sha256` must equal the digest
computed directly from the frozen contract by
`conformal_area_reference.frozen_contract_sha256()`.

## Atomic admission command

After independent review and a decision-bearing synthetic server run, validate
the record and compact directory together:

```sh
python3 paper/validate_conformal_area_admission.py \
  /path/to/admission.json /path/to/compact-directory
```

Success emits one sorted JSON object containing only `reviewed_mode`,
`admission_record_sha256` and `compact_directory_sha256`.  Missing or extra
record fields, placeholders, malformed identities, contract drift, a non-PASS
decision, any deviation, extra compact files, an inconsistent decision or a
mutation during validation fails closed.

Only the literal `reviewed_mode` returned by this command may be copied into an
experiment proposal.  Local prototype parity, a narrative review, or a mode
name guessed from the method does not authorize a proposal.

## Focused verification

```sh
python3 -m unittest -v paper/test_conformal_area_runner_prototype.py
python3 -m unittest -v paper/test_validate_conformal_area_admission.py
PYTHONPYCACHEPREFIX=/tmp/conformal_pycache python3 -m py_compile \
  paper/conformal_area_reference.py \
  paper/conformal_area_runner_prototype.py \
  paper/validate_conformal_area_admission.py
git diff --check
```

The scientific stop/go rule remains unchanged: `CONFORMAL_USEFUL` requires
held-out coverage of at least `0.85` and mean-width ratio no greater than
`1.50`; otherwise the frozen outcome is `CONFORMAL_NEGATIVE`.  Either outcome
closes only the conformal minimum-tier row after its compact identity is
propagated consistently to the manuscript, claim ledger, readiness audit and
reproducibility handoff.  It cannot establish member calibration or overall
no-compensation eligibility.
