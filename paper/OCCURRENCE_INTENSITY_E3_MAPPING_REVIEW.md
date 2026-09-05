# E3 controller-visible mapping review record

Status: `IMPLEMENTED_FAIL_CLOSED_PENDING_INDEPENDENT_VALUES`

After the unchanged E1/E2 tests and E3 engineering sentinel pass in the
independent environment, the controller reviewer supplies one JSON object to
`paper/validate_occurrence_intensity_e3_mapping_record.py`. The record contains
the literal accepted upstream base and decision; independently assigned mode,
commit and runner digest; SHA-256 bindings to the frozen contract, config and
both compact files; the test command and sentinel; literal validation `PASS`;
and an empty deviations list.

The validator recomputes every artifact digest and the E3 compact-result
semantics. It accepts no waivers or extra keys and returns only
`READY_FOR_CONTROLLER_ADMISSION_REVIEW` with `launch_authorized=false`.
Therefore this record closes the publication-to-controller handoff but cannot
create a trusted mode or authorize GPU execution.
## Atomic independent preflight

`scripts/run_occurrence_intensity_e3_admission_preflight.sh` is the single
publication-side entrypoint for the independent handoff.  It fails before
creating sentinel evidence if `torch` is unavailable, runs the unchanged E1,
E2 and E3 correctness tests, executes the frozen eight-case sentinel, and
passes both compact files through the semantic validator.  With no second
argument it stops at the explicit
`E3_MAPPING_RECORD_PENDING_CONTROLLER_IDENTITIES` boundary.  After the
controller supplies a mapping record containing real reviewed identities, the
same command accepts that record as its second argument and invokes the
fail-closed mapping validator.  The entrypoint does not create a mode or
authorize a launch.
