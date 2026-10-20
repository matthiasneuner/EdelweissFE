# Plan: robustness round R1 (2026-10-08)

Source: REVIEW_MPI_2026-10-08.md (third review round; items confirmed by the orchestrator) and its "Decisions"
section. Base: `feat/mpi-explicit-dynamics-v2` after the gather-to-reader branch is merged. Branch
`fix/robustness-r1`. Rabbit, env next_v2611_mpi; references `mpi_explicit_benchmarks/refactor_reference_marmot_e992a3f4`
(harness default; BASELINE for c1_150: `…/c1_150_baseline.npy` in that set).

## Items

1. **Stale state after an agreed step failure (reproduced).** After `StepFailedOnAllRanks`, nothing synchronizes the
   model before `finalizeStep` of field outputs/output managers; on a cutback failure at n=3, 31 of 98 U entries of
   the final state differ from serial NED by up to 0.095 (repro deck:
   /tmp/claude-1000/-home-matthias-constitutive-modeling-next-v2611/9b4e6ca5-3207-4233-8964-0733bf6b0f74/scratchpad/exp/div/{ser,dec}/c/test.inp).
   Fix: on an agreed failure, bring the last *accepted/published* state (not the attempted increment) to the readers
   before finalizeStep (as serial NED leaves it). Test: the repro deck at n=2/3 bitwise vs serial NED (U, P, V,
   last field-output frame); fails on the base.
2. **Abort before any `finally` block communicates.** A rank-local exception outside the agreed blocks (e.g. in
   beginStep / step-start actions) currently reaches `finally` blocks (StepBase.solve → finalizeStep gathers) before
   the driver aborts. Under MPI, any non-agreed exception must abort before collectives in cleanup paths. Test: a
   rank-local failure in a step-start action on rank 1 of 3 → clean abort, no hang (timeout-guarded test), with a
   negative control.
3. **CI compares serial NED.** The CI workflow (.github/workflows/run_tests_mpi.yml) must also produce the serial
   reference with `solver=NED` (strip NEDMPI-only options, as the docs recipe does) and compare NEDMPI at 2/3 ranks
   against it at tolerance 0, including restart checkpoints (h5 dataset-by-dataset) for the restart decks. Verify the
   job's commands locally.
4. **Wider layout fingerprint**: include element-set and surface order (the load ordinals depend on them) and the
   MPC operator (slave/master DOFs and coefficients); loud error on mismatch. Consider PYTHONHASHSEED guidance in
   the docs.
5. **Ensight**: `visualizedElementsOf` reads `mesh.elementSets[elementSet.name]` (ensight.py ~765) — use the set's
   own `elementNumbersOfWholeSet()` (works for ElementSetOfSurfaceFace too). Test.
6. **External work as one exact scalar per increment**: instead of storing every prescribed-DOF product per
   increment and gathering all of them at output, accumulate per rank per increment an exact (integer/compensated
   exact) sum such that the combined result equals `math.fsum` of all products bit for bit (e.g. exact
   integer-scaled partial sums combined exactly, then one correctly rounded conversion), skip zero products.
   Bitwise unchanged; measure on a deck with many Dirichlet DOFs and infrequent output.
7. **Plain textbook element loop in serial NED** (user decision 2026-10-08, revising 2026-10-07): NED's
   `assembleInternalForces` shows `for each element: compute Pe; P[el] += Pe` in element order; the chunked plan
   (ElementPlan, bincount, threads) lives only in NEDParallel/NEDMPI; a unit test proves plain loop == plan bit for
   bit (several meshes, threads 1/4). Serial NED's external work becomes a plain readable sum if that stays bitwise
   with the NEDMPI exact sum (it must — check; otherwise keep fsum with a one-line reason). Performance: serial NED
   (1 thread) must not get slower; NEDParallel unchanged.

## Gates
Bitwise everything (verify_refactor.sh quick after each code commit, full once at the end on an idle machine —
check ps, leave no background runs); testfiles/mpi -n 2/3/4 `--tolerance 0` vs fresh serial NED; mode check;
checkpoints; full pytest; suites; sphinx; lint; timings serial/serial32/8x4 vs base. Rules: PLAN_MPI_REFACTOR.md
"Rules for both"; small Conventional Commits with the two footer lines; stage by name; no push; no helper agents.
Time cap ~5 h; stop and report rather than force anything costing bit-identity or needing a design decision.
