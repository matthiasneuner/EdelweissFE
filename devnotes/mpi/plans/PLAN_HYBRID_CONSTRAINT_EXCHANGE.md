# Plan: point-to-point constraint forces ("hybrid"), bit-identical (2026-10-08)

Source: the ordering-cost experiment, mpi_explicit_benchmarks/ordering_cost/RESULTS.md (+ prototype.patch, mode
`hybrid`). Base: `feat/mpi-explicit-dynamics-v2` (pushed to mn at 0cdc9524, or later after the Marmot update).
Branch `perf/constraint-forces-point-to-point`. Rabbit, env next_v2611_mpi. Starts only after the Marmot update
of 2026-10-08 is finished (the env must be stable; use the reference set it names as valid).

## Why

`ConstraintForceExchange` shares every constraint's forces with every rank (`Allgatherv`) and every rank adds all
of them in model order. On c1_150 at 32x1 that is 86.5 MB per increment (63 MB = one rank's big contact constraint
copied 31 times) and +43 % per increment (134.9 vs 93.5 ms, mostly waiting on the rank stuck in the
`Allgatherv`). The prototype's `hybrid` mode — ordered interface forces as today, constraint forces sent point to
point only to the ranks integrating the constraint's DOFs, added in model order — was bitwise to serial on all 16
reference decks (3/4 ranks) and on c1_150 at 8x4 and 32x1, at 94.5 ms.

## Work

1. Replace the `Allgatherv` sharing by a point-to-point exchange: the owner of each constraint sends its forces at
   the DOFs it shares with a rank to that rank (layout exchanged once per rebuild, like today); every rank adds the
   forces of all constraints touching its integrated DOFs **in model order** (so the per-DOF fold equals serial).
   Implement it as the one way (not a mode switch), reusing the tagged-exchange machinery
   (`_TaggedContributions` / subdomaininterface) rather than a parallel implementation.
2. The prototype's caveat: entries of P at DOFs a rank does not integrate now differ from the Allgatherv path.
   Nothing reads them — make that explicit (document which entries are defined on a rank; zero or poison them in
   a debug check/test so a future reader of non-integrated entries fails loudly instead of reading stale values).
3. Constraint ownership unchanged (round-robin by name); the guard (`withoutCommunication`) must cover the new
   calls; failure agreement unchanged.
4. Docs: domaindecomposition.rst (how constraint forces travel, cost), the overrides table.

## Gates

- Bitwise everything: verify_refactor.sh quick after each code commit, full once at the end on an idle machine
  (check ps; leave no background runs), with REF/BASELINE = the valid reference set; testfiles/mpi -n 2/3/4
  `--tolerance 0` vs fresh serial; mode check; the constraint-force ordering unit test (adversarial) adapted to
  the new exchange; full pytest; suites; sphinx; lint.
- Performance (alternate base/head on the idle machine): c1_150 at 8x4 and 32x1 — expect ≈134.9 → ≈95 ms at 32x1;
  report constraint-exchange time per rank and message volume (the prototype's EDELWEISSFE_MPI_REPORT_VOLUME
  idea may be kept as a debug option if cheap, otherwise dropped).
- Rules: PLAN_MPI_REFACTOR.md "Rules for both"; small Conventional Commits with the two footer lines; stage by
  name; no push; no helper agents. Time cap ~3 h.
