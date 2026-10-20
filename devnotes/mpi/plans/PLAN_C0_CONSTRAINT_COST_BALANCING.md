# Plan C0: constraint cost in the load balancer (2026-10-08)

Source: DESIGN_CONTACT_BY_SLAVE.md (phase C0, user decision: C0 first, then decide on splitting contact).
Base: `feat/mpi-explicit-dynamics-v2` at `d4d44f98` (pushed). Branch `perf/constraint-cost-balancing`.
Rabbit, env next_v2611_mpi; references = harness default (`refactor_reference_marmot_e992a3f4`).

## Problem
Constraints are evaluated whole by one owner (round-robin by name). The balancer weighs only element kernel
times, so it cannot relieve the rank that owns a big contact: on c1_150 the owner of `contactBottom` (rigid-body
contact, 10,656 facets) spends ~7 ms per ~90 ms increment evaluating it while the other ranks wait (7.3 ms at 8x4,
7.4 ms + 7.2 ms at 32x1, DESIGN_CONTACT_BY_SLAVE.md §1).

## Work
1. **Measure constraint cost**: time each owned constraint's evaluation (forces + connectivity search amortized),
   like element costs are timed.
2. **Balance with it**: the partition gives the owner of an expensive constraint correspondingly fewer elements —
   e.g. METIS target part weights (tpwgts) reduced by the constraint cost share, or an equivalent mechanism in
   `partitioning`; and/or assign constraints to ranks by cost (largest first, to the least loaded) instead of
   round-robin by name. Choose the simplest mechanism that measurably removes the wait; keep the decision identical
   on every rank (measured costs gathered/broadcast as today).
3. **Gain guard / imbalance metric** include constraint time (busy time = elements + owned constraints).
4. Docs: domaindecomposition.rst (load balancing).

Partitioning changes speed only, never results: everything stays bitwise (migration path already verified).

## Also: reproduce a suspected defect
DESIGN_CONTACT_BY_SLAVE.md mentions, by reading only: an expression field output of a constraint's
`getNormalPressures()` on rank 0 reads an unsynchronized constraint copy (zeros unless rank 0 owns it). Reproduce
with a small contact deck under NEDMPI at -n 2/3 vs serial NED; if real, fix (the constraint state of the owner must
reach the reader — follow the gather-to-reader rules) with a test that fails on the base; if not real, say why.

## Gates
Bitwise everything (verify_refactor.sh quick after each code commit, full once at the end on an idle machine —
check ps; no background runs); testfiles/mpi -n 2/3/4 `--tolerance 0` vs serial NED; mode check; full pytest; suites;
sphinx; lint. Performance (alternate base EdelweissFE-mpi2 / head, idle machine, load balancing ON with the default
tolerance and also a forced repartition): c1_150 at 8x4 and 32x1 — per-increment time, per-rank busy/wait times, the
contact owner's element count before/after. Rules: PLAN_MPI_REFACTOR.md "Rules for both"; small Conventional
Commits with the two footer lines; stage by name; no push; no helper agents. Time cap ~4 h.
