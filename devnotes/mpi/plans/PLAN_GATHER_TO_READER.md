# Plan: gather only to the reader (2026-10-08)

Source: REVIEW_MPI_2026-10-07b.md (architect, priority 3: "gather only to the reader"), mpi_explicit_benchmarks/
ordering_cost/RESULTS.md. Base: `feat/mpi-explicit-dynamics-v2` at `ca6c1d91` (pushed). Worktree
`EdelweissFE-gather`, branch `perf/gather-to-reader`. Rabbit, env next_v2611_mpi (Marmot e992a3f4); references:
`mpi_explicit_benchmarks/refactor_reference_marmot_e992a3f4` (harness default; BASELINE for c1_150:
`…/refactor_reference_marmot_e992a3f4/c1_150_baseline.npy`).

## Today (what moves where, O(model))

- Output increments: `synchronizeModel` allgathers U, V, P (3 × nDof) to EVERY rank and publishes node fields
  everywhere; element field outputs allgather whole-set results (pickled, per element) to EVERY rank.
- Before a periodic contact search: the same full U/V/P allgather, although a search reads only the candidate
  surface positions on the constraint owner.
- Checkpoints already gather element states to rank 0 only (keep).
- Readers that really need whole-model data on every rank: the replicated AMR (markers on whole-set field outputs,
  node-value interpolation on topology change, the recovery-error marker's node fields), anything deciding
  collectively (conditional stop is decided on rank 0 and agreed; energy table uses allreduced sums).

## Goal

Every gather goes only to the ranks that read the data, with results unchanged bit for bit:
1. Plain output increments (no topology check, no marker reading): vectors and element field outputs gathered to
   rank 0 only; other ranks keep their own part. Field outputs that a marker reads (or that decide anything on all
   ranks) still reach every rank — determine this from the model (markers declare which field outputs they read),
   not by a blanket rule; markers could alternatively allgather only per-element scalars/marked numbers.
2. Topology-check increments: whatever the replicated refinement needs (node fields everywhere) — make it explicit
   which data the topology update requires and gather exactly that.
3. Contact search: the constraint owner receives only the positions (and whatever else the search reads) of its
   candidate surface nodes — surface-sized, point to point — instead of the full-vector allgather.
4. Non-root ranks must not silently hold stale whole-model data that someone later reads: make "this vector/node
   field is complete only on rank 0 / only at the integrated DOFs" explicit and guarded (like partial element
   sets), so a reader on another rank fails loudly.
5. Docs: domaindecomposition.rst — what each rank holds after each synchronization point.

## Gates

Bitwise everything (verify_refactor.sh quick after each code commit, full once at the end on an idle machine —
check ps; leave no background runs); testfiles/mpi -n 2/3/4 `--tolerance 0` vs fresh serial (strip NEDMPI-only
options as the docs recipe does); mode check; checkpoints == serial; full pytest; suites; sphinx; lint.
Performance and memory (alternate base EdelweissFE-mpi2 / head, idle machine): c1_150 at 8x4 and 32x1 —
per-increment time, output-increment time, contact-search time, bytes per output increment and per search,
rank-0 vs other ranks' peak RSS. Rules: PLAN_MPI_REFACTOR.md "Rules for both"; small Conventional Commits with
the two footer lines; stage by name; no push; no helper agents. Time cap ~5 h; stop and report rather than force
anything costing bit-identity or needing a design decision.
