# Plan: element migration — load balancing in distributed mode (2026-10-06)

Context: DESIGN_DISTRIBUTED_MEMORY.md (§3, §5, §7), DESIGN_P2_MESH_AS_DATA.md §5b (the migration recipe),
PLAN_P3_OWNED_ELEMENTS.md and handoff §5g. Base: `feat/mpi-explicit-dynamics-v2` at `1da2add5` (pushed to
mn, no PR). Worktree `EdelweissFE-migration`, branch `feat/element-migration`. Machine rabbit.

## Situation

In distributed mode (P3) each process creates only its own elements (+ loaded neighbours). Rebalancing is
switched off there ("Load balancing is off …") and element kernels are not timed, because a changed
partition would need elements to move between processes. In whole-model mode rebalancing works (every
process already holds every element).

## Goal

Rebalancing by measured element cost works in distributed mode, with results bit-identical to the serial
solver whatever the partition (they are, by construction of the interface sums in model element order).

## The recipe (P2 §5b)

An element is fully described by the mesh (number, type, provider, node labels — replicated), replicated
definitions (section, material incl. materialParameterFromField, element properties) and its **state
vector**. Moving element e from process A to B:
- B: `model.createElementOfMesh(e)`, assign its section and properties (`assignSectionToElement`, the
  element-property path), `setStateVars(state)` with the state received from A — the converged state at a
  synchronization point (rebalancing runs right after the output synchronization).
- A: drop the element object (and every reference to it).
Everything indexed by the elements a process holds must follow, as after a topology change: the
DofManager's per-element index arrays, the element plan / interface assembly, the halo of load-carrying
elements (recompute with the new partition; create/drop as needed), the lumped operators only if they are
assembled from local elements (they are: own elements, completed at the interface — rebuild them or prove
them unchanged), field outputs' element result views (the result collectors hold pointers into element
state buffers — they must be rebuilt for moved elements; a `refreshResultViews` existed before P1 and was
removed as unnecessary then; it is necessary again), sets resolved from the mesh (`createElementsOfMesh`
already updates sets/surfaces in place), and the element timing arrays.
Re-enable element timing and the rebalancing check in distributed mode.

## Gates

- **Bitwise**: `mpi_explicit_benchmarks/tools/verify_refactor.sh <wt> <label> quick|full` (full at most
  twice). Add a deterministic way to *force* a repartition in tests (e.g. a test-only option or a
  deliberately skewed first partition, so rebalancing really moves elements at a known increment), and a
  testfiles/mpi deck that rebalances in distributed mode with restart output — bitwise vs serial at -n 2/3
  (also `--tolerance 0` vs fresh serial `--create`), checkpoint written after a migration == serial
  checkpoint (h5cmp), and the per-deck mode check extended to prove that elements actually moved (e.g.
  created-element sets per process differ before/after; journal line with counts).
- Memory: after a migration the sender holds no object of a moved element (count elements per process;
  peak RSS on the 192k block with forced rebalancing stays ≈ distributed P3 numbers, 1.77 GB at 8).
- Performance: the 192k block at 8 processes with rebalancing on vs P3 (off) — report per-increment time
  and imbalance before/after; c1_150 harness timings unchanged (whole-model mode).
- Suites, full pytest, sphinx, lint unchanged vs base; CI workflow still valid (add the new deck to
  EXPECTED modes if the mode check lists them).
- Rules: PLAN_MPI_REFACTOR.md "Rules for both" (teaching vocabulary, docstrings, no hasattr/getattr/
  setattr, no duplication, elements parallelization-agnostic, commits with the two footer lines, stage by
  name, no push). Docs: domaindecomposition.rst (load balancing in distributed mode; what moves, what is
  rebuilt). Time cap ~4 h; stop and report rather than force anything that costs bit-identity or needs a
  design change (state the question precisely).
