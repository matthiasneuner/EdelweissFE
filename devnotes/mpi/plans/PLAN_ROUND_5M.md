# Plan: review round 5m on the distributed-memory MPI work (2026-10-07)

Source: REVIEW_MPI_2026-10-07b.md (two independent Fable reviews of d75b3723; every item below confirmed in the
code by the orchestrator). Base: `feat/mpi-explicit-dynamics-v2` at `cf0594d6` (= d75b3723 merged with upstream
next_v26.11 1870025c). Worktree `EdelweissFE-5m`, branch `refactor/review-round-5m`. Rabbit, env next_v2611_mpi.

A parallel agent converts the new upstream `generators/cylindergenerator.py` to describe the mesh (worktree
`EdelweissFE-cyl`). Do NOT touch that file or its tests; do not change the signatures of `Mesh.addElement`,
`Mesh.setElementSet`, `Mesh.addSurface`, `FEModel.createElementsOfMesh` that generators use (if item 4's
`provider=None` fix needs a mesh API change, make it backward compatible for generator calls and say so).

## Items (in this order)

1. **Hang risks, made structural.**
   - No collective inside `allRanksFailTogether`: `NEDMPI.writeIncrementOutput` runs field-output finalization,
     whose `resultsOfWholeSet` does an allgather, inside the agreed context. Restructure as *evaluate locally →
     agree → gather/write* (two-phase field outputs if needed), so a rank failing before the gather cannot leave
     the others in a mismatched collective. Add a guard that makes a collective inside the context fail loudly
     (e.g. a flag on the communicator wrapper checked by the collectives), and a test injecting a failure on one
     rank before the gather (all ranks end with the same error, no hang).
   - A rank-local `StepFailed` (driver catches it and finishes normally, `inputfiledrivensimulation.py` ~295) must
     be agreed across ranks or abort the communicator when nProcesses > 1 — never leave other ranks blocked. Test.
2. **Lumped operators across migration**: after `_moveElements`, the mass/damping are reassembled from the
   current elements while serial keeps its operators. Either carry them like U/V/P (preferred, by construction),
   or assert bitwise equality with the previous operators at each migration (loud). State which and why.
3. **Serial stale-set trap**: `Mesh.removeElement` leaves `mesh.elementSets` stale ("whoever removes elements
   updates the sets"); a serial run then raises an incomplete-set error speaking of processes, and
   `checkElementsAgreeWithMesh` does not cover sets. Fix by mechanism: removing an element from the mesh removes
   it from the mesh's sets (and surfaces), or the agreement check covers sets — and the incomplete-set message
   must not talk about processes when there is one process. Test.
4. **MPI knowledge out of the serial core**:
   - `FEModel` must not import `isRootProcess` (femodel.py:39, used at ~720 for the material-property export):
     the writer decides who writes, outside the model (e.g. the section exporter / driver / an output step given
     the gathered data).
   - `FEModel.advanceToTime(time)` back to its old signature; NEDMPI accepts its own elements' state itself (e.g.
     `model.acceptElementStates(partition.elements)` as a public, documented model method if needed).
   - Checkpoint gathering (`gatherStatesForCheckpoint`/`forgetGatheredStates`/`statesOfElements`) out of
     `ElementDistribution` into the MPI layer (`Subdomain`/NEDMPI); the model's restart writer reads states through
     one model-level method with a trivial serial implementation.
   - Child placement during refinement as one `FEModel` method (e.g. `createChildElementOfMesh(child, parent)`
     returning the element or None) so `hadaptivity.py` never names a process.
   - Single-consumer hooks on shared base classes move into the MPI layer where possible:
     `OutputManagerBase.writesCheckpointAtNextIncrement`, `TimeStepperBase.incrementsLeftEstimate`,
     `NonlinearSolverBase.writeIncrementOutput` (keep any that cannot move, and say why), and
     `ModelPartition.ownedDofMask` (unused by NED → NEDMPI's own partition data).
   - `provider=None`: one meaning only (auxiliary elements get an explicit marker; `None` = default provider, or
     the reverse) — `MeshElement.isAuxiliary` keyed on that marker.
   - `Subdomain` (≈1300 lines) mixes partitioning, interface, loads, constraint ownership, state sync, failure
     agreement and load balancing: split it into a few cohesive modules/classes if it can be done cleanly within
     the time cap (optional; report).
5. **Smaller**: the docs' "verify bit-identity yourself" recipe must work for rebalance decks (the sed recipe
   leaves NEDMPI-only options such as `load-balance-costs` that NED rejects — strip them, or make NED ignore
   nothing and the recipe use a provided script); document the `np.bincount` left-fold assumption with the numpy
   version it was verified on, next to its test.

## Gates

Bitwise everything (`mpi_explicit_benchmarks/tools/verify_refactor.sh <wt> <label> quick|full`, REF=
`…/refactor_reference_fsum`; full once at the end on an idle machine — check `ps` —, leave no background runs);
testfiles/mpi -n 2/3 `--tolerance 0` vs fresh serial `--create`; mode check; full pytest; suites; sphinx (no new
warnings); lint; no slowdown (c1_150 serial32/8x4). Rules: PLAN_MPI_REFACTOR.md "Rules for both" (teaching
vocabulary, numpydoc, no hasattr/getattr/setattr, no duplication, elements parallelization-agnostic, small
Conventional Commits with the two footer lines, stage by name, no push, no helper agents/worktrees of your own).
Time cap ~6 h; stop and report rather than force anything costing bit-identity or performance, or needing a
design decision (state the question precisely).
