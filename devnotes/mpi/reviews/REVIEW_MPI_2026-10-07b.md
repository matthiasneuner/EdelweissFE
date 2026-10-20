# Fable reviews of feat/mpi-explicit-dynamics-v2 at d75b3723 (2026-10-07, second round)

## Architect — keep and evolve
- Sound: partition before creation, DOF layout by construction + fingerprint, tagged interface exchange, loads by
  owner in deck order, migration releasing everything first, fsum external work, CI 2/3 ranks --tolerance 0.
- Could NOT break bit-identity: 3 adversarial experiments at n=3 (two steps, output every increment, body force +
  follower pressure + node forces, 31 migrated; contact with forced migration of 53 elements incl. facets; live AMR
  6 refinements + 10 migrations) — U and checkpoints identical.
- Risks (confirmed by orchestrator): (1) collectives (field-output allgather) inside allRanksFailTogether in
  writeIncrementOutput -> hang if a rank fails before the gather; make it structural (evaluate locally -> agree ->
  gather). (2) driver catches StepFailed and finishes normally on one rank -> hang if a future rank-local
  StepFailed appears; agree or abort when nProcesses > 1. (3) lumped operators reassembled after migration
  (state-independence assumed, unasserted) — carry or assert. (5) updateRigidBodies on partial fields, relies
  on owner-only reads. (6) doc recipe sed fails on rebalance decks (NED rejects load-balance-costs).
- Bit-identity rests on numpy bincount's undocumented left fold (tested); pin / note numpy version, or make the
  serial sum reproducible by construction.
- Priorities: fail-together structural + StepFailed agree/abort; carry/assert lumped operators; gather only to
  the reader (results to rank 0, markers one scalar per element, checkpoints per rank); constraints partitioned
  by slave entity + constraint cost in balancer; P6 local DOFs/ghosts, then retire the replicated fallback.

## Teaching professor — acceptable with changes
- Gains: model construction reads as two textbook stages ("a real teaching gain, keep it even without MPI");
  NED's increment as four named steps with formulas; marking by element number cleaner.
- Costs: `partition.dofs` indirections in the velocity/displacement update; AMR modifier calls
  distribution.placeChildElement / isLocal (process placement inside refinement).
- MPI knowledge in serial core (confirmed): FEModel imports isRootProcess (femodel.py:39,720);
  advanceToTime(elements=, constraints=); ElementDistribution carries a checkpoint protocol; single-consumer hooks
  on base classes (OutputManagerBase.writesCheckpointAtNextIncrement, TimeStepperBase.incrementsLeftEstimate,
  NonlinearSolverBase.writeIncrementOutput); ModelPartition.ownedDofMask unused by NED; two ownership tables
  (DistributedElements.owners vs Subdomain._elementOwners); Subdomain 1294 lines mixing concerns.
- Serial trap (confirmed): Mesh.removeElement leaves elementSets stale -> serial run raises an error speaking of
  processes; checkElementsAgreeWithMesh does not cover sets. provider=None means both "auxiliary" and "default
  provider" (confirmed). model.elements not fail-safe (surfacesnap, tie iterate it).
- Required: (1) fix the stale-set trap; (2) remove MPI knowledge from FEModel (isRootProcess, advanceToTime args,
  checkpoint gathering into Subdomain/NEDMPI); (3) child placement in one FEModel method, hadaptivity names no
  process; model.elements fail-safe or documented; (4) drop single-consumer base hooks into the MPI layer;
  provider=None ambiguity; (5) explicit ordered reduction instead of bincount order dependence.

## Environment hazard (orchestrator, confirmed)
env next_v2611_mpi: `import edelweissfe` without PYTHONPATH loads the superseded EdelweissFE-mpi worktree.
