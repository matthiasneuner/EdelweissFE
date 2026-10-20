# Plan: P6.3 — rebuilds without whole-model work, then local vectors (2026-10-15)

Source: DESIGN_P6_LOCAL_DOFS.md (§2, §4, §5, §8, §10 P6.3, all "Decisions"), REVIEW_MPI_2026-10-15_architect.md F1
and F9 ("now, as part of P6.3"), REVIEW_MPI_2026-10-15_teaching.md (NED loses `[dofs]`). Vocabulary after T2:
`Subdomain` = the piece a process computes, `DomainDecomposition` = NEDMPI's machinery, "partition" = METIS only.
Split into two rounds, each with the full bitwise gate.

## P6.3a — rebuild cost (global-length vectors unchanged)
Base `feat/mpi-explicit-dynamics-v2` at 7cbfa8b4; worktree `EdelweissFE-p63a`, branch `perf/p6-3a-local-interface`.
1. **Interface from neighbour-local data (F1).** `SubdomainInterface` must not `Allgatherv` every process's
   subdomain DOFs nor allocate nDof-length arrays per rebuild where avoidable. Each process knows the replicated
   element partition (owners of all elements), the replicated mesh connectivity, the constraint owners and the
   `GlobalDofNumbering`: derive, locally, which processes integrate each of its subdomain DOFs (and so owners, =
   lowest rank, and neighbours), exchanging with neighbours only what cannot be derived (e.g. MPC closure
   results, constraint footprints after a search). Results bitwise unchanged; same owners and same tags.
2. **Fingerprint from data.** `layoutFingerprint` digests replicated arrays (mesh node/element arrays,
   GlobalDofNumbering.digest(), set/surface label arrays, MPC operator) instead of walking Python objects.
3. **MPC closure** without an nDof×nDof COO where avoidable (restrict to the DOFs MPCs touch).
4. **Constraint ownership by DOF majority (F9):** a constraint is evaluated by the process that owns most of its
   DOFs (ties: lowest rank), decided identically everywhere from replicated data; re-decided at repartition/
   topology change; load balancing (C0 shares) unchanged in principle. Measure interface size and force traffic.
5. Measure rebuild time per process (start, topology change, contact-search rebuild) and peak memory of a
   rebuild on c1_150 at 8x4 and 32x1, base vs head alternated; expect the rebuild time to fall with P.

## P6.3b — local nodes, DofManager and vectors
Base `feat/mpi-explicit-dynamics-v2` at ab188f8c (pushed); worktree `EdelweissFE-p63b`, branch `feat/p6-3b-local-vectors`.
Per DESIGN_P6_LOCAL_DOFS.md §2/§4/§5: NEDMPI holds Node objects, NodeFields, DofManager, DofVectors, lumped
operators and MPC transformation for its subdomain only, in global order (order-preserving local numbering;
DofManager unchanged); exchanges keyed by global DOF; gathers to rank 0 for output/checkpoint (format unchanged);
resume by rows; `Subdomain.dofs` becomes `slice(None)` in NEDMPI too, so NED's update steps lose `[dofs]`
(teaching review). Models with AMR or contact keep global length for now (one logged rule, as in P3), until
P6.4 (compact AMR mirror, local nodes under AMR) and P6.5 (contact: the evaluating process holds the constraint's
surface nodes, received copies for those it does not integrate). Gate adds: 192k block memory per process.

## Gates (each round)
Bitwise everything: verify_refactor.sh quick after each code commit, full once at the end on an idle machine
(check ps; no leftover processes, no self-matching pgrep waiters); testfiles/mpi -n 2/3/4 `--tolerance 0` vs
fresh serial NED; checkpoints; mode check; full pytest (3 test_tie failures possible at base); suites (base:
CDPPlane); sphinx; lint; c1_150 serial32 + 8x4 timing vs base (no regression). Rabbit, env next_v2611_mpi
(editable install points at EdelweissFE-mpi2: worktree first on PYTHONPATH, print edelweissfe.__file__; copy .so
files + built_extensions.log from EdelweissFE-mpi2). Rules: PLAN_MPI_REFACTOR.md "Rules for both"; small
Conventional Commits; stage by name; no push; no helper agents; temporary WIP commits, not git stash. Time cap
~5 h per round; stop and report rather than force a design decision.

## Decisions (user, 2026-10-15) after P6.3b
1. Ties/MPC models keep whole-model vectors for now (local MPC transformation moves to P6.3c).
2. NED keeps `[dofs]` until the whole-model rule is retired (P6.6); no second update path.
3. New round **P6.3c** before P6.4: local Node objects and NodeFields (held-node decision at model setup via
   `createNodesOfMesh(isHeldHere)`, node sets as the part held here, rank-0 gathers for perNode output/Ensight/
   checkpoints, resume by rows of local node fields) and the local MPC transformation (closure known before the
   local DofManager), so tie models leave the whole-model rule.

## P6.3c — local Node objects, NodeFields and MPC transformation
Base `feat/mpi-explicit-dynamics-v2` at 2e1b6aa1 (pushed); worktree `EdelweissFE-p63c`, branch `feat/p6-3c-local-nodes`.
1. Held-node decision at model setup: after the partition, `createNodesOfMesh(isHeldHere)` (P6.1) makes Node objects
   only for the nodes of the subdomain (elements of the subdomain + nodes of its constraints + MPC closure); serial
   and the whole-model rule hold every node. Node sets = the part held here (guard as for element sets).
2. NodeFields local (subdomain nodes, mesh order); publish/gather keyed by global DOF/node row; rank-0 gathers for
   per-node field outputs, Ensight and checkpoints (file formats unchanged); resume by rows into local node fields.
3. Migration: nodes acquired/released with elements (`putNodesInMeshOrder`), values carried by global DOF.
4. Local MPC transformation: closure known before the local DofManager (from replicated MPC definitions + the
   partition, as P6.3a's multiPointConstraintGroups), transformation built on the subdomain DOFs, fold order
   unchanged (bitwise); tie models leave the whole-model rule.
5. Readers of `model.nodes` listed in the P6 round-1 report: convert those reachable in subdomain mode to the mesh
   arrays or the held part; anything left must fail loudly (complete-or-raise), never read silently stale data.
6. Docs. Measure: block deck (mpi_explicit_benchmarks/p63b_block) peak RSS per process at 8 and 32, build time,
   per-increment time base vs head; a tie deck in subdomain mode; c1_150 unchanged (bitwise, timing).
Gates as above (each code commit quick; full once at the end idle; testfiles/mpi -n 2/3/4 tol 0 vs fresh serial NED;
checkpoints; mode check; pytest; suites; sphinx; lint). Cap ~6 h; stop at a clean gated commit if it doesn't fit.
