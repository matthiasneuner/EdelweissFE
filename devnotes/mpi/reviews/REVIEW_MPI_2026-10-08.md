# Opus reviews of feat/mpi-explicit-dynamics-v2 at ca6c1d91 (2026-10-08, third round)

## Architect — keep and evolve (redesign constraint ownership + vector locality)
- Could not break bit-identity on successful runs: all 12 edelweiss-only decks at n=7, all 23 marmot decks at n=5,
  body forces on out-of-order/duplicate sets at n=4.
- BUG (reproduced by orchestrator): after an agreed step failure (StepFailedOnAllRanks) nothing synchronizes the
  model before finalizeStep/output managers; last frame + final state stale (31 of 98 U entries differ from serial
  NED by up to 0.095 at n=3, cutback failure). Fix: gather last published node-field values before finalizeStep.
- Risk: rank-local exceptions outside the inner try (e.g. beginStep) run collectives in `finally`
  (stepbase finalizeStep) before the driver aborts — abort before any finally communicates.
- Fingerprint misses set/surface order (load tags depend on it) and MPC coefficients; add them; PYTHONHASHSEED.
- Large pickled gathers (element states to rank 0, whole-set field-output allgather) hit 2 GB message limits.
- External work: every Dirichlet DOF product stored per increment and gathered to every rank at output
  (150k support DOFs, output-frequency 1000 -> ~1.2 GB + 3.3 s fsum per output); use one exact big-integer
  accumulator per increment per rank, skip zero products.
- 2–3 global Allreduces per increment (one per agreement block) — merge into one per increment; overlap comm.
- Priorities: failure paths; local vectors (P6) first; contact by slave on host process with halo; compact
  replicated data, per-rank checkpoints; wider fingerprint + merged agreement.

## Teaching professor — acceptable with changes
- Better: NED's four named steps; "mesh is data" (a gain); adaptivity by element numbers.
- Worse: serial NED lost the textbook element loop (ElementPlan/bincount); partition.dofs/IncrementPlan/
  fsum external work in serial code; NED 2101 -> 2517 lines.
- MPI concepts in shared classes: FEModel (elementDistribution, elementStatesFromElsewhere, …), ElementSet raising
  on incomplete sets, replicatedElementsReason on 4 base classes, field-output two-phase state, single-consumer
  hooks (writesCheckpointAtNextIncrement, incrementsLeftEstimate), serial driver imports domaindecomposition.
- CONFIRMED: CI never compares against serial NED (testfiles/mpi decks say solver=NEDMPI; the "fresh serial"
  reference is NEDMPI on 1 process). CONFIRMED: ensight.visualizedElementsOf reads mesh.elementSets[name] (fails
  for ElementSetOfSurfaceFace).
- Other: NEDParallel docstring claims bit-identity without the psi/energy qualifier; jargon (namesDofMoreThanOnce,
  ownershipVersion, AUXILIARY, describedBy, whole set); misleading pre-existing comment at inputfilehelpers.py:276;
  isRetry reset fix should be its own commit.
- Required: plain element loop in NED (plan in NEDParallel/NEDMPI, test plain == plan) — NOTE: conflicts with the
  user's 2026-10-07 decision "one loop, made readable"; CI job NED vs NEDMPI incl. checkpoints; distribution out of
  the serial core (DistributedFEModel / registry; replicatedElementsReason as a table); nodes in the mesh / single
  createElementsOfMesh / ensight; glossary + names.

## Decisions (user, 2026-10-08)
- After gather-to-reader: robustness round (stale state after agreed failure; abort before finally-block
  collectives; CI job serial NED vs NEDMPI incl. checkpoints; wider fingerprint; Ensight fix; external work as one
  exact scalar per increment), then C0, then P6.
- REVISED (supersedes 2026-10-07 "one loop, made readable"): serial NED gets the plain textbook element loop;
  the plan-based loop lives in NEDParallel/NEDMPI; a unit test proves both give the same bits.
- Distribution machinery out of the serial core (DistributedFEModel/registry, replicatedElementsReason as a
  table): after P6, with a design note first.
