# Independent reviews of feat/mpi-explicit-dynamics-v2 (2026-10-06)

Two read-only Fable reviewers, different prompts; claims spot-checked against the code (all held).

## Architect ("would you build this from scratch?") — keep and evolve
- Increment design + bit-identity exchange (per-element contributions, model-order lexsort/bincount) = what one would build; invariant checks good.
- Diverges: whole model per rank (compute accelerator, not memory scaler); constraints owned whole round-robin (contact = serial wall; partition by slave node); state sync = Allgatherv of all states to all ranks + Python loop each output increment (gather to root + allgathered marker scalars instead); field outputs evaluated on every rank.
- Risks: rebalance ignores elementsWithoutState (subdomain.py:368 only reports); fingerprint omits connectivity/coords (subdomain.py:316-321); stale non-owner footprints read by a from-scratch DofManager build (precondition unasserted); KeyboardInterrupt swallowed on one rank (driver) -> hang; topology update outside agreedOnByAllParts; no CI runs testfiles/mpi under mpirun.
- P2 right next step; cautions: partial ElementSet must refuse whole-model consumers (explicit gather API); design element migration recipe before P3.

## Teaching maintainer ("suitable for a teaching code?") — acceptable with substantial changes
- NED 2101 -> 2687 lines; solveIncrement's update wrapped in integratedValuesOf/storeIntegratedValues round trips; "read 'here' as 'everywhere'" admits the serial text no longer reads as the method.
- ModelPartition too wide (23 abstract members, >half communication/agreement); WholeModel 115 lines of no-ops; collectives belong in NEDMPI overrides. Proposal: 4-member partition (elements, constraints, dofs, ownedDofMask) + NEDMPI overrides of computeElements, assembleConstraintForces, acceptIncrement, critical dt, reportEnergyBalance; extract the energy table (NED 1214-1335).
- Shared-base hooks: writeIncrementOutput, updateConnectivityOf justified; elementsLoadedHere not.
- Issues: define() inside _assembleLumpedOperators; cached worldCommunicator() installs excepthook; updateRigidBodies twice per output increment (NED 964, 1405); hard-coded CutbackRequest 0.5 (subdomain.py:541); object-typed protocols; addNodalForces in modelpartition.py; stale solveIncrement docstring; print shadowing in run-tests CLI; rst 99-100 contradicts code (lumped operators).
- Bit-identity explained well but not verifiable from the repo (no mpirun CI, no command in docs).

## Next steps (DONE 2026-10-06, see handoff §5e)
1. Narrow partition, collectives into NEDMPI, energy table extracted, drop elementsLoadedHere.
2. Close risks: refuse stateless elements in DD/rebalance; topology update in agreedOnByAllParts; KeyboardInterrupt; fingerprint incl. connectivity+coordinates.
3. CI job: testfiles/mpi under mpirun -n 2,3 vs serial U.ref.
4. Cleanups above.
5. Revise DESIGN_P2 (explicit gather API + refusal; migration recipe) before code.
Deferred to P3: gather-to-root state sync, contact partitioned by slave node.
