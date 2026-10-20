# Plan: P6.4 — compact AMR mirror, then local nodes under AMR (2026-10-16)

Source: DESIGN_P6_LOCAL_DOFS.md §5 (AMR, recovery-error marker rows), §7, §10 P6.4, decisions 2026-10-08/14/15
(item 2: the AMR registry stops storing coordinates of model nodes and reads them from the mesh;
`syncNodeCoordinates` disappears). Base `feat/mpi-explicit-dynamics-v2` at 4a3fb63b (pushed).
Note: c1_150 has contact too, so it keeps every node until P6.5; for it P6.4 brings only the mirror's memory.

## P6.4a — the AMR mirror as compact arrays (serial and MPI alike)
Worktree `EdelweissFE-p64`, branch `feat/p6-4-compact-amr`.
1. Replace the per-element dicts / NodeKey tuples / `_keyOfPoint` / coordinate dict of the h-adaptivity
   mirror (adaptivity/refinement.py, rootentities.py, hadaptivity.py) by a struct of arrays (int32 connectivity,
   level/parent/first-child/root/active, int64 lattice box origin+edge, NodeKeys packed into int64 keys); the
   algorithms stay as they are and read as before (teaching: keep the octree explainable — named accessors, no
   index soup).
2. The mirror stores no physical coordinates of model nodes: corner/edge coordinates are read from
   `model.mesh` at subdivide time; new nodes are described into the mesh when minted; `registry.coordinates`,
   per-leaf `coords` and `syncNodeCoordinates` disappear; surfacesnap/tie move nodes only through
   `FEModel.moveNodes`.
3. Gate: byte-identical refinement plans, node/element labels, topology histories and fingerprints (serial and
   MPI) on every AMR deck (testfiles + tests: NEDLiveAMR, GCDPNEDExplicitAMR, AMR_* decks, hadaptivity tests),
   c1_150 replay byte-identical history + bitwise full harness; memory of the mirror on c1_150 before/after
   (expect ~334 → ~55 MB per process).

Compatibility (accepted 2026-10-16 after review): subdividing from the mesh changes coordinates by 1-2 ulp
where a level>=1 cell's node was created by a differently oriented neighbouring root on a warped/non-dyadic
mesh (reviewer's A/B: ~15 level-2 coordinates). Checkpoints written before this branch with level>=2
refinement on general meshes stop on resume with a fingerprint mismatch (loud); level 1 is unaffected.
Packed node keys are Python integers (100-200 bits), numbered per cell as int32 -- not int64 keys.

## P6.4b — local nodes under AMR
Base `feat/mpi-explicit-dynamics-v2` at 072b1b14 (pushed; P6.4a merged; user accepted the multi-level checkpoint
compatibility break 2026-10-16). Worktree `EdelweissFE-p64b`, branch `feat/p6-4b-amr-local-nodes`.
AMR leaves the whole-model rule (contact/rigid bodies still hold every node): the topology is decided on the
replicated mirror; Node objects/fields only for the subdomain; new node values interpolated by the owner of the
parent element from local values (interface nodes redundantly, identical bits); "fields resize & restore" local;
recovery-error marker: interim transient allgather of its one node field at topology checks (documented as
O(N), rare). Gate: AMR decks without contact at -n 2/3/4 tol 0 vs fresh serial NED, incl. live refinement +
migration + restart; memory on an AMR block deck at 8/32.

## Gates (each round)
Bitwise everything: verify_refactor.sh quick after each code commit, full once at the end on an idle machine
(BASELINE=mpi_explicit_benchmarks/refactor_reference_marmot_e992a3f4/c1_150_baseline.npy); testfiles/mpi -n 2/3/4
`--tolerance 0` vs FRESH serial NED; checkpoints; mode check; full pytest of BOTH `tests` and `edelweissfe` (base: 3
test_tie failures); suites (base: CDPPlane); sphinx; lint. Rabbit, env next_v2611_mpi (editable install points at
EdelweissFE-mpi2: worktree first on PYTHONPATH, print edelweissfe.__file__; copy .so + built_extensions.log).
Rules: PLAN_MPI_REFACTOR.md "Rules for both"; small Conventional Commits; stage by name; no push; no helper agents;
WIP commits not git stash; waiters poll the file that receives the marker; no leftover processes. Cap ~6 h per
round; stop at a clean gated commit if it doesn't fit; stop and report rather than force a design decision.
