# Plan: P6 round 1 — mesh owns the nodes, numbering as data (P6.0–P6.2) (2026-10-14)

Source: DESIGN_P6_LOCAL_DOFS.md (§3, §4, §6.1, §10, §11 and both "Decisions" sections). Base:
`feat/mpi-explicit-dynamics-v2` at `1ac71f9d` (pushed). Worktree `EdelweissFE-p6`, branch `feat/p6-local-nodes`.
Rabbit, env next_v2611_mpi — NOTE the env's editable install points at EdelweissFE-mpi2: run everything with this
worktree first on PYTHONPATH and print `edelweissfe.__file__` before trusting any result. References: harness default
(`refactor_reference_marmot_e992a3f4`, BASELINE `…/c1_150_baseline.npy`).

## Work
- **P6.0 (measure only, small commits if cheap):** free the parsed input after model construction on every rank;
  lazy/deferred imports on non-root ranks only if trivial. Report RSS per rank on c1_150 8x4 before/after.
- **P6.1:** the Mesh owns node data as arrays (labels, reference coordinates, plus whatever generators/readers need);
  generators describe nodes into the mesh (as P2 did for elements); `createNodesOfMesh(isHeldHere)` makes Node objects
  afterwards — serially all of them, so serial behaviour and reading path are unchanged. `NodeSetOfMesh` = the part
  held here, with the ElementSet guard (`isComplete`/`requireComplete`), loud on incomplete use. In this round NEDMPI
  still holds every node (isHeldHere always true); readers of `model.nodes` that will need non-local nodes later are
  listed (not converted) in the report, except where conversion to mesh arrays is trivial and natural.
- **P6.2:** the global DOF numbering as replicated data (per field node-membership + offsets, scalar variables last,
  §4), built from the mesh + constraints' fieldsOnNodes, asserted bitwise-equal to the DofManager's numbering on every
  deck (an always-on cheap assertion under NEDMPI, plus tests incl. AMR and contact decks and c1_150). Include its
  digest in the layout fingerprint.
- Docs: domaindecomposition.rst / model docs for mesh nodes and node sets.

## Gates
Bitwise everything: verify_refactor.sh quick after each code commit, full once at the end on an idle machine (check
ps; no leftover processes); testfiles/mpi -n 2/3/4 `--tolerance 0` vs fresh serial NED; mode check; checkpoints ==
serial; full pytest (base failures: 3 in test_tie.py); suites edelweiss-only + marmot (base: CDPPlane); sphinx (17
warnings base); lint. Serial timing must not regress (serial, serial32), c1_150 8x4 time + per-rank RSS base/head
alternated. Rules: PLAN_MPI_REFACTOR.md "Rules for both" (teaching-oriented, no duplication, no
hasattr/getattr/setattr, loud invariants, elements parallelization-agnostic); small Conventional Commits with the two
footer lines; stage by name; no push; no helper agents. Time cap ~5 h; stop and report rather than force anything
costing bit-identity or needing a design decision.
