# Plan P4: adaptive mesh refinement with owned elements (2026-10-07)

Context: DESIGN_DISTRIBUTED_MEMORY.md (§3 layer 6, §5 P4), DESIGN_P2_MESH_AS_DATA.md (§3 guard, §5b, §9,
§10), PLAN_P3_OWNED_ELEMENTS.md, PLAN_MIGRATION.md, handoff §5g/§5h. Base: `feat/mpi-explicit-dynamics-v2`
at `ef12738d` (pushed to mn, no PR). Worktree `EdelweissFE-p4`, branch `feat/distributed-amr`. Rabbit.

## Situation

Distributed mode (P3) creates only a process' own elements (+ loaded neighbours); migration moves elements
when rebalancing. Every `*modelModifier` — hence live AMR (hAdaptivity) — still triggers the whole-model
fallback. AMR today (map of 2026-10-02, summarized in DESIGN_DISTRIBUTED_MEMORY.md §2):
- initialization reads every element object (sets, properties, node labels for connected components, node
  coordinates for the octree roots);
- marking reads field outputs over whole sets (element field outputs now allgather whole-set results in
  distributed mode), halo growth by node adjacency, and a recovery-error marker averaging over the
  elements sharing a node;
- planning, 2:1 balance, hanging-node detection, conformity are pure topology over the replicated octree;
- materialization creates children from the parent's geometry, assigns the parent's section/properties,
  transfers state parent → child (parent's state and quadrature-point coordinates only), interpolates node
  values, removes parents, rebuilds sets/surfaces, mirrors into the mesh (P2);
- new node/element numbers come from global counters in creation order (deterministic if every process
  runs the same plan); restart replays the topology history on the root mesh.

## Goal

Live AMR runs in distributed mode: the octree topology, the plan and all numbering stay **replicated and
deterministic on every process** (they are data, cheap, and global by nature), while **element objects and
states stay owner-local**: the owner of a parent creates its children, transfers the state from its own
parent and drops the parent; other processes only update the mesh. Results bit-identical to serial NED.

## Work items

1. **AMR from the mesh.** Initialization reads `model.mesh` (records, sets, node coordinates/labels), never
   element objects; `requireCompleteMesh` in hAdaptivity is replaced by mesh-based reads.
2. **Marking in distributed mode.** Field-output markers read the (allgathered) whole-set results — verify
   identical marks on all processes. Recovery-error marker: make it read only gathered/global data (node
   fields are global; anything element-based through the gather path), or keep it under the fallback rule
   with a reason if that is not clean — report which.
3. **Owner-local materialization.** Children owned by the parent's owner (record in the element
   distribution); only the owner creates them (`createElementOfMesh`), assigns section/properties, transfers
   state, and drops the parent; every process updates the mesh, sets, surfaces and nodes identically.
   Loaded-neighbour halo and the equation system are rebuilt as after any topology change (the existing
   path). Then the normal rebalancing check (migration + gain guard) may move elements.
4. **Restart.** The replay creates children only where the replayed distribution says; element states
   restored by number (checkpoints gathered to rank 0, unchanged format).
5. **Fallback rule.** hAdaptivity leaves the rule (other model modifiers, e.g. surface snap, stay unless
   trivially mesh-only — report); hanging-node constraints created by AMR are not deck `*constraint`s —
   make sure the MPC closure keeps a slave and all its masters in one subdomain (it does for replicated
   models; verify with owned elements). Contact still falls back (P5) — so c1_150 stays in the fallback.
6. **Memory**: the AMR mirror (~4.4 KB of Python objects per root element, ~334 MB on c1_150) — measure it
   per process in distributed mode; compacting it into arrays is optional here (report the cost/benefit;
   do it only if clean and bit-identical).
7. **Docs**: domaindecomposition.rst and the AMR docs (what is replicated, what is owner-local, the rule).

## Gates

- Bitwise everything (`verify_refactor.sh … quick|full`; full at most twice). The AMR decks without contact
  (`testfiles/mpi/marmot/GCDPNEDExplicitAMR`, `GCDPNEDExplicitHyperbolicAMR`, `NEDLiveAMR`,
  `NEDLiveAMRStepEndsOffInterval`) must now run **distributed** (mode check updated: EXPECTED_DISTRIBUTED),
  bitwise vs fresh serial at -n 2/3 (`--tolerance 0`); add a deck where live AMR + rebalancing migrate
  children, and one AMR deck with restart written after a refinement in distributed mode (checkpoint ==
  serial, resume == straight).
- A larger live-AMR model without contact (e.g. the 192k GCDP block with a live fieldOutput marker, or a
  boxgen beam) at 8 processes: per-process memory distributed vs fallback, refinement time per event.
- Suites, full pytest, sphinx, lint unchanged vs base; CI workflow still valid.
- Rules: PLAN_MPI_REFACTOR.md "Rules for both" (teaching vocabulary, docstrings, no hasattr/getattr/
  setattr, no duplication, elements parallelization-agnostic, commits with the two footer lines, stage by
  name, no push). Time cap ~5 h; stop and report rather than force anything costing bit-identity or
  needing a design change (state the question precisely).
