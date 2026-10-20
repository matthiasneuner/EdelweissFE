# Plan P5: contact, ties and constraints with owned elements (2026-10-07)

Context: DESIGN_DISTRIBUTED_MEMORY.md (§2 constraints row, §3 layer 7, §5 P5), PLAN_P3_OWNED_ELEMENTS.md,
PLAN_P4_DISTRIBUTED_AMR.md, handoff §5g–§5i, REVIEW_MPI_2026-10-06.md (architect: contact owner as a serial
wall). Base: `feat/mpi-explicit-dynamics-v2` at `373d1496` (pushed to mn, no PR). Worktree `EdelweissFE-p5`,
branch `feat/distributed-contact`. Rabbit.

## Goal

**c1_150 runs in distributed mode** — bitwise to serial NED, no slower than the whole-model fallback, with
much less memory per process. That means every whole-model reason c1_150 triggers today leaves the rule:
- `*constraint` of the types it uses: `tie`, `surfaceToDeformableSurfacePenalty`,
  `surfaceToDiscreteRigidBodyPenalty` (also `nodeToDeformableSurfacePenalty` and the other constraints in
  `testfiles/mpi`, where clean; anything not converted keeps a stated reason in the rule);
- generators `surfaceElementGenerator` (contact facets) and `discreteRigidBodyGenerator` (point mass +
  rigid surface), both used with `executeAfterManualGeneration=True` (late generators are currently a
  whole-model reason as well — they become allowed when they only create owner-made, surface-sized
  elements, i.e. facets and point masses, decided by the generator, not by a blanket rule).

## Facts (code maps of 2026-10-02 and the P2–P4 work)

- Constraint data is surface-sized; each constraint is evaluated whole by one owning process (round-robin
  by name), forces shared by Allgatherv in model order (bitwise). The owner's subdomain contains the
  constraint's DOF footprint; force completion and lumped operators at those DOFs already work by DOF
  (interface assembly, sharing from owners), independent of which element objects exist.
- Searches read current positions from the global node fields (synchronized before every search) and
  master facet geometry; facets are elements without kernels (`hasKernels=False`), cut from the mesh (P2),
  numbered from the mesh's allocators — identical on every process.
- Ties: snapping moves replicated Node objects; `setNodes` refresh over the elements created here.
- MPC transformation is global-DOF based; the closure keeps a slave and all its masters in one subdomain.
- After P3 a process holds only its own solid elements (+ loaded neighbours). Anything a constraint reads
  through a *solid element object* (parent faces of slave facets, element sets of surfaces) must come from
  the mesh/facets/nodes instead, or the needed elements must be created where the constraint is owned.

## Work items

1. **Facets and rigid-body point masses on every process** in distributed mode (surface-sized, no kernels,
   identical numbering), and the generators' `wholeModelReason` → None. Late generators allowed when they
   create only such owner-made elements.
2. **Constraints without solid element objects**: audit every read in tie / s2s / rigid / node-to-surface
   constraints (incl. construction, refresh after AMR, restart data, footprints) and make each read mesh-,
   facet- or node-based — or create the minimal elements needed on the owner (state why). Constraint
   ownership stays round-robin (or weighted by footprint size if trivial); partitioning contact by slave
   points (the architect's "serial wall") is **not** in scope — measure the cost on c1_150 and report.
3. **Fallback rule**: remove the converted constraint types and generators; keep stated reasons for the rest.
4. **c1_150 distributed**: the harness full stage's 8x4 run then runs distributed — bitwise vs baseline,
   checkpoint == NED's, resume == straight. Measure c1_150 per-process peak RSS distributed vs fallback
   (7.5 GB/proc at 8x4) and increment time (≈0.088–0.090 s/inc at 8x4), and the contact owner's share of
   the increment.
5. **Docs**: domaindecomposition.rst (what constraints need, what is replicated, ownership, limits).

## Gates

- Bitwise everything (`verify_refactor.sh … quick|full` with `REF=…/refactor_reference_fsum`; full at most
  twice, idle machine, check `ps` before and leave no background runs). The contact/tie decks in
  testfiles/mpi (NEDContact, NEDSurfaceContact, NEDSurfaceToDiscreteRigidBodyContact, TieNED, …) run
  **distributed** (mode check EXPECTED_DISTRIBUTED updated), bitwise vs fresh serial at -n 2/3
  (`--tolerance 0`); NEDRestart1Write/2Resume (contact + restart) distributed with checkpoint == serial.
- Suites, full pytest, sphinx, lint unchanged vs base; CI workflow still valid.
- Performance: no slowdown vs the fallback on c1_150 at 8x4 (alternate base/head runs on the idle machine
  if close).
- Rules: PLAN_MPI_REFACTOR.md "Rules for both" (teaching vocabulary, docstrings, no hasattr/getattr/
  setattr, no duplication, elements parallelization-agnostic, commits with the two footer lines, stage by
  name, no push). Time cap ~5 h; stop and report rather than force anything costing bit-identity or
  needing a design change (state the question precisely).
