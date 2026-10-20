# Plan: teaching restructure of the distributed-memory MPI work (2026-10-07)

Source: the professor's review in REVIEW_MPI_2026-10-07.md (all claims spot-checked). Base:
`feat/mpi-explicit-dynamics-v2` at `ac025e4f` (pushed to mn, no PR). Worktree `EdelweissFE-teach`, branch
`refactor/teaching-restructure`. Rabbit, env next_v2611_mpi. This is a pure restructuring: every result stays
bitwise, nothing gets slower.

## User decisions (2026-10-07)

- **One element loop, made readable.** Keep the single shared plan-based loop (NED, NEDParallel, NEDMPI —
  no duplication, no speed loss), but NED must read like the central-difference method: the increment shows
  the scheme in a few named calls (e.g. `assembleInternalForces`, `assembleExternalLoads`,
  `updateVelocities`, `advanceDisplacements` — choose good FE names), and the plan's implementation
  explains itself in teaching terms where it lives. Owned-DOF masks, node-field slots, the deferred
  external-work queue, `releaseEquationSystem`, and anything else that exists only for MPI move behind
  NEDMPI overrides or into named helpers; serial NED should mention the partition at most where it is
  genuinely part of the method (e.g. "the degrees of freedom integrated here").
- **Mesh stays `model.mesh`**, documented clearly: element connectivity, sets and surfaces; nodes are
  `model.nodes` (they join the mesh in P6). Docstrings and mesh.rst say so up front.
- **Standard vocabulary**: replace project-specific terms with the domain-decomposition literature's
  (owned / ghost elements, local / global, halo, rank, owner): e.g. "created here / declined", "made by its
  owner", "reported here", "settle external work", "agreed on by all parts", "whole-model reason",
  "hostElement" stays if FE-correct. Code, docstrings, docs, tests. Keep names a student meets in deal.II /
  PETSc docs; avoid inventing.

## Work items (the professor's five, plus the small findings)

1. **NED readable** as above.
2. **Mesh as single source of truth by construction**: `model.elements` cannot silently diverge from the
   mesh — the consistency check runs both directions (an element in `model.elements` not described in the
   mesh raises), or `model.elements` becomes a read-only derived view; `FEModel.createElement` (today the
   special path for owner-made facets/point masses) gets a name that says so, and the normal path is the
   obvious one (`createElementsOfMesh`); update tests that build models programmatically to the normal path.
3. **Partial element sets fail safe**: iterating a set that is not complete on this process raises; a reader
   that wants the local part asks explicitly (e.g. `localElements()` / `ownedElements()` in standard terms);
   convert every consumer (≈57 `elementSets[...]` reads) to the right one; serial runs (always complete) are
   unaffected.
4. **MPI obligations off the shared base classes**: `TimeStepperBase.incrementsLeftEstimate` must not be
   abstract (a sensible default, the MPI balancer copes with "unknown"); collect the per-class distributed
   capability declarations (`wholeModelReason` → standard name) so a student's subclass needs nothing to keep
   working (safe default, documented once); reduce `FEModel.elementDistribution`'s null-object hooks to the
   minimum (ideally one object the model holds, with a trivial serial implementation, not ~10 hooks).
5. **Bit-identity claim tightened**: docs state exactly what is bitwise (U, V, P, element and constraint
   states, checkpoints; which diagnostics are not, if any), name the `np.bincount` sequential-accumulation
   dependence (or replace it with an explicitly ordered sum if that is cheap and bitwise), add a unit test of
   the interface assembly ordering with adversarial values (like the load assembly's), and extend the
   "verify it yourself" recipe to state variables and checkpoints.
6. **Small findings**: revert the CRLF→LF conversion noise of `timesteppers/adaptivetimestepper.py` (keep
   only the real change, in the file's original line endings); the `if subset is not None:` serial behaviour
   change for an empty element set in `inputfilehelpers.py` — decide and document (an empty set should give
   an empty output, not the whole field — state it); the `mesh` name clash in `hadaptivity.py` (octree
   `AdaptiveMesh` vs `model.mesh`) — rename the octree variable; trivial wrappers `_lumpedInertiaOf/
   _lumpedDampingOf`; the prototype element built with number 0; boxgen's `Node(-1)` sentinel objects
   (a sentinel label without a Node object, or document); `ElementSetOfMesh.__init__` parameter `label`
   documented as `name`; the rigid-body model leak (a finished model with a rigid body stays alive in the
   process through a closure cell holding {'model': …}) — find the root and fix if small.

## Gates

- Bitwise everything: `verify_refactor.sh … quick` after each code commit, `full` once at the end on an idle
  machine (REF=…/refactor_reference_fsum; check ps; leave no background runs); testfiles/mpi -n 2/3
  `--tolerance 0` vs fresh serial `--create`; mode check; full pytest; suites; sphinx (no new warnings);
  lint. No slowdown (c1_150 serial32 and 8x4, alternate base/head if close).
- Report: NED line count before/after and a short "how to read NED" walk-through (the increment's named
  steps), the renames table (old → new), and the consumer list of item 3.
- Rules: PLAN_MPI_REFACTOR.md "Rules for both" (teaching vocabulary, numpydoc docstrings, no
  hasattr/getattr/setattr, no duplication, elements parallelization-agnostic, small Conventional Commits with
  the two footer lines, stage by name, no push). Time cap ~6 h; stop and report rather than force anything
  costing bit-identity or performance, or needing a design decision (state the question precisely).
