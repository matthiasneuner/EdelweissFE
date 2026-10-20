# Design note: contact evaluated by slave entity (draft, 2026-10-08)

Base: `EdelweissFE-mpi2` at `ca6c1d91` (feat/mpi-explicit-dynamics-v2). Plan only, no code. Context:
DESIGN_DISTRIBUTED_MEMORY.md §3.7, PLAN_P5, PLAN_HYBRID_CONSTRAINT_EXCHANGE.md (done, 5p),
REVIEW_MPI_2026-10-07(b) ("split contact by slave entity + constraint cost in balancer").

## 1. Today

- **Ownership**: round-robin by name (`partitioning.py:229-247`). The owner evaluates the whole
  constraint, and *integrates its whole footprint*: owned constraints' DOFs join the subdomain
  (`subdomain.py:310-317`).
- **Evaluation**: `NEDMPI.assembleConstraintForces` (`nonlinearexplicitdynamicmpi.py:586-619`) runs
  `NED._evaluateConstraintForce` (`nonlinearexplicitdynamic.py:2126-2168`: zeroed buffer, `applyConstraintExplicit`).
  Then `ConstraintForceExchange` (`subdomaininterface.py:446-542`) tags each entry
  `position·2³² + entryIndex` (`:60`, `:498-500`) and adds it per constraint in model order with
  `addNodalForces` (`numerics/assembly.py:34-54`). Repeated DOFs use `np.add.at`, a left fold in entry
  order, which is the serial order.
- **Search**: owner only (`subdomain.py:627-653`, `nonlinearexplicitdynamicmpi.py:1042-1064`), after
  the full U/V/P allgather (`:1110ff`).
- **State**: the frozen projection only (`frozencontactsearch.py:79-137`; s2rigid `:441-448`, s2s
  `:503-519`). There is no friction and no augmented Lagrange (s2s docstring `:213-215`). The owner
  sends it to everyone by a pickled allgather before every output (`statesynchronization.py:159`).

**c1_150 (measured, `mpi_explicit_benchmarks/constraint_p2p/runs/`)**

Penalty constraints, by model position:

| Position | Constraint | Type | Size | Entries |
|---|---|---|---|---|
| 0 | contactPlateConcrete | s2s | 1,434 points | – |
| 1 | contactShaftHole | s2s | 1,440 points | – |
| 2 | **contactBottom** | s2rigid | 10,656 facets × 3 = 31,968 points | **255,750** |
| 3, 4 | front, top | s2rigid | 144 facets each | – |

The five ties are MPCs and not affected.

| | 32x1 (`split_head_32x1`) | 8x4 |
|---|---|---|
| Increment | 91.7 ms | 88.2 ms |
| Rank 0 waits in "assemble constraints" | 7.4 ms | 7.3 ms |
| Also: wait in elements | 7.2 ms | – |
| Ranks that also integrate contactBottom's DOFs (layout dumps) | 11 (largest share 40,464 entries = 16 %) | 3 (largest 76,308 = 30 %) |

- contactBottom's owner (rank 31 at 32x1) spends about 7 ms evaluating it (domaindecomposition.rst),
  then 10.7 ms per call exchanging (post 6.0 ms, add 3.9 ms).
- The 6 ms "post" for 2.2 MB is most likely a rendezvous wait, not copying. Not investigated.
- The deck defines contactBottom/Front/Top twice (lines 253-269). This is harmless: the model keeps
  one per name.

## 2. Proposal: constraint pieces

- **Slave entity**:
  - a slave **facet** for surface contacts;
  - a slave **node** for node-to-surface contacts.
- **Evaluator**:
  - for a facet: the process of its host element (`processOfAuxiliaryElement`,
    `partitioning.py:121-142`), where the facet already lives;
  - for a slave node: the owner of its DOFs (the lowest rank integrating them).
- **Piece**: the same constraint class, built over the local part of its slave surface. A contact
  constraint already is "a constraint over a facet set" (`_buildFromSlaveSurface`, s2rigid `:276-319`).
  - One protocol method is new: `overSlaveFacets(facets)`. It is meaningful serially and never used
    there.
  - Serially, the constraint is its only piece.
  - Pieces live in `domaindecomposition` (`ConstraintPieces`). `model.constraints` keeps the whole,
    replicated definitions, so DOF-layout activation and tie snapping are untouched.
- **Invariant kept**: an evaluator integrates every DOF its piece reads.
  - The slave face nodes are already in its subdomain, through the host element.
  - The s2s master face nodes join it, as today's owner footprint does, but per piece.
- **Tags**: (constraint position, serial entry index).
  - s2rigid: facet *f* covers entries `[f·24, (f+1)·24)`.
  - s2s: point *p*'s block starts at `blockDof × (assigned points before p)` (`_assignMasterFacets`
    `:440-494`). This needs the global assigned mask after each search: one byte per point,
    allgathered at search ticks only.
  - `ConstraintForceExchange` is then reused unchanged.

## 3. Bit-identity (central)

Serial NED adds a constraint as one block, folding repeated DOFs in entry order. Pieces reproduce
this exactly if three conditions hold.

**A. Every entry value equals the serial value.**
- The s2rigid (`:493-523`) and s2s (`:616-659`) kernels are row-wise: einsum, `frozenNormals @ R.T`,
  and the per-facet quadrature sum.
- Experiment (scratch dir, env next_v2611_mpi, NumPy 2.5.2, 1 and 4 threads): 300 random contiguous
  and scattered facet subsets of a 10,656-facet replica of this kernel were **bitwise equal** to the
  whole evaluation.
- NumPy does not guarantee this. A permanent *piece-independence* unit test per splittable type must
  check it: evaluate random subsets and compare bytes.

**B. No cross-entity reduction.** One exists: the s2rigid reference-point (RP) force and moment,
`normalForces @ normals[closed]` (`:526-529`).
- This is a BLAS gemv whose summation order is implementation-defined. In the experiment it differed
  from a left fold by 1e-8 (absolute).
- It must change **serially**: an RP contribution per facet, folded in facet order by the solver's
  `np.add.at` (an RP block per facet in the local layout, or per-facet sums folded on axis 0).
- That is a one-time, rounding-level change of serial results, so it needs a new reference set.
- `nodetodiscreterigidbodypenalty.py:377-378` accumulates the RP block node by node in one buffer.
  Splitting that also changes the fold, so it needs the same restructuring.

**C. Every rank integrating a DOF receives every entry there.** The invariant guarantees it: an
evaluator is an interface neighbour of every other process integrating its DOFs.

Diagnostics (`totalNormalForce = f_n.sum()`, pressures) are in neither P nor the checkpoint. Readers
gather them.

Suspected existing defect (found by reading, not reproduced): an expression field output reading
`model.constraints[c].getNormalPressures()` reads rank 0's copy. Its `_normalForceCurrent` is never
synchronized because it is not restart data, so it holds zeros unless rank 0 owns `c`.

## 4. Where the RP DOFs are integrated

- **(a) Every piece holder integrates the RP.** Simple, but the per-facet RP entries then go
  all-to-all: about 64k values × 10 receivers ≈ **5 MB/increment** at 32x1. That is more than the
  whole exchange today (3.1 MB).
- **(b) One process integrates the RP** (the point mass's). The others receive the RP's U as a
  6-value read-only copy each increment, sent after `advanceDisplacements` and overlapping the element
  loop. The per-facet RP entries go to that process only (≈ 0.5 MB). This is the first "received
  value" of P6 (DESIGN_P6_LOCAL_DOFS.md). **Recommended.**

## 5. State, checkpoint, migration, AMR, search

- **State per slave entity**: the projection rows per point:
  - s2rigid: normal(3) and offset;
  - s2s: assignment, Nm and normal.

  Future friction history or multipliers would be per-point rows of the same kind, keyed by
  (facet number, quadrature point).
- **Checkpoint**:
  - Rank 0 assembles each constraint's arrays in serial point order from the pieces, keyed by the
    layout key `slaveFacets`. The datasets are identical, through `FrozenContactSearch.getRestartData`.
  - This replaces today's allgather before every output (gather to the reader).
  - On resume, every process reads the whole arrays and each piece adopts its rows.
- **Migration**:
  - Facets move with their hosts (`distributedelements.py:253`).
  - Migration follows an output synchronization, so: assemble the whole state (surface-sized, rare),
    rebuild the pieces, and let each adopt its rows.
  - The projection is carried, not searched again, so the result stays bitwise.
- **AMR**:
  - `refresh` (s2rigid `:450-459`, s2s `:521-542`) rebuilds from the regenerated facets and searches
    again. Pieces do this on the children's process (`placeChildElement`), point by point, which keeps
    the result bitwise.
  - `requireConstraintCopiesCurrent` (`subdomain.py:655-683`) compares a digest of the pieces' union.
- **Search**: each piece searches its own points.
  - s2rigid needs only the RP pose.
  - s2s needs the current master-facet coordinates. PLAN_GATHER_TO_READER item 3 (master-surface
    positions, surface-sized) applies, extended from "the owner" to "the piece holders".
  - `closestFacetCandidates` (`utils/facetcontactgeometry.py:416ff`) is per query point, and its
    candidates depend on the master surface only, so it is split-safe.
  - A changed footprint is still decided by `allreduceAny`.

## 6. Expected gain on c1_150

These are estimates from timers and layouts, not from a prototype.

**32x1**
- contactBottom splits over about 11 processes; the largest piece is about 16 %, ≈ 1.1 ms to evaluate.
- Rank 31 no longer integrates about 96k bottom DOFs, and no longer sends 278k values.
- Of rank 0's 7.4 ms constraint wait plus the knock-on wait in elements, about **6-10 ms/increment
  (7-11 %)** should go.

**8x4**
- Largest piece about 30 %, ≈ 2 ms.
- Gain about **4-6 ms (5-7 %)**.

**More processes**: at 64 processes and above (≈ 50 ms increments), the whole-constraint owner's
≈ 18 ms (evaluation plus exchange) becomes a serial wall. Pieces scale instead.

## 7. Cheaper alternative first

**Give the balancer the constraint cost.** Add each owner's measured constraint time to its part's
target weight (METIS `tpwgts`).

- It does not touch bit-identity.
- It probably recovers most of §6 at 8-32 processes.
- It does not scale beyond that.

After the pieces exist, each facet's cost belongs to its host element's weight, which is natural.

## 8. Risks

- Piece-independence (A) is tested, not guaranteed. The fallback is elementwise kernels.
- New serial references (B).
- s2s needs an extra allgather and offset recomputation per search tick.
- Interfaces grow by the master/RP DOFs.
- Whole and piece paths both need tests.
- The 6 ms wait is unexplained; investigate it in C0.

## 9. Phases and gates

**Every gate**: `verify_refactor.sh` quick and full, `testfiles/mpi` at `-n 2/3/4` with
`--tolerance 0`, checkpoints == serial (h5cmp), resume == straight, the mode check, and c1_150
alternated base/head.

| Phase | Content | Extra gate |
|---|---|---|
| C0 | Constraint cost in the balancer; explain the 6 ms post and the 3.9 ms add on the owner | timing |
| C1 | Serial: per-facet RP fold; piece-independence tests | new reference set, user approval |
| C2 | s2rigid pieces, RP option (b), state/checkpoint/migration/AMR | forced facet migration deck; c1_150 8x4 and 32x1 |
| C3 | s2s pieces (assigned mask, master gathers) | NEDSurfaceContact, c1_150 |
| C4 | Node-to-surface pieces (optional) | NEDContact |
| C5 | Facet costs in host-element weights | timing |

## 10. Decisions for the user

1. **C0 first**, and continue to C1-C3 only if pieces are still worth it after measuring? Recommended:
   yes.
2. **Accept a one-time change of serial s2rigid results** (the per-facet RP fold) and a new reference
   set? It is required for bitwise pieces.
3. **RP integration**: (b) one process plus a 6-value received copy (recommended), or (a) every piece
   holder?
4. **Evaluator rule**: the host element's process for facets and the DOF owner for nodes. Pieces are
   built through a new `overSlaveFacets` method on contact constraints, and the piece vocabulary stays
   in `domaindecomposition`. Acceptable?
5. **Scope**: s2rigid and s2s; node-based types later. equalValue and directionalSpring stay whole.
6. **The suspected stale `getNormalPressures` on rank 0**: reproduce and fix it independently of this
   work?


## Decisions (user, 2026-10-08)

C0 first (constraint cost to the balancer as METIS weights), then decide whether to split contact (C1–C3).
Order after gather-to-reader: C0, then P6.
