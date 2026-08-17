# The Topology Pipeline — Implementation Plan

**One-line thesis.** A restart replay must be a *faithful re-execution* of the recorded topology
history, not a reconstruction that happens to land on the same mesh. Everything below follows from
that.

**Status.** Design agreed 2026-08-17. Supersedes `PLAN_ELEMENT_IDENTITY.md` (deleted — its
"identity dict" proposal was symptom treatment and was rejected; §1.4 records why, so it is not
re-proposed).

**Companion docs.** `ELID_UNIQUENESS.md` (problem statement + pattern survey),
`PLAN_BLOCKAMG_EFFICIENCY.md` §restart (how the bug surfaced), `PLAN_RESTART.md` (restart v1 scope),
`PLAN_AMR_CONTACT.md` (the P0–P4 phasing whose deferred items resurface here).
Diagrams: <https://claude.ai/code/artifact/e8fac0d5-bf65-427f-9e88-65ba74cc1cce>.

---

# 0. Strategy: branches, machines, verification

## 0.1 Branch strategy

**Base: `perf/blockamg-efficiency` @ `d8dd996`** — not `next_v26.11`, and not `perf/linsolve-restart`.

The reason is not convenience. P1–P4 development needs `1393803` (the restart hotfix) and `b0eb74f`
(eid-batched replay) **present and working**, because without them a deep AnchorPryOut resume does
not converge at all (13 return-mapping failures, cutback spiral) — and that resume is the
verification workload for this whole plan. Both commits are deleted by P5; until then they are the
working baseline. The blockamg perf commits that ride along (`b10d765` Inv 1, `f513e0a` Inv 4) are
solution-preserving and orthogonal.

**Two branches, not one:**

| branch | base | content | when |
|---|---|---|---|
| `perf/amr-topology-round` | `d8dd996` | **P0** + the three AMR-side items of **P6** (hanging-node classification, field-variable relink, set sync) | **first, independently** — a live-run AMR win that should not wait behind a large refactor, and it is what pays for reverting the batched replay |
| `feat/topology-pipeline` | `perf/amr-topology-round` | **P1–P5, P7**, plus the facet half of P6 (needs P3) | after |

Keep P1 (allocator + mutation window) as the *first two commits* of `feat/topology-pipeline` so it
can be split into an early PR on its own: it is small, independently valuable (it fixes the live
element-number clash hazard patched by hand at `hadaptivity.py:509-517`), and reviewable without any
pipeline semantics.

**PR base** is whatever integration branch the AMR+restart stack is accumulating on — today
`perf/blockamg-efficiency`, itself based on `perf/linsolve-restart`, because the restart feature
does not exist on `next_v26.11` at all (`getRestartData` is not there). Landing the whole AMR +
restart + pipeline stack on `next_v26.11` is a separate exercise and is **out of scope here**;
do not try to rebase this work onto `next_v26.11` directly.

Push to `mn` only, per the workspace convention.

## 0.1a Integration with `next_v26.11` — do NOT merge at the top (revised 2026-08-17)

**#57 is merged upstream (squashed as `0612a4d5`), and the correct response is to do nothing about
it here.** An earlier version of this section said "land #57, then merge `next_v26.11` into this
branch before P3". That was wrong, and `HANDOFF_PR_STRATEGY.md` is why.

**This branch is the top of a seven-deep unlanded stack.** Verified by ancestry — every one of these
is an ancestor of `feat/topology-pipeline`, and **none** is upstream:

```
feat/amr-hanging-nodes (#64, still a DRAFT PR)
  -> feat/amr-recovery-marker -> feat/amgcl-lgmres-outer-solver
  -> perf/linsolve-investigation -> feat/restart -> perf/linsolve-restart
  -> perf/blockamg-efficiency -> feat/topology-pipeline  (here)
```

The project lands such a stack **bottom-up**, rebasing each branch onto the then-current
`next_v26.11` as its turn comes (`HANDOFF_PR_STRATEGY.md`, "Next steps" item 5). So:

- The reconciliation debt against #57 belongs to **`#64`'s rebase at the bottom**, not to a merge at
  the top. Paying it here would pay it twice.
- A trial `git merge origin/next_v26.11` produces **43 conflicts** (25 source/doc + 18 add/add
  testfiles), concentrated in exactly the contact files: `surfaceelementgenerator.py` 16 hunks,
  `nodetodeformablesurfacepenalty.py` 16, `nonlinearimplicitstatic.py` 10, `femodel.py` 1. That is
  the stack's debt made visible, not this branch's integration task.
- A merge commit in the middle of a stack that will later be rebased/flattened is debt in the wrong
  place, and it would drag contact work this branch has no need for.

**Nuance on rebase vs merge, since the project prefers rebasing:** rebasing is right for *prepping a
branch for its own PR*. It is **not** right for *updating a long messy branch onto a moved base* --
`HANDOFF_PR_STRATEGY.md` records exactly that experiment on #57 ("Did not do a linear git rebase ...
tried it first, aborted immediately" -- ghost conflicts from superseded history on the very first
replayed commit) and a merge was used instead. Same conclusion as
`memory/feedback_rebase_vs_merge_long_branch.md`. Neither applies *yet* here, because this branch's
turn has not come.

### Consequences for the work order

| phase | files | debt against #57 | do it now? |
|---|---|---|---|
| **P5** restart onto plans + fingerprint | `femodel.py` (1 hunk), `hadaptivity.py` (**0**), `modelmodifierbase.py` (0) | negligible | **yes -- this is where the original bug dies** |
| **P7** domain/round conflict detection | `modelmodifierbase.py`, `hadaptivity.py` | none | yes |
| docs (§9) | new file | none | yes |
| **P3** facets become a modifier | `surfaceelementgenerator.py` | **16 hunks, the worst file** | **defer** until the stack catches up |
| **P6 item 1** incremental refresh | `surfaceelementgenerator.py`, `tie.py` | worst | defer, blocked behind P3 |

**When P3 does come**, design it against #57's *final* `surfaceelementgenerator.py` rather than this
branch's, so the reconciliation is "take theirs, then apply the modifier wrapper" instead of a
hand-merge of two rewrites.

### A correction to this plan's own test method

`HANDOFF_PR_STRATEGY.md` ("Working-directory note") warns that running `testfiles/marmot/` from a
worktree with copied `.so` files produces **~74 spurious failures** from a `MarmotElementWrapper`/
`BaseElement` module-identity mismatch -- the same setup every suite run in this plan used. Those
runs showed 6 failures, not 74, and base-vs-feature sets were identical, so the *comparisons* here
hold. But the absolute marmot numbers from a worktree are not trustworthy on their own: **confirm any
marmot-suite verdict in the main checkout** before treating it as a property of the code.

## 0.2 Machines

| machine | role |
|---|---|
| **xeon** (36 cores) | **all** builds, profiling, benchmarks and AnchorPryOut verification. Holds the pinned checkpoints (`restart_ckpt_470_*.h5`, `LATE_incr466…477.h5`) and the probes in `examples/AnchorPryOut/`. |
| **x9** | canonical handoff-doc store, editing, fast unit/integration tests (L1 below) |

Rules for anything measured on xeon — these are not optional, they cost hours when skipped
(`PLAN_BLOCKAMG_EFFICIENCY.md` §1b and §5):

- **The machine must be idle.** Wall-clock comparisons are meaningless otherwise; prefer iteration
  counts and fingerprints, which are load-robust.
- **Fixed `OMP_NUM_THREADS=16` across every arm** of a comparison; `PYTHON_GIL=0`.
- **Check what is already running first.** The long Inv 5+6 A/B may still occupy xeon, and two runs
  against the same working tree and build cache will corrupt each other's results.
- **Pin comparisons to the same increment.** Comparing across different increments produced a
  completely false lead once already.
- Setup + replay is a fixed ~500 s per resume and is identical across arms — subtract it before
  quoting any delta.

## 0.3 The verification ladder

Three scales, cheap to expensive. Do not skip upward.

| level | where | what it proves | cost |
|---|---|---|---|
| **L1** unit + integration (§8 tests 1–12) | x9 or xeon | pipeline mechanics, determinism, fingerprint behaviour, replay-uses-no-markers | seconds |
| **L2** `examples/AnchorPryOut/tie_restart_probe.py` (harness (c)) | either | fingerprint-identical replay on a 2-block C3D20 model with a tie **and** AMR. **Verify `amr_hanging`/element counts actually changed, or the run is vacuous.** | seconds–minutes |
| **L3** AnchorPryOut at production scale, resume ~increment 470 | **xeon, idle** | the real workload: 380k dof, GCDP, contact + tie + AMR, multi-modifier interleaving | ~500 s + increments |

## 0.4 Can the existing pry-out checkpoints verify the new framework?

**Yes for three of the four things it needs to prove — with one conversion step — and no for the
fourth. That distinction matters.**

The pinned checkpoints were written *after* `b0eb74f`, so they record AMR's committed occasions as
`occasionEids`. **That is exactly the plan sequence the new design wants.** So the throwaway bridge
of §4.3 is not a wholesale old reader; it is a small **converter**:

```
old checkpoint                        synthesized topologyHistory
  occasionEids[i]          ────▶        TopologyRecord(round=1, modifier="amr",
                                                       plan=RefinementPlan(eids=occasionEids[i]))
  stateEids/stateData      ────▶        (dropped — state is restored by number after replay)
  occasionLabels/pendingLabels ──▶      (dropped — element numbers, not reproducible; §5.1)
```

The facet modifier needs no conversion: it re-derives its own plans per round from the
surface-touching changes (§10, open question in P3).

**What the converted checkpoints do prove:**

1. **End-to-end health at production scale.** Resume at 470 through the new pipeline and expect
   **0 cutbacks, 0 return-mapping failures**, with per-solve iteration counts for increments 471/472
   matching the uninterrupted run — numbers already recorded during the hotfix validation. This is a
   sharp test, not a smoke test: mis-restored state showed up there as 13 rm-failures and a cutback
   spiral.
2. **Multi-modifier realism.** Real ties, real contact, real AMR at 380k dof — the interleaving no
   toy model reproduces.
3. **P6 profiling at scale**, on a genuine refinement occasion.

**What they cannot prove: faithfulness itself.** They predate the fingerprint, and the original
run's element numbering was never recorded, so there is nothing to compare against. After conversion
the continuation will use *different* element numbers than the original live run did — the old
interleaving is unrecoverable by construction. That is expected and harmless; it is simply not
evidence of faithfulness.

**The cheap path to production-scale faithfulness** (do this once, in P5 — the old checkpoints are a
launchpad, not the reference):

1. resume from the converted `restart_ckpt_470` and run ~4–6 increments, **writing new-format
   checkpoints with fingerprints**;
2. from the new checkpoint at increment *k*, continue uninterrupted to *k+3* — **arm A**;
3. separately resume from that same checkpoint *k* and run to *k+3* — **arm B**;
4. assert per-increment fingerprints identical and element state identical element-for-element.

Cost: one ~500 s setup plus a handful of increments per arm — **hours, not the 1 d 20 h** a fresh run
to increment 470 would take. This is the L3 form of §8 test 9.

---

# 1. Why

## 1.1 The observable bug

Restarting a run deep in AnchorPryOut produced return-mapping failures and a cutback spiral. Cause:
`FEModel.readRestart` restored element material state **by element number**, and AMR-refined child
element numbers are not reproducible across a replay — 5211 solids were renumbered, so damaged
elements were restored virgin (and some numbers landed on stateless facet elements, crashing with
`NotImplementedError`).

Hotfixed in `1393803` (modifiers restore their own elements by octree eid and publish
`restoredElementLabels`; `readRestart` skips those). That works but leaves the split responsibility
this plan removes.

## 1.2 Why the numbers drift

Everything the AMR replay does *in its own namespace* is already deterministic:

- octree eids come from a private counter (`adaptivity/refinement.py:199-202`) advanced only by
  `refine()`, and `_replayOccasionsByEid` (`hadaptivity.py:736-740`) refines the recorded eids in
  **recorded order**;
- node labels are coordinate-keyed (`NodeRegistry`, `refinement.py:48-107`) and minted in that same
  order.

The only non-reproducible quantity is `elNumber`, because **two parties mint from one global
namespace during the analysis**, and the replay does not reproduce their interleaving:

| site | when | reproducible |
|---|---|---|
| setup generators (`abqmodelconstructor`, `boxgen:194`, `pipegen:237`, `planerectquad:170`, `cuboidlatticegenerator`, `microstructuregenerator`, `discreterigidbody:84`) | setup only, from the `.inp` | **yes** — the reconstruct phase re-runs them identically before `readRestart` |
| `surfaceelementgenerator.py:213` (`buildContactFacets`) | setup **and** every tie/contact reconcile — deletes and re-mints the whole facet set | **no** |
| `hadaptivity.py:323, 517, 556` | every materialisation | **no** |

Live, AMR children and facet rebuilds interleave per occasion. On replay, `_replayOccasionsByEid`
materialises **once per refinement level** (the `b0eb74f` 2.04× win) while facets are rebuilt from
the observer cascade at different points.

## 1.3 The deeper defect

Element numbers are only the first *observable* symptom. The real defect:

> **Replay runs through different code than the live run.** `setRestartData` is a bespoke
> reimplementation of what `updateModel` does live. Two implementations of one mutation always
> drift.

The drift is already visible as three "replay differs from live, but we argued it's fine" patches:

1. `_replayMode` skipping state transfer and warm-start interpolation (`hadaptivity.py:524, 560-563`);
2. `_isFirstCall = False` on the replay path (`hadaptivity.py:794`);
3. level-batched materialisation vs. per-occasion live (`hadaptivity.py:742-755`).

Each is a correctness *argument*, not a guarantee, and each must be re-argued when anything nearby
changes. Additional latent instances found while writing this plan:

4. `hadaptivity.py:815` still restores pending marks **by element number**
   (`{model.elements[int(label)] for label in data["pendingLabels"]}`) — latent only because
   `_pendingMarkedElements` is normally empty at checkpoint time (cleared at line 449);
5. `newChildEids` is a `set` (`hadaptivity.py:545`) and element numbers are assigned in its
   **iteration order**, which is not stable across differently-sized sets;
6. `buildContactFacets` deletes stale facets (lines 208-209) *before* reading
   `max(model.elements)+1` (line 213), so freed numbers are recycled — while
   `ModelChange.mergedWith` (`modelchange.py:85`) explicitly documents "Element/node labels are
   never reused" as an assumption. It does not bite today only because facet rebuilds emit no
   changeset;
7. **(found 2026-08-17, fixed in `a8bfda33`)** every parsed `*elSet` was built as
   `ElementSet(name, set(els))` (`abqmodelconstructor.py:189`). `ElementSet` is an `OrderedSet` that
   keeps the order it is fed, and elements are identity-hashed — so the member order came from
   object *addresses*. Measured: stable only for a bit-identical allocation history, reshuffled by
   any perturbation of it, and a resumed run has exactly such a perturbation (`*restart` in the
   `.inp`, an extra open file). That order reaches numbering: an element-based `*surface` is built
   from such a set (`abqmodelconstructor.py:237`) and `surfaceElementGenerator` walks it handing out
   sequential facet labels. **So contact/tie facet numbering was already irreproducible between a
   run and its own restart, independently of AMR.** Facets are stateless, so restored state was not
   corrupted directly, but everything minted after them shifted. See §2.3;
8. the three penalty constraints mint facets from a **pull** tick (`nodetodeformablesurfacepenalty.py:461`,
   `nodetorigidsurfacepenalty.py:172`, `nodetodiscreterigidbodypenalty.py:323`), i.e. at "the
   consumer's next increment", which is not a point the recorded history contains at all.

## 1.4 Rejected alternatives (do not re-propose)

| alternative | why rejected |
|---|---|
| **Checkpointed `(scope, localKey) → elNumber` registry + creation factory** (ELID Pattern C) | buys *stable* numbers, which nothing needs — verified: nothing persists an element number across a restart except `hadaptivity` itself, and nothing indexes arrays by element number. Cost: a persisted map, adoption semantics, reserved-vs-live bookkeeping, a factory API, a new ordering dependency. |
| **Lineage on the element** (ELID Pattern A) | needs every element constructor touched; `MarmotElementWrapper` is a `cdef class` (`elements/marmotelement/element.pyx:56`) and cannot take arbitrary attributes. |
| **Integer bands per creator** (`AMR_BASE + eid`) | with N runtime minters you must partition a fixed int32 budget (Ensight writes int32 element ids, `ensight.py:261`) among creators whose count and capacity are unknown until the input file is parsed. A wrong capacity guess wraps silently into a neighbouring namespace. |
| **A published identity string** (`model.elementIdentities`) | treats the symptom: state placement is protected while the replay stays unfaithful, so the next unfaithful-replay bug is only postponed. Also a second ID system where one suffices. |
| **Removing facets from `model.elements`** as a prerequisite | explicitly out: the design must work with facets as ordinary numbered elements. Kept only as an optional, independent performance change (§7 P8). |

---

# 2. Architecture

## 2.1 The three phases of an increment

```
╔══════════════════════════════════════════════════════════════[ window OPEN ]══╗
║ PHASE 1 — TOPOLOGY UPDATE   the only place elements/nodes are born or die     ║
║                                                                               ║
║   amr ─────────▶ printer ─────────▶ facets ─────────▶ any plan applied?       ║
║   plan(model)    plan(model)        plan(model)              │   │            ║
║      ↓              ↓                  ↓                     │   │            ║
║   apply(…)       apply(…)           apply(…)                 │   │            ║
║        modifiers run in declared order                       │   │            ║
║   ┌──────────────────── yes: run another round ◀─────────────┘   │            ║
║   └──▶ (back to amr)                          no: fixed point ───┼──┐          ║
╚══════════════════════════════════════════════════════════════════╪══╪═════════╝
                                                                   │  │
╔══════════════════════════════════════════════════════════════════▼══▼═════════╗
║ PHASE 2 — REFRESH  [ window CLOSED ]  pure readers: nothing born, nothing dies ║
║   tie_1        contact_1        bc_top        …        order irrelevant        ║
║   each asks: has topologyVersion moved since I last looked?                    ║
║              → refresh once, from the net change across ALL rounds             ║
╚═══════════════════════════════════════════════════════════════════════════════╝
                                      │
╔══════════════════════════════════════▼════════════════════════════════════════╗
║ PHASE 3 — SOLVE   assemble the equation system and iterate to convergence      ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

The two-phase skeleton already exists at `solvers/nonlinearimplicitstatic.py:260-265` (modifier
sweep, then constraint sweep, one pass each). This plan turns the first into a rounds-to-fixed-point
loop and formalises the second.

## 2.2 Terminology

Two names are already taken and must be routed around: `*modelUpdate` is a step action
(`stepactions/modelupdate.py`, fired from `steps/base/stepbase.py:161-163`), and `Journal` is the
logging class.

| today | name | rationale |
|---|---|---|
| `modifier.updateModel(model, step, dt)` | **`plan(model, change)`** + **`apply(model, plan)`** | separates deciding (reads solution state) from doing (does not) |
| `modifier.setRestartData(...)` | *deleted* | replay calls the same `apply` |
| `modifier.getRestartData()` | **`model.topologyHistory`** | one ordered record of plans, not a private history per modifier |
| the modifier sweep | **`model.updateTopology(step, timeStep)`** | "model update" is taken |
| — | **round**, **fixed point** | "iteration" collides with Newton, "sweep" with multigrid |
| — | **`with model.topologyChanges():`** | outside the window, create/delete raises |
| `max(model.elements.keys())+1` | **`model.reserveElementNumbers(n)`** | "reserve" states the invariant: never recycled |
| `MeshDependent.reconcileIfChanged` | **`refreshIfMeshChanged`** | plainer for a first-time reader |
| `MeshDependent.reconcile` | **`refresh(model, change)`** | same |
| the constraint sweep | **`model.refreshMeshDependents()`** | names the phase after who runs in it |
| `ModelChangeObserver`, `registerObserver`, `onModelChanged` | *deleted* | one notification mechanism |
| — | **`model.topologyFingerprint()`** | the hash that proves a replay matches |

Teaching mnemonic: **modifiers plan and apply; mesh-dependents refresh; then we solve.**

## 2.3 Phase 1 — the pipeline

```python
def updateTopology(self, step, timeStep) -> bool:
    """Run every model modifier to a fixed point. Returns True if the topology changed."""
    changed = False
    with self.topologyChanges():
        lastPlannedVersion = {name: -1 for name in self.modelModifiers}
        for roundNumber in itertools.count(1):
            progress = False
            for name, modifier in self.modelModifiers.items():     # parse order — deterministic
                change = self.changesSince(lastPlannedVersion[name])
                plan = modifier.plan(self, change)                 # reads solution state
                lastPlannedVersion[name] = self.topologyVersion
                if plan is None:
                    continue
                modelChange = modifier.apply(self, plan)           # mutates; reads no solution state
                self.recordTopologyChange(roundNumber, name, plan, modelChange)
                progress = changed = True
            if not progress:
                break
            if roundNumber >= self.maxTopologyRounds:
                raise TopologyPipelineError(self.topologyHistory.summarizeCurrentUpdate())
    return changed
```

**Why rounds rather than a declared dependency DAG.** A DAG requires users to declare dependencies
correctly and cannot express mutual dependence: AMR refines what the printer deposited; the
printer's next layer must connect to what AMR refined; AMR's 2:1 balance may need to refine what
another modifier just activated. Rounds handle that natively and terminate when nobody has work
left. Determinism is preserved because the *within-round* order is fixed.

**Termination rule (a modifier contract).** `plan` must return `None` when the change since its own
last plan does not touch its domain. `change is None` on the first round of an update means
"evaluate freshly". A modifier that keeps planning on its own output does not converge, and the
`maxTopologyRounds` guard turns that into a loud error carrying the history, not a hang.

**Typical round count is 2.** Round 1: AMR refines, the facet modifier rebuilds the affected
surfaces. Round 2: AMR sees only facet changes (not its domain) → `None`; the facet modifier sees
no surface change → `None`; fixed point.

**Inter-modifier communication uses the same pull-by-version mechanism as consumers**
(`model.changesSince`). One mechanism, two granularities: modifiers pull *between rounds*, consumers
pull *once, after the window closes*.

## 2.4 The keystone — `plan` / `apply`

```python
class ModelModifierBase(ABC):

    @abstractmethod
    def plan(self, model, change) -> "Plan | None":
        """Decide what to do. MAY read solution state (markers, fields, time, step).
        Returns a serializable Plan, or None if there is nothing to do.

        `change` is the net ModelChange since this modifier last planned within the current
        topology update, or None on the first round. Return None when it does not touch this
        modifier's domain -- that is what makes the pipeline converge.

        NEVER called during a restart replay: the plan is read from the recorded history instead.
        """

    @abstractmethod
    def apply(self, model, plan) -> ModelChange:
        """Carry out `plan`, mutating the model. MUST NOT read solution state -- a pure function of
        (model, plan). This is the single code path a live run and a restart replay share, and the
        reason the replayed mesh is numbered identically."""

    @abstractmethod
    def encodePlan(self, plan) -> dict[str, np.ndarray]:
        """Serialize a Plan for the checkpoint."""

    @abstractmethod
    def decodePlan(self, data: dict[str, np.ndarray]) -> "Plan":
        """Inverse of encodePlan."""

    def declaredDomain(self, model) -> set:
        """The element numbers this modifier claims exclusive authority over, checked for overlap
        against every other modifier at setup (see §2.9)."""
        return set()
```

Live: `plan` → `apply`, and the plan is recorded. Replay: read the plan from the history → **the
same `apply`**. `setRestartData` ceases to exist as a concept; there is no second implementation to
keep in step, and every future modifier inherits the guarantee without arguing for it.

```
        LIVE RUN                                      RESTART REPLAY
   modifier.plan(model, change)                    read recorded plan
   reads markers, state, time                      no markers re-evaluated
        │        │                                      │            │
        │        └── records ──▶ topologyHistory ── restores ────────┘
        │                                                            │
        └──────────── applies ──▶┌────────────────────────┐◀── applies┘
                                 │ modifier.apply(model,  │
                                 │              plan)     │
                                 │ ONE code path          │
                                 │ reads no solution state│
                                 └────────────────────────┘
                                             │
                   identical element numbers · identical topology
```

**Plans are data.** A `Plan` is a small dataclass of arrays/scalars — for AMR, the eids to refine;
for the printer, the layer index and the elements to activate; for the facet modifier, the recipe
names to rebuild. It must contain everything `apply` needs, because `apply` may not consult
solution state.

## 2.5 The mutation window

```python
@contextmanager
def topologyChanges(self):
    """The only scope in which elements and nodes may be created or deleted."""
    self._topologyOpen = True
    try:
        yield
    finally:
        self._topologyOpen = False

def createElement(self, element, ...):
    if not self._topologyOpen:
        raise TopologyError(
            f"element {element.elNumber} was created outside a topology update. Only model "
            "modifiers may create or delete entities, inside model.updateTopology() -- see "
            "doc/.../topologypipeline.rst"
        )
```

Setup is exempt: the window is opened once around model construction, then again per increment.
This is what makes "only modifiers mutate" an enforced property rather than a convention, and it is
what allows §2.6's allocator to guarantee determinism.

## 2.6 Element number allocation

```python
def reserveElementNumbers(self, count: int = 1) -> range:
    """Reserve `count` fresh element numbers. Monotonic: never recycled, never derived from
    max(self.elements). Callable only inside a topology window."""
    first = self._nextElementNumber
    self._nextElementNumber += count
    return range(first, self._nextElementNumber)
```

`_nextElementNumber` is initialised **once**, at the end of setup, to `max(elements)+1`, and never
recomputed from the dict thereafter. Consequences:

- numbering is a pure function of the **ordered creation sequence** alone; with `max()+1` it was a
  function of creations *and deletions and their interleaving*, all of which the replay would have
  had to reproduce as well;
- a number never refers to two different elements over the run's lifetime, so `ModelChange`'s
  documented no-reuse assumption (`modelchange.py:85`) becomes true rather than aspirational, and a
  cached-by-number reference cannot silently alias;
- the live clash hazard patched by hand at `hadaptivity.py:509-517` disappears structurally; that
  defensive resync and its comment are deleted.

**Growth.** Facets are re-minted on every rebuild, so consumption is roughly
`n_facets × n_rebuilds` — order 10⁵–10⁷ on a long AnchorPryOut run against the int32 Ensight ceiling
of 2.1e9. Comfortable. If it ever threatened, the answer is to stop deleting and re-creating
unchanged facets (§7 P6), never to recycle numbers.

## 2.7 Phase 2 — refresh, pull only

After the window closes, every mesh-dependent consumer catches up **once**, from the net change
across all rounds:

```python
def refreshMeshDependents(self) -> bool:
    """Let every mesh-dependent consumer catch up. Returns True if any reported that its DOF
    footprint changed (which forces an equation-system rebuild)."""
    return any([d.refreshIfMeshChanged(self) for d in self.meshDependents])   # materialise: no short-circuit
```

**Why pull and not push.** A consumer needs the *net* effect of the pipeline, and that is only
knowable once the pipeline has converged. Push fires per mutation — i.e. at moments which are by
construction *mid*-pipeline. Round 3 rebuilds what round 2 invalidated; a consumer notified in
round 1 is handed a state that no longer exists when the solve begins, and would have to redo its
work in every subsequent round. Push is not merely riskier here, it is at the wrong granularity: it
reports transients, consumers care about the settled model.

Because consumers are pure readers (§2.5 forbids them to mutate), **their order is irrelevant and no
fixed-point iteration is needed in phase 2.**

`ModelChange.mergedWith`/`coalesce` therefore becomes **load-bearing**, not vestigial: computing the
net change across rounds — dropping entities created and destroyed within the same update — is
exactly what a consumer must see. Its transient-dropping semantics should be documented as
intentional.

**Three tiers of "the mesh changed", deliberately kept:**

| tier | mechanism | use when | replay behaviour |
|---|---|---|---|
| 1 | `_checkSetChanged(theSet)` (`constraintbase.py:208`, `stepactionbase.py:207`; 8 call sites) | you only need "did my set grow" | checks at point of use → **no timing dependency at all**, faithful for free |
| 2 | `MeshDependent.refreshIfMeshChanged` + `changesSince` | you need the changeset, or cache derived geometry | faithful because phase 2 runs once per applied update |
| 3 | ~~push observers~~ | — | *deleted* |

Tier 1 is not a wart. It is the cheapest correct answer for the majority of consumers and it is the
only one with no timing dependency whatsoever.

## 2.8 Determinism rules

Each is checkable, and each is pinned by a test in §8.

1. **Modifier order** = `model.modelModifiers` insertion order = parse order. Never a `set`, never
   an unordered map.
2. **Sorted iteration wherever entities are created.** `newChildEids` (`hadaptivity.py:545`) is a
   `set` and numbers are assigned in its iteration order → `sorted(...)`. Audit every `set` on a
   creation path.
3. **One monotonic allocator**, usable only inside the window (§2.5, §2.6).
4. **`apply` reads no solution state.** Enforced by review; violations surface immediately as
   fingerprint mismatches (§2.10).
5. **Round boundaries are recorded**, not inferred.
6. **Consumers never mutate.** Enforced by the window (§2.5).

## 2.9 Robustness

**Setup-time domain conflicts.** `hadaptivity` already refuses two AMR modifiers claiming
overlapping `refineElSet` (constructor, ~lines 300-318). Promote that to `ModelModifierBase` via
`declaredDomain(model)`: pairwise overlap across all modifiers is a hard error at construction, with
a message naming both modifiers and an example element.

**Per-round conflicts.** The pipeline tracks entities touched in the current round and by whom. Two
modifiers mutating the same entity in one round, or B deleting what A created in the same round, is
an error carrying both names — not a race to be reasoned about after the fact.

**Non-convergence.** `maxTopologyRounds` (default 16) with an error that dumps the current update's
history: which modifier kept reporting progress, in which rounds, and what each plan contained.

## 2.10 The topology history and the fingerprint

One structure, three jobs — this is where the design earns its keep.

```python
@dataclass
class TopologyRecord:
    increment: int
    roundNumber: int
    modifier: str
    plan: dict[str, np.ndarray]     # from modifier.encodePlan
    # summary fields, for logging and forensics only:
    nElementsAdded: int
    nElementsRemoved: int
    nNodesAdded: int
    elementNumberRange: tuple[int, int]
    touchedSurfaces: tuple[str, ...]
    wallTime: float
```

1. **Human log.** Emitted through `Journal` at a verbosity level:
   `round 2 · amr_concrete · refined 48 -> 384 elements #12043-12426 · 0.42 s`.
2. **The restart history.** The same records, serialized. Not a debugging add-on sitting beside the
   authoritative record — it *is* the authoritative record.
3. **Per-round fingerprint.** `model.topologyFingerprint()` =
   `blake2b(sorted (elNumber, elType, sorted node labels))`, plus the node-coordinate hash.
   Recorded per round, it converts "the resumed run diverged somewhere" into **"increment 471,
   round 2, modifier `amr_concrete`"** — a bisectable location instead of a hunt. Cheap enough to
   leave enabled in CI, and the direct answer to the concern that motivated this whole plan.

---

# 3. Interfaces

## 3.1 `FEModel` additions

```python
# state
self.modelModifiers: dict[str, ModelModifierBase]   # existing; parse order matters
self.meshDependents: list[MeshDependent]            # new registry (was: model.constraints only)
self.topologyHistory: TopologyHistory               # new
self.maxTopologyRounds: int = 16                    # new
self._nextElementNumber: int                        # new
self._topologyOpen: bool = False                    # new

# API
def topologyChanges(self) -> ContextManager         # §2.5
def createElement(self, element) / removeElement(self, elNumber)   # window-guarded
def reserveElementNumbers(self, count=1) -> range   # §2.6
def updateTopology(self, step, timeStep) -> bool    # §2.3
def refreshMeshDependents(self) -> bool             # §2.7
def recordTopologyChange(self, roundNumber, name, plan, modelChange) -> None
def topologyFingerprint(self) -> bytes              # §2.10
def registerMeshDependent(self, consumer) -> None
```

Removed: `registerObserver`, `unregisterObserver`, `_modelChangeObservers`,
`notifyModelChanged`'s observer cascade (the version bump and change-log append survive inside
`recordTopologyChange`).

## 3.2 The solver loop

`solvers/nonlinearimplicitstatic.py:260-265` becomes:

```python
modelHasChanged = model.updateTopology(step, timeStep)          # phase 1, rounds to fixed point
connectivityHasChanged = model.refreshMeshDependents()          # phase 2, one sweep
if modelHasChanged or connectivityHasChanged or self.theDofManager is None:
    ...                                                          # phase 3, unchanged
```

Mirror the same two calls in the explicit/dynamic/arclength solvers (`nonlinearexplicitstatic`,
`nonlinearexplicitdynamic`, `nonlinearimplicitstaticparallelarclength`), which today either skip the
modifier sweep or duplicate it.

## 3.3 `MeshDependent`

```python
class MeshDependent(ABC):
    _lastSeenTopologyVersion = 0

    @abstractmethod
    def refresh(self, model, change) -> bool:
        """Patch cached mesh-derived state for `change` (the net ModelChange since this consumer
        last looked). MUST NOT create or delete entities -- that is a model modifier's job, and
        the topology window is closed here. Returns True if this consumer's DOF footprint changed."""

    def refreshIfMeshChanged(self, model) -> bool:
        if model.topologyVersion == self._lastSeenTopologyVersion:
            return False
        change = model.changesSince(self._lastSeenTopologyVersion)
        self._lastSeenTopologyVersion = model.topologyVersion
        return change is not None and self.refresh(model, change)
```

---

# 4. Restart

## 4.1 Checkpoint layout

```
/                        attrs["restartFormatVersion"] = 2
/topologyHistory         attrs["nextElementNumber"], attrs["count"]
    /000000              attrs: increment, round, modifier, fingerprint
                         datasets: <plan arrays, from modifier.encodePlan>
    /000001              ...
/elements/<elNumber>     state vars -- keyed by NUMBER again; this is the payoff
/nodeFields/...          unchanged
/constraints/...         unchanged
/modelModifiers/         GONE (no per-modifier private history, no element state)
```

## 4.2 Replay

```python
def replayTopologyHistory(self, records):
    with self.topologyChanges():
        for record in records:                                   # recorded order, globally ordered
            modifier = self.modelModifiers[record.modifier]
            plan = modifier.decodePlan(record.plan)
            modelChange = modifier.apply(self, plan)             # THE SAME apply
            self.recordTopologyChange(record.roundNumber, record.modifier, plan, modelChange)
            if self.verifyFingerprints:
                assert self.topologyFingerprint() == record.fingerprint, (
                    f"replay diverged at increment {record.increment}, round {record.roundNumber}, "
                    f"modifier {record.modifier}")
    self.refreshMeshDependents()
```

Then `readRestart` restores element state **by element number**, in one uniform loop, with no skip
set, no `getattr`, and no exception swallowing — because the numbers are now faithful.

## 4.3 No legacy support (user directive, 2026-08-16)

Neither AMR nor restart has been merged into `next_v26.11`, so no user can hold an old checkpoint.
Shipped code reads and writes **one** format and **refuses** any other `restartFormatVersion` with a
"regenerate this checkpoint" message.

The single concession is a **throwaway converter for the validation window**: the pinned benchmark
checkpoints (`restart_ckpt_470_*.h5`, `LATE_incr4xx.h5`) came out of a 1 d 20 h run, back the
in-flight Inv 5+6 A/B, and are the launchpad for L3 verification (§0.4).

It is a *converter*, not a second reader: the old `occasionEids` are already a plan sequence, so it
synthesizes a `topologyHistory` from them (§0.4) and everything downstream is the normal path.
Roughly 40 lines.

- one commit, **last** in the sequence: `TEMP: convert pre-pipeline checkpoints during validation —
  REVERT BEFORE MERGE`, containing the converter and nothing else;
- **reverted, not edited away**, once the A/B closes or fresh checkpoints exist at a comparable
  increment;
- deliberately **not covered by a test**, so nothing pins it in place;
- the branch does not merge while it is present. Track it in `PLAN_BLOCKAMG_EFFICIENCY.md` §Open.

Sequence consciously: reverting the bridge kills the pinned checkpoints. Either finish the A/B
first, or budget one run to regenerate a checkpoint at ~increment 470.

---

# 5. Migration, component by component

## 5.1 `modelmodifiers/adaptivity/hadaptivity.py`

| today | becomes |
|---|---|
| `updateModel` (lines 394-451): markers → eligible → `_refineAndMaterialize` | `plan(model, change)` → `RefinementPlan(eids=...)`; returns `None` when `change` does not touch its domain or the cutback guard fires |
| `_refineAndMaterialize` (453-503) | `apply(model, plan)`: refine the octree at `plan.eids`, `balance_2to1`, materialise **one occasion**, return the `ModelChange` |
| `_replayOccasionsByEid` (722-764) | **deleted** — replay calls `apply` per recorded plan |
| `setRestartData` (775-833) | **deleted** |
| `getRestartData` (672-720), incl. `stateEids`/`stateSizes`/`stateData` | **deleted**; plans are recorded centrally |
| `restoredElementLabels` publication (813) | **deleted** |
| `_replayMode` (524, 560-563) | **deleted** — there is no replay-specific path any more. State transfer and warm-start interpolation now run identically in both, which is *required*, not merely permitted: `apply` must be one function. (Their results are still overwritten by the state restore; that is a small, honest cost — see §10.) |
| `_isFirstCall` hack (794) | **deleted** |
| `_nextElLabel` + resync (323, 517) and the 8-line comment at 509-516 | **deleted** → `model.reserveElementNumbers` |
| `for eid in newChildEids` (545, 552) | `for eid in sorted(newChildEids)` |
| `model.elements[child.elNumber] = child` (578) | `model.createElement(child)` |
| parent removal (598-601) | `model.removeElement(el.elNumber)` |
| `_committedOccasions` / `_committedOccasionEids` / `_pendingMarkedElements` restore by label (815) | **deleted** — the history holds the plans; pending marks are part of the modifier's own in-memory state and are re-derived by the next `plan` |
| the constructor's overlap check (~300-318) | generalised to `declaredDomain` on the base class |

`_pendingMarkedElements` deserves a note: it exists so refinement waits until `minMarkedElements` is
reached. Under the new interface that is state internal to `plan`, and it never needs checkpointing
— the marks are re-derived at the next `plan` from the restored solution state, which is exactly
what the live run would have done.

## 5.2 `generators/surfaceelementgenerator.py` → a facet modifier

Facet generation was *already* a generator (`*modelGenerator, generator=surfaceElementGenerator`,
`examples/AnchorPryOut/test.inp:227`) with the recipe recorded in `model.contactFacetRecipes`
(`femodel.py:76`, written at `surfaceelementgenerator.py:312`) precisely so it can be re-run.
Runtime regeneration got bolted onto the constraints because `reconcile` was the convenient hook.
Promoting it to a modifier finishes the design that is already there.

- new `modelmodifiers/surfacefacets/` modifier, **implicitly created** by `FEModel` when the first
  recipe is registered, and **always ordered last** among modifiers. *Rejected alternative:* making
  users declare it in the `.inp` — redundant with the recipe they already declare, and easy to
  forget.
- `plan(model, change)` → `FacetPlan(recipeNames=(...))` for every recipe whose surface the change
  touched; `None` otherwise.
- `apply(model, plan)` → today's `buildContactFacets` body, minting via
  `model.reserveElementNumbers` and publishing through `model.createElement` / `removeElement`.
- `tie.py:402-404` and `nodetodeformablesurfacepenalty.py:548-551` **drop their
  `buildContactFacets` calls entirely** and become pure readers of
  `model.elementSets[facetSetName]`.

## 5.3 `constraints/tie.py` — push → pull

- delete `model.registerObserver(self)` (line 257) and `onModelChanged` (273-275);
- `reconcile` (388-410) → `refresh(model, change)`, minus the two `buildContactFacets` calls;
- register as a `MeshDependent` instead.

Ordering note: tie drops records for slave nodes already claimed by AMR's hanging-node MPC. That
dependency is satisfied by *phase separation* (AMR sets the hanging records inside phase 1), not by
consumer ordering — which is why phase 2 needs no ordering at all.

## 5.4 The three penalty constraints

`nodetodeformablesurfacepenalty.py`, `nodetorigidsurfacepenalty.py`,
`nodetodiscreterigidbodypenalty.py`: keep their pull tick, rename `reconcile`→`refresh` and
`reconcileIfChanged`→`refreshIfMeshChanged`, and **remove their facet minting** (§5.2). They call
`model.registerMeshDependent(self)` instead of relying on being in `model.constraints`.

## 5.5 `rigidbodies/discreterigidbody.py:84-88`

Setup-only, so not a correctness problem — but it is the last `max(model.elements.keys())+1` outside
the setup generators. Route it through `reserveElementNumbers` inside the setup window.

## 5.6 Unchanged

Base-mesh generators (`boxgen`, `pipegen`, `planerectquad`, `cuboidlatticegenerator`,
`microstructuregenerator`, `abqmodelconstructor`) need no `plan`/`apply` and no recorded history:
their element numbering is a function of the input file, the reconstruct phase re-runs them
identically before `readRestart`, and nothing renumbers survivors.

~~They mint inside the setup window; that is enough.~~ **Disproved by the suites, 2026-08-17.**
They place elements *directly* into `model.elements` without the allocator, and the ones declared
`executeAfterManualGeneration=True` run *after* setup's adoption pass — so the allocator's
high-water mark was stale when `surfaceElementGenerator` reserved facet numbers, and every tie /
deformable-contact case failed with "element number 1 is already taken" (10 cases). Patched in
`1a2a05d8` by re-adopting at the two setup-time minting entry points (monotonic, so it can never
reissue a deleted facet's number).

**Follow-up, do this in P1 or early P2:** route the base generators through
`reserveElementNumbers` as well, and delete both transitional `adoptSetupElementNumbers()` calls.
Then "only the allocator hands out element numbers" holds without exception, which is what the rest
of the design assumes.

**But that argument was not free, and its failure mode is instructive (`a8bfda33`, §1.3 item 7).**
It holds for labels *parsed* from the input file. It did **not** hold for anything numbered by
walking a parsed container: element sets were built through a raw `set()` of identity-hashed
elements, so their member order came from object addresses, and `surfaceElementGenerator` numbers
contact facets by walking exactly those sets. Setup-time facet numbering was therefore already
irreproducible between a run and its own restart, before any AMR and before any of this plan.

Generalise the lesson rather than the fix: *"setup is reproducible because the same code runs on the
same input"* is an argument about **code**, and it fails silently wherever an unordered container
sits between the input and a number. Determinism rule 2 (§2.8) is the standing check for that, and
the per-round fingerprint (§2.10) is what makes a violation visible instead of latent.

The 8 tier-1 `_checkSetChanged` consumers are **not migrated** (§2.7).

---

# 6. Performance

Reverting `b0eb74f`'s batched replay costs the 2.04× (first solve 499 s → ~1017 s). That is paid
back — and then some — by fixing the cost at its source, which benefits the **live run too**.

`b0eb74f` collapsed 54 materialisations into 1, saving ~518 s: **~9.6 s per occasion**, for
occasions adding a few hundred elements. That ratio is the smell. Four whole-mesh rebuilds per
occasion, none of which scale with the size of the change:

| cost | where | why it is O(N_mesh) |
|---|---|---|
| hanging-node classification | `adaptivity/refinement.py:329` (`classify_hanging`) | the *scan* is spatially hashed and restricted to the finer-neighbour interface shell, but its *setup* — `nodeGrid` over every used node, `elemGrid` and `box[eid]` over every active element — is rebuilt from scratch every call. Matches the 7.55 s hotspot already recorded. |
| field-variable relinking | `hadaptivity.py:669` (`_linkFieldVariableObjects(model.nodeSets["all"])`) | whole-model relink per occasion |
| set membership sync | `hadaptivity.py:614-640` | `present = {n.label for n in model.nodeSets[setName].nodes}` per tracked set |
| facet rebuild + re-projection | `surfaceelementgenerator.py` + `tie._buildTiedRecords` | regenerates *every* facet of a touched surface and re-projects *every* slave node |

The fourth is the deferred item from `006490eb`'s own commit message: *"the O(refined-area) partial
facet rebuild via `ModelChange.faceMap` (currently a whole-surface rebuild on any touch)"*. The
machinery for it exists and is **written but never read**: `parentToChildren`, `faceMap`,
`addedElements`, `removedElements` and `childFacesOf()` have zero consumers today — every consumer
uses only `touchesSurface()`/`touchesElementSet()` and then rebuilds wholesale.

### P0 RESULTS (measured 2026-08-17 on xeon, `examples/AmrRoundProfile/profile_amr_round.py`)

Method: refine **one** element in meshes of growing size, so a stage that grows with the mesh is
doing whole-mesh work for a local change. Seconds per topology round, tie variant (`--tie 4 8 12`):

| stage | 128 el | 512 el | 1152 el | growth | verdict |
|---|---|---|---|---|---|
| `elements & state transfer` | 0.0482 | 0.0479 | 0.0485 | **1.0×** | **already change-proportional — leave alone** |
| **`refresh mesh dependents`** | 0.0054 | 0.0181 | **0.0394** | **7.3×** | **largest growing cost — attack first** |
| `hanging: interface-shell scan` | 0.0035 | 0.0071 | 0.0152 | 4.3× | grows |
| `fields resize & restore` | 0.0012 | 0.0037 | 0.0086 | 7.2× | grows |
| `hanging: whole-mesh index build` | 0.0012 | 0.0037 | 0.0078 | 6.5× | grows |
| `sets & fields sync` | 0.0007 | 0.0018 | 0.0036 | 5.1× | grows |
| `relink field variables` | 0.0003 | 0.0008 | 0.0017 | 5.7× | grows |

(9× element growth across the row. The plain no-tie variant gives the same picture with
`refresh mesh dependents` at zero, since nothing is registered.)

**Two corrections to the hypothesis above:**

1. **The consumer refresh dominates, not the hanging-node work.** At the largest size it is 2.6× the
   hanging scan and still growing fastest. That is the whole-surface facet rebuild plus the
   full slave re-projection — exactly what `ModelChange.faceMap` was designed to make incremental
   and what nothing reads. **P6 should start here**, and P3 (facet generation becomes a modifier)
   is the restructuring that makes it addressable.
2. **Within the hanging-node work, the *scan* grows faster than the index build** (4.3× vs 6.5× on a
   smaller base; the scan is 2× the absolute cost at every size). §6 above guessed the index build
   dominated. The reason is visible in the code: the scan does early-out per element, but it still
   iterates *every* active element to ask `hasFinerNeighbour(eid)` — the early-out is not free.

**P6 item 2 delivered (`ccf79a9`).** The scan now records the highest level per (grid cell, body)
during the grid pass and rejects an element before any set union unless some cell it touches holds a
strictly finer element of the same body -- a necessary condition, with the exact test unchanged
behind it. Measured on the same probe:

| stage @1152 el | before | after |
|---|---|---|
| `hanging: interface-shell scan` | 0.0152 | **0.0095** (−37%) |
| `hanging: whole-mesh index build` | 0.0078 | 0.0081 (+4%, the new bookkeeping) |
| hanging total | 0.0230 | **0.0176** (−23%) |
| `topology update` | 0.1153 | 0.1098 (−5%) |

The win grows with mesh size (−1% / −20% / −37% at 128 / 512 / 1152 elements), which is the
signature of skipping a conforming majority that grows with the mesh -- so it should be worth
considerably more at AnchorPryOut scale. Exactness was verified by loading the previous
implementation side by side from git and running it unbound on live meshes from real AMR runs:
`classify_hanging` output identical, including a tied two-body case with 44 hanging records.

**Still the dominant cost: `refresh mesh dependents` (0.0399 s, unchanged).** That is P6 item 1 and
it lives in `surfaceelementgenerator.py`/`tie.py` -- blocked behind PR #57 (§0.1a).

**One stage is already right:** `elements & state transfer` is flat to three digits across a 9×
mesh growth *and* across both variants — 0.048 s of fixed cost for one refined element. It needs no
work; it should be the model for the others.

Work item (parallel track, §7 P6): make the growing stages incremental from the `ModelChange`
payload, in the order the table gives.

---

# 7. Staging

| phase | content | depends on | verified at | independently valuable |
|---|---|---|---|---|
| **P0** | Profile one topology round: decompose the four costs of §6 on a live AnchorPryOut refinement. | — | **L3, xeon idle** | yes (gates P6) |
| **P1** | Monotonic allocator + mutation window + `createElement`/`removeElement`. Route `hadaptivity`, `surfaceelementgenerator`, `discreterigidbody` through them. Delete the `_nextElLabel` resync. | — | L1 + L2 | yes — fixes the live clash hazard |
| **P2** | `plan`/`apply` split on `ModelModifierBase` + `hadaptivity`; `model.updateTopology` rounds-to-fixed-point; solver loop. Restart still on the hotfix. | P1 | L1 + L2 | no |
| **P3** | Facet generation becomes a modifier; constraints stop minting. | P1, P2 | L1 + L2, then **L3** | yes — removes reentrancy |
| **P4** | Pull only: convert `tie`, add `model.meshDependents`, rename `reconcile*`→`refresh*`, delete `ModelChangeObserver` and the observer cascade. | P3 | L1 + L2 | yes |
| **P5** | `topologyHistory` + `topologyFingerprint`; restart replays plans through `apply`; **delete the hotfix** (`restoredElementLabels`, the `getattr` at `femodel.py:453`, the `NotImplementedError` guard, `setRestartData`, modifier-side state serialization). | P2, P4 | L1 + L2 + **the full L3 ladder of §0.4** | **this is where the bug is actually fixed** |
| **P6** | Incremental refresh: the four costs of §6, ranked by P0. | P0 (P3 for the facet one) | **L3, xeon idle** | yes — live-run AMR win |
| **P7** | `declaredDomain` overlap check, per-round conflict detection, `maxTopologyRounds`, history logging. | P2 | L1 | yes |
| **P8** | *Optional, separate PR:* facets out of `model.elements` — geometry-only no-ops (`contactsurfaceelement.py:175-233`) that are dead weight in the solver's element loop. Explicitly **not** required by this design. | — | L1 + L3 | yes |

P0 and the AMR-side items of P6 live on `perf/amr-topology-round` and merge first (§0.1); P1–P7 on
`feat/topology-pipeline`.

The hotfix stays in place and correct until P5 lands, then is deleted wholesale.

## Commit sequence

```
 1. feat(model): monotonic element-number allocator and topology mutation window   [P1] DONE 198fd511
 2. fix(topology): make entity creation order reproducible across runs            [P1] DONE a8bfda33
 3. refactor(topology): mint every runtime element number through the allocator   [P1] DONE f7e4e511
 4. fix(topology): re-adopt setup element numbers before minting facets           [P1] DONE 1a2a05d8
    ↳ P1 verified on xeon: pytest 283 passed / 5 pre-existing; edelweiss-only and marmot
      failure sets IDENTICAL to the baseline table above.
 5. refactor(generators): reserve base-mesh element numbers from the allocator    [P1] DONE b12600ab
    ↳ **P1 COMPLETE**, re-verified on xeon at b12600ab: pytest 283 passed / 5 pre-existing;
      edelweiss-only and marmot failure sets IDENTICAL to the baseline table above.
      `grep -rn "max(model.elements.keys()" edelweissfe/` -> nothing.
      Two adoptSetupElementNumbers() calls remain **by design**, at the two places where element
      numbers enter from outside the allocator: .inp user labels, and cuboidLatticeGenerator's
      wholesale replacement of model.elements. Both are monotonic and cannot reissue a number.
 3. feat(modifiers): split plan() from apply() on ModelModifierBase                      [P2]
 4. refactor(amr): plan/apply, sorted child iteration, one occasion per apply            [P2]
 5. feat(model): run model modifiers to a fixed point in updateTopology()                [P2]
 6. feat(topology): split modifiers into plan/apply, run to a fixed point         [P2] DONE 2ec278da
    ↳ **P2 COMPLETE**, verified on xeon: pytest 288 passed / 5 pre-existing; both suite failure
      sets IDENTICAL to the baseline. Restart still runs on the 1393803 hotfix until P5.
      Resolved open question: `plan` DOES take `step`/`timeStep` (forwarded, unused by hAdaptivity).
 7. refactor(topology): one pull-based refresh phase, retire the push observer    [P4] DONE 5d043f22
    ↳ **P4 COMPLETE**, verified on xeon: pytest 291 passed / 5 pre-existing; suites 2 + 3 failures,
      no regressions, and every remaining failure is explained by the table above.
      Load-bearing discovery: `tie` lives in `model.multiPointConstraints`, which **no**
      per-increment sweep iterates -- push was the only mechanism reaching it, so
      `registerMeshDependent` is what makes pull work at all, not ceremony.
 8. feat(modifiers): promote contact facet generation to a model modifier                [P3]
 7. refactor(constraints): tie and contact become pure readers of their facet sets       [P3]
 8. refactor(model): single pull-based refresh phase; delete ModelChangeObserver         [P4]
 9. feat(model): topologyFingerprint + the restart invariant it makes checkable   [P5] DONE ec26affc
10. feat(restart): replay recorded plans through apply(), retiring the hotfix     [P5] DONE 5850147a
    ↳ **P5 COMPLETE**, verified on xeon: pytest 297 passed / 5 pre-existing; suites 2 + 3, identical
      to the baseline. The 1393803 hotfix is gone in full -- `restoredElementLabels`, the `getattr`,
      the doubled element-state serialisation, `_replayMode`, `_isFirstCall`, `_replayOccasionsByEid`,
      the read-side `except NotImplementedError`, and both per-modifier restart hooks.
      **The original bug is fixed at its root**: replay and the live run now share one `apply()`.
11. perf(amr): skip the neighbour search away from a refinement front            [P6.2] DONE ccf79a9
12. perf(amr): incremental hanging-node classification / set sync / relink               [P6]
13. perf(contact): rebuild only the facets of changed faces, via ModelChange.faceMap     [P6]
14. feat(modifiers): domain overlap check, round conflict detection, convergence guard   [P7]
15. test(topology): fingerprint round-trip, determinism, conflict and convergence        [tests]
16. docs(topology): the pipeline, the contract, the three tiers                          [docs]
17. TEMP: convert pre-pipeline checkpoints during validation -- REVERT BEFORE MERGE      [bridge]
```

---

# 8. Tests

**New `tests/test_topology_pipeline.py`:**

1. `test_rounds_reach_fixed_point` — two toy modifiers, one reacting to the other; asserts the
   round count and the order of `apply` calls.
2. `test_non_convergence_raises_with_history` — a modifier that always plans; asserts
   `TopologyPipelineError` and that its message names the culprit.
3. `test_creation_outside_window_raises` — `model.createElement` after the window closes.
4. `test_consumer_may_not_mutate` — a `MeshDependent` that tries to create an element in `refresh`.
5. `test_element_numbers_never_recycled` — create, delete, create; the new element gets a fresh
   number.
6. `test_domain_overlap_rejected_at_setup` — two modifiers claiming the same elements.

**New `tests/test_topology_fingerprint.py`:**

7. `test_fingerprint_is_order_independent_of_dict_iteration` — same mesh built two ways → same hash.
8. `test_fingerprint_detects_renumbering` — swap two element numbers → different hash.

**Extended `tests/test_restart_integration.py`:**

9. **The headline test — replay is bit-faithful.** Run N increments with AMR + a tie continuously;
   restart at increment k; assert `topologyFingerprint()` is identical at every increment > k, and
   that element numbers, node labels, and element state match element-for-element. This is the
   invariant the whole plan exists to establish, and it *fails on today's code*.
10. `test_replay_uses_no_markers` — instrument `plan` to raise if called during replay.
11. `test_fingerprint_mismatch_localises` — corrupt one recorded plan; assert the error names the
    increment, round and modifier.
12. Delete `d8dd996`'s test — it pins the hotfix's contract (`restoredElementLabels` ⊇ managed
    elements) and goes with it.

**L2 — integration:**

13. `examples/AnchorPryOut/tie_restart_probe.py` (harness (c), `PLAN_BLOCKAMG_EFFICIENCY.md` §1)
    with `AMR_THRESHOLD`/`REFINE_SLAVE` set so refinement actually fires — **verify `amr_hanging`
    and element counts changed, or the run is vacuous** (§5 of that doc). Extend it to compare
    `topologyFingerprint()` per increment, not just records.
14. `run_tests_edelweissfe ./testfiles/marmot/` and `./testfiles/edelweiss-only/` — no reference
    changes expected; any diff is a bug in this work.

    **The baseline is not green — always compare against it, never against zero.** But read the
    next block first: **most of these failures are a property of the machine's Marmot build, not of
    this code.** (Measured on xeon, 2026-08-17, base `d8dd9963`, in worktrees with the 15 built
    `.so` *and* `built_extensions.log` copied in — omitting that file silently reports every Marmot
    extension as unavailable and turns 6 failures into 93):

    **Current baseline on xeon, after the MKL pin (`61125069`) — every entry explained:**

    | suite | failing | why |
    |---|---|---|
    | `edelweiss-only` (2) | `MeshPlot` | xeon has no `latex` |
    | | `NodeToDeformableSurfaceContactPullOut` | razor-edge facet tie; **fixed on PR #57** (`7815c011`), not here |
    | `marmot` (3) | `OutputManagers`, `IndirectDisplacementControl` | xeon has no `latex` |
    | | `AMR_ContactRefineShear` | **the one genuinely open failure**; not touched by PR #57 |
    | `pytest tests/` (5) | `test_inputlanguage_golden`, 3× `test_schemasurface`, `test_stepoptions` | goldens/greps belonging to the input-system branch |

    Historical, before `61125069`: `marmot` also showed `AMR_MinMarkedElements`,
    `AMR_MixedMeshRefine` and `AMR_RecoveryError`. Those were MKL AVX-512-vs-AVX2 dispatch, not
    code — see below.

    The four `AMR_*` cases run to completion and fail on `U` vs `U.ref`, i.e. stale references or
    genuine drift — **not** crashes. Logs preserved on xeon at `~/tp-suite-logs/`.

    ### The failures are mostly a stale-`libMarmot` artifact (measured 2026-08-17)

    Running the same 8 cases on **x9** with the identical EdelweissFE commit — and with the same
    `OMP_NUM_THREADS=4 PYTHON_GIL=0`, so it is not a threading effect:

    | case | xeon | x9 |
    |---|---|---|
    | `MeshPlot`, `OutputManagers`, `IndirectDisplacementControl`, `AMR_MinMarkedElements`, `AMR_MixedMeshRefine`, `AMR_RecoveryError` | FAIL | **PASS** |
    | `NodeToDeformableSurfaceContactPullOut`, `AMR_ContactRefineShear` | FAIL | FAIL |

    Cause: the two machines run **different `libMarmot` builds**, and *neither matches its own repo*.

    | | Marmot repo HEAD | installed `libMarmot.so.1.0.0` | contains |
    |---|---|---|---|
    | x9 | `58d02d0e` (2026-08-12 18:12) | **built 2026-07-25** | up to `7db16d99` (Jun 21) |
    | xeon | `ffef6995` (2026-08-12 18:05) | **built 2026-08-12 17:31** | one commit more |

    The tips are the *same* commit message on the same branch with different hashes — an amended /
    rebased `fix(mechanicscore): guard haighWestergaard() rho derivative singularity at J2=0`, and
    neither machine's repo knows the other's hash. That guard is the only Marmot change between the
    two builds and is therefore the prime suspect for the drift, though that is not yet proven.

    **So "it passes on x9" means "x9 links a 2.5-week-old Marmot that the `U.ref` files were
    generated against", not "x9 is healthy".** Consequences:

    - the two genuinely code-level failures are only `NodeToDeformableSurfaceContactPullOut` and
      `AMR_ContactRefineShear` — they fail on both machines;
    - **the MPC-condensation regression hypotheses below are refuted** for `AMR_MixedMeshRefine` and
      `AMR_RecoveryError`: identical EdelweissFE code passes on x9, so the variable is Marmot;
    - the A/B comparisons in this plan remain valid (both arms ran in the same environment), but the
      label "pre-existing failures at `d8dd9963`" should read **"failures under xeon's Aug-12 Marmot
      build"**;
    - **production consequence:** AnchorPryOut runs on xeon against the newer Marmot while the
      references were generated against the older one. Someone must decide whether that drift is a
      Marmot regression or an intended change requiring reference regeneration.
    - **Rebuild both machines' Marmot from a known commit before trusting any suite result again.**

    **Triage (2026-08-17, git forensics only, written BEFORE the measurement above — the first two
    verdicts are now refuted; kept for the method, not the conclusions):**
    Independently corroborated: commit `cabfc869` (Jul 29) states its suites showed "only the same
    pre-existing failures (1 deformable-surface-contact, 7 AMR) as an unmodified checkout", so these
    were red well before any pipeline work.

    | case | verdict | evidence |
    |---|---|---|
    | `AMR_MixedMeshRefine` | REGRESSION, medium confidence | `U.ref` regenerated `fd04fb19` (Jul 28), *before* `0f530baa` (Aug 5) "fix two bugs the exactness assertion caught" in MPC condensation (`-0` slice with zero slave DOFs; CSR@CSR zero-cancellation with opposite-signed weights) and `00529d86` (Aug 6) making the rewritten condensation the default. A mixed hex/tet hanging-node pattern is exactly what hits those edge cases. |
    | `AMR_RecoveryError` | REGRESSION, medium confidence | same mechanism; `U.ref` from `85380760` (Aug 1), also pre-`0f530baa`. Sibling `AMR_RecoveryErrorSPR` differs only by `recovery=spr` and **passes** — consistent with a topology-dependent condensation bug rather than marker logic. |
    | `AMR_ContactRefineShear` | REGRESSION, low confidence | oldest `U.ref` (`65c767f1`, Jul 27); the same timeline argument applies but many other things changed meanwhile. Sibling contact/tie cases with identical option-migration patterns pass, so the two later `test.inp` edits are probably not the cause. |
    | `AMR_MinMarkedElements` | **UNKNOWN** | `U.ref`/`test.inp` landed together in `1137d5db` (Aug 2); `git diff 1137d5db..d8dd9963 -- hadaptivity.py` shows *no* functional change to the `minMarkedElements` path. By the test's own comment refinement never fires (1 of 2 elements crosses the threshold vs `minMarkedElements=2`), so **no hanging-node MPCs are ever created** — which rules out the condensation hypothesis above. A plain 2-element AT2PhaseField/GC3D20R solve; cause may lie in Marmot or in nondeterminism, not in this repo. |

    Next diagnostic (needs xeon): rerun `AMR_MixedMeshRefine` and `AMR_RecoveryError` at `0f530baa`'s
    parent vs. itself with `EDELWEISS_MPC_ASSERT_EXACT=1` — that confirms or kills the condensation
    hypothesis directly. For `AMR_MinMarkedElements`, rerun twice to test determinism, then rerun at
    `1137d5db` to establish whether it *ever* passed.
15. A new end-to-end case with **two** modifiers active (AMR + a stub deposition modifier) to
    exercise the multi-modifier interleaving that per-modifier histories cannot express, and to
    force a genuine multi-round fixed point.

**L3 — production scale (xeon, idle; §0.2 rules apply):**

16. **Converted-checkpoint health run** (available from P3 on). Resume `restart_ckpt_470_*.h5`
    through the converter (§0.4) and run increments 471-472. Pass = **0 cutbacks, 0 return-mapping
    failures**, per-solve iteration counts matching the uninterrupted run. Element numbers will
    differ from the original live run — expected, see §0.4.
17. **Production-scale faithfulness A/B** (P5, once). Mint new-format checkpoints from run 16, then
    arm A = continuous *k → k+3*, arm B = resume at *k* → *k+3*; assert per-increment fingerprints
    identical and element state identical element-for-element. This is test 9 at production scale;
    budget hours, not days.
18. **P6 timing**: per-round topology cost before/after, same increment, machine idle,
    `OMP_NUM_THREADS=16`.

## Acceptance criteria

- **test 9 (L1/L2) and test 17 (L3) pass**: a resumed run is fingerprint-identical to the continuous
  one, per increment, at both scales;
- test 16: resume at increment 470 of AnchorPryOut gives 0 cutbacks, 0 return-mapping failures, and
  per-solve iteration counts matching the uninterrupted run;
- `grep -n "getattr\|hasattr" edelweissfe/models/femodel.py` → nothing;
- `grep -rn "max(model.elements.keys()" edelweissfe/` → only setup-time base generators (or nothing);
- `grep -rn "registerObserver\|onModelChanged" edelweissfe/` → nothing;
- per-round topology cost reduced from ~9.6 s (P6 target set by P0), measured on xeon idle;
- net first-solve time on a deep resume no worse than `b0eb74f`'s 499 s despite the batching revert
  — if P6 cannot reach that, say so explicitly rather than quietly accepting the regression;
- the converter commit is reverted before the branch merges.

---

# 9. Documentation (required for merge, per CLAUDE.md)

- **new** `doc/source/documentation/topologypipeline.rst` — the three phases with the §2.1 diagram,
  the `plan`/`apply` contract, the window, the three tiers of change notification, and the worked
  example of writing a new modifier. This is teaching material: it is the page a student reads to
  understand how an adaptive analysis mutates itself.
- `doc/source/documentation/modelmodifiers.rst` — the new base-class interface; modifiers no longer
  serialize element state, and `setRestartData` is gone.
- `PLAN_RESTART.md` — checkpoint layout, the single-format policy, replay via recorded plans.
- `ELID_UNIQUENESS.md` — record that none of its three patterns was adopted, and why (§1.4): the
  problem was not element identity but replay fidelity.
- `PLAN_AMR_CONTACT.md` — close out the deferred P2/P5.1 (partial facet rebuild) and P3 (push→pull)
  items; both land here.
- `PLAN_BLOCKAMG_EFFICIENCY.md` — note the `b0eb74f` revert, the P6 replacement, and the bridge
  as a merge gate.

---

# 10. Risks and open questions

| risk | mitigation |
|---|---|
| **Replay fidelity is a strong claim.** Any remaining nondeterminism (an unsorted `set`, a dict keyed by object identity, a modifier reading solution state in `apply`) breaks it. | The fingerprint is not decoration — it is the enforcement. Recorded per round, checked in CI (test 9) and available at runtime via `verifyFingerprints`. Failures localise to one modifier in one round. |
| Reverting the batched replay costs 2.04× until P6 lands. | P6 targets the same cost in the live run too. If P6 underdelivers, batching can return *after* P3 — with facets no longer minting from the shared counter and numbering a pure function of the plan sequence, batched materialisation could be made numbering-neutral. Not assumed; measured. |
| `_replayMode`'s dead-work skip is deleted, so replay now pays state transfer + warm-start interpolation whose results are overwritten. | Honest cost of "one code path". Measure it in P2; if material, the *plan* can carry a "state will be restored" flag — but that flag must then be part of the recorded plan, not an ambient replay mode. |
| A modifier that never converges hangs the increment. | `maxTopologyRounds` + the history dump (§2.9). |
| Consumer ordering assumptions hidden in today's push order. | Phase separation removes the class; the tie/hanging-MPC case (§5.3) is the one known instance and is satisfied structurally. Test 4 pins that consumers cannot mutate. |
| Plans must be serializable; a future modifier may want to record something awkward. | `encodePlan`/`decodePlan` are per-modifier and array-shaped, matching the existing `getRestartData` convention. |
| Two modifiers legitimately need to interleave *within* a round. | Not supported by design: a round is a pass in declared order. If a real case appears, the answer is more rounds, not intra-round interleaving. |

**Open question (decide in P2).** Should `plan` receive the `step`/`timeStep` explicitly, as
`updateModel` does today? Leaning yes — markers may be rate- or time-dependent — but it must be
clear that these are `plan` inputs only, never `apply` inputs.

**Open question (decide in P3).** Whether the facet modifier records plans at all: its output is a
deterministic function of the recipes plus the current mesh, so it *could* be re-derived rather than
replayed. Recording is safer and uniform; re-deriving is less data. Recommendation: record, for
uniformity — the volume is trivial (a list of recipe names).

---

# 11. Out of scope

- **Coarsening.** No new questions until it exists — but note that a re-refined parent must receive
  its original eid, or a recorded plan referencing that eid replays onto a different element. Flag
  it for whoever implements it.
- **Node identity.** Node labels are coordinate-keyed and reproducible; nothing is broken.
- **P8** (facets out of `model.elements`).
- **Bit-identical *solutions* across restart** (`PLAN_BLOCKAMG_EFFICIENCY.md` §Open item 3) — that
  residual is solver-side (fresh Eisenstat–Walker/AMG/Krylov state) and unrelated to topology.
- **Threading.** The window makes a future lock trivial to place, but concurrent modifiers are not
  in scope.
