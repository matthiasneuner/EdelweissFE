# Restart / checkpoint support for EdelweissFE

Branch: `feat/restart` (split off `feat/amr-recovery-marker`)

## Status (v1)

**P0-P4 all done.** Corrections/findings made along the way, relative to this plan's original text:

- Decision #2 was wrong for pure-Python elements: only `marmotelement.pyx` had `getStateVars`/
  `setStateVars` before this branch. Added them to `DisplacementElement`, `DisplacementTLElement`
  (also serializing `_Eold`, the converged Green-Lagrange strain the TL formulation's incremental
  strain depends on) and `MarmotMaterialWrappingElement`.
- P0 also added `ConstraintBase.getRestartData`/`setRestartData` (default `None`, i.e. stateless)
  and implemented it for `NodeToDeformableSurfacePenaltyConstraint`'s frictional-force/
  augmented-Lagrange-multiplier history (`_tangentialForceConverged`/`_lambdaN` only --
  `_assignedFacetIdx`/`_gapCurrent` are recomputed every increment from restored node positions, so
  they carry no history of their own).
- P1's "register in `inputfileparser.py`'s `_DISPATCH_CATEGORY_BY_KEYWORD`" is stale: that
  mechanism is for name-dispatched sub-keywords (`*output, type=...`) only. A structural keyword
  like `*restart` registers in `config/registry.py`'s `"keyword"` category instead (see `*job`).
  This repo's golden-surface gate (`tests/golden/inputlanguage_surface.txt`,
  `PLAN_INPUT_SYSTEM_UNIFICATION.md`) had to be extended alongside the new keyword and the new
  `*output, type=restart` output manager.
- P2's per-step timestepper handoff does **not** skip constructing steps before the resumed one --
  it constructs every step normally (so `StepAction`s still accumulate/update exactly as an
  uninterrupted run would) and only skips calling `.solve()` on them, which is strictly better than
  the plan's original "skip prior steps entirely" and avoids that limitation.
- `AdaptiveTimeStepper`/`SimpleTimeStepper`'s `writeRestart`/`readRestart` were narrowed to the step's
  *dynamic* progress state only (`currentTime`, `finishedStepProgress`, increment counters, ...),
  deliberately excluding *configuration* (`stepLength`, `maxIncrement`, `maxNumberIncrements`, ...) --
  the resumed step's own construction (from the `.inp` used to resume) already supplies that,
  consistent with `FEModel.readRestart` only overwriting converged state, never structural
  definition. The original unwired scaffold wrote/read both together, which would silently
  re-clobber e.g. a raised `maxNumberIncrements` on resume.
- Verified end-to-end (not just unit-level): `tests/test_restart_integration.py` runs a real
  Marmot-backed `VonMises` job three times (uninterrupted / truncated-via-low-`maxNumInc` /
  resumed) through the actual `.inp`/parser/driver/solver stack and diffs the final `U` against the
  uninterrupted reference (matches to ~1e-17).
- P5 (fallback-on-failure) remains out of scope, per the plan's own stretch-goal framing.

## Why now

EdelweissMeshfree already has a working restart mechanism, and it turns out **the shared
low-level half of it already lives in EdelweissFE, unwired**:

- `edelweissfe/models/femodel.py:359-396` — `FEModel.writeRestart`/`readRestart`
  (HDF5, dumps `model.time` + every `NodeField._values` entry — `U`, `dU`, `P`, ...).
- `edelweissfe/timesteppers/adaptivetimestepper.py:223-267` —
  `AdaptiveTimeStepper.writeRestart`/`readRestart` (scalar bookkeeping: `currentTime`,
  `increment`, `incrementCounter`, `dT`, ...).

`MPMModel` (EdelweissMeshfree) subclasses `FEModel` and calls `super().writeRestart/readRestart`
unchanged, adding only particle state on top. Nobody in EdelweissFE calls these two methods —
no keyword, no CLI flag, no driver wiring, no test.

EdelweissMeshfree's restart is a pure Python-API feature (kwargs to `solveStep(...)`,
manual `readRestart()` call in a user script) with no `.inp` keyword at all. EdelweissFE
is keyword/schema-driven for everything else in its input language (`edelweissfe/keywords/`,
`PLAN_INPUT_SYSTEM.md`), so this plan diverges from Meshfree's approach where it matters:
restart configuration should be an `.inp` keyword, not a `solveStep()` kwarg pile.

## Design decisions worth flagging up front

1. **Reconstruct-then-overwrite, not full serialization.** Like Meshfree, we do *not*
   serialize model topology (nodes/elements/sets/sections/materials) — the model is
   rebuilt from the `.inp` file as normal, then converged state is overwritten from the
   checkpoint. This matches the existing scaffold and avoids the Cython/Marmot-pointer
   pickling problem (`MarmotElementWrapper` holds a raw C++ pointer, not stock-picklable).

2. **Element state already has the right accessor pair.** `BaseElement.getStateVars()` /
   `setStateVars()` (`edelweissfe/elements/base/baseelement.py:323-345`) already exist and
   are implemented for both Marmot-backed (`marmotelement/element.pyx:318-333`) and pure
   Python (`displacementelement/element.py`) elements — reused today by AMR for
   parent/child history transfer. This is actually a better starting point than
   EdelweissMeshfree has: Meshfree's equivalent (`BaseParticle.getRestartData`) is only
   implemented for the Marmot particle wrapper, not the Python one. **No new interface
   needed here, just wiring it into `FEModel.writeRestart`/`readRestart`.**

3. **AMR interaction is an open risk, not solved by this plan.** The reconstruct-then-overwrite
   assumption implicitly requires that rebuilding from the `.inp` file reproduces the exact
   same mesh topology. That's false once adaptive mesh refinement (hanging-nodes AMR,
   `feat/amr-hanging-nodes`) has altered the mesh at runtime based on error markers — the
   refined topology is not recoverable from the original `.inp` alone. **v1 of restart should
   explicitly only be supported for static-topology analyses**; AMR+restart is called out as
   follow-up work (would need to serialize the refinement history or the element/node tree
   itself, not just state).

4. **Writing vs. fallback-on-failure are different mechanisms, keep them that way.**
   Periodic checkpoint *writing* fits neatly into the existing `OutputManagerBase` lifecycle
   (`initializeJob`/`finalizeIncrement`/`finalizeJob`, see `outputmanagers/statusfile.py` as
   the template) — no new hook needed. Automatic rollback-and-retry on non-convergence
   (Meshfree's `allowFallBackToRestart`) is solver-loop logic (needs access to the cutback
   path) and doesn't belong in an output manager; if we want it at all, it's a P3/stretch
   item scoped only to `nonlinearimplicitstatic.py` first.

## Phases

### P0 — Complete the existing (unwired, partial) state scaffold

- `FEModel.writeRestart`/`readRestart`: add an `elements` HDF5 group, looping
  `self.elements.values()`, keyed by `str(element.number)`, using the existing
  `getStateVars()`/`setStateVars()`.
- Add `model.scalarVariables` (Lagrange multipliers, e.g. arc-length load factor,
  penalty constraint multipliers) to the serialized set — currently completely missing.
- Check constraint-internal state (e.g. `NodeToDeformableSurfacePenaltyConstraint`'s
  `acceptLastState`-managed contact-active flags in
  `constraints/nodetodeformablesurfacepenalty.py:837`) — decide what, if anything, needs
  a `getRestartData`/`setRestartData`-style pair there too.
- Fix the stale `fileName` docstrings in `femodel.py` (parameter is actually an open
  `h5py.File`).
- Unit test: build a small model in-memory, write to a temp HDF5 file, mutate state,
  read back, assert round-trip equality (nodeFields, element state, scalarVariables).

### P1 — `*restart` input-file keyword

- New `edelweissfe/keywords/restart.py`: `RestartSchema` (dataclass) + `RestartKeyword`,
  following the `JobKeyword`/`OutputKeyword` pattern. Fields:
  - `write: bool = False`
  - `writeInterval: int = 1` (write every N converged increments/steps — TBD which
    granularity matches EdelweissFE's step model best)
  - `baseName: str = "restart"`
  - `numberOfFilesToKeep: int = 3` (ring buffer, mirrors Meshfree's `RestartHistoryManager`)
  - `readFrom: str | None = None` (path to an existing checkpoint to resume from)
- Register in the parser's dispatch table (`inputfileparser.py`'s
  `_DISPATCH_CATEGORY_BY_KEYWORD`) analogous to `job`.

### P2 — Driver wiring (resume path)

In `drivers/inputfiledrivensimulation.py`, right after
`model = fillFEModelFromInputFile(...)` / `model.prepareYourself(journal)` and *before*
`model.advanceToTime(job.get("startTime", 0.0))`:

- If a `*restart, readFrom=...` was parsed: open the HDF5 file, call
  `model.readRestart(f)`, and — since the timestepper is constructed later, per-step —
  thread the restored `currentTime`/`increment`/counters into whichever step's
  timestepper corresponds to resuming (need to determine, from the checkpoint's stored
  time, which step in `stepManager.generateSteps(...)` to resume at and skip prior
  completed steps entirely).
- This "which step do we resume into" question is the trickiest new part relative to
  Meshfree (Meshfree doesn't have EdelweissFE's multi-`*step` structure — it just calls
  `solveStep` once per script). Needs a small design spike before implementation:
  simplest viable v1 is probably "restart is only supported to resume within the step it
  was written in" (store the step name/index in the checkpoint, skip prior steps, hand
  the restored timestepper state to that step's `_createTimeStepper()`).

### P3 — Restart-writing output manager

- New `outputmanagers/restart.py`, same shape as `statusfile.py`: schema-driven
  (`write`, `writeInterval`, `baseName`, `numberOfFilesToKeep` — reuse fields from the
  `*restart` keyword rather than duplicating, e.g. the keyword's parsed config is just
  handed to this output manager's constructor).
- Hooks into `finalizeIncrement()` — call `model.writeRestart(f)` +
  `timeStepper.writeRestart(f)` into a rotating `"{baseName}_{n}.h5"` file every
  `writeInterval` converged increments, using a small `RestartHistoryManager`-equivalent
  ring buffer ported near-verbatim from
  `EdelweissMeshfree/edelweissmeshfree/solvers/base/nonlinearsolverbase.py:68-88`.
- Registered via `*output, type=restart` through the existing `_DISPATCH_CATEGORY_BY_KEYWORD["output"]`
  path — no parser changes needed beyond what already exists for other output managers.

### P4 — Test

- New test case, e.g. `testfiles/marmot/RestartTest/` (needs a real Marmot element with
  nontrivial state, e.g. `VonMises` or `CDP`, to actually exercise `getStateVars`/`setStateVars`
  round-tripping — a `LinearElastic` case wouldn't catch bugs in state transfer since it
  has no history).
- Mirror EdelweissMeshfree's `examples/114_marmot_micropolar_snni_quad_restart_test/` pattern:
  run once uninterrupted to produce the reference `U.ref`; run again deliberately truncated
  (e.g. a step with an artificially low `maxNumberIncrements` or similar), write a restart
  checkpoint, then run a second `.inp`/driver invocation with `*restart, readFrom=...` to
  resume and finish, and diff final `U` against the uninterrupted reference.
- This two-invocation shape doesn't fit `run_tests_edelweissfe`'s single
  `test.inp`/`U.ref` model directly — needs either (a) a small wrapper script akin to
  Meshfree's pytest-style example test, or (b) extending the test runner to support a
  "run twice, second run resumes" case. Decide at implementation time; don't force it
  into the existing single-shot runner if it doesn't fit.

### P5 (stretch, not v1) — Fallback-on-failure

- Port `allowFallBackToRestart` behavior (`nonlinearsolverbase.py:402-435`,
  `_tryFallbackWithRestartFiles`) into `nonlinearimplicitstatic.py`'s existing cutback
  path only, reusing the ring buffer from P3's output manager. Skip for explicit/dynamic
  and arc-length solvers initially (avoid re-introducing Meshfree's per-solver duplication
  — if this proves useful, factor the cutback-fallback logic into a shared solver-base
  mixin instead of copy-pasting per solver).

## Explicit non-goals for v1

- AMR-refined mesh restart (see decision #3).
- True material-point (non-particle) restart — N/A in EdelweissFE, only relevant to
  EdelweissMeshfree, and even there it's an existing gap (material points aren't
  serialized at all today, only RKPM particles).
- Cross-version / cross-schema checkpoint compatibility (no versioning of the HDF5 layout
  planned initially).
