# Handoff: MPI explicit solver (NEDMPI), tie fix, topology-check performance — session 2026-09-30 … 10-02

Start here. The detailed documents are referenced below and stay valid:

- `HANDOFF_MPI_EXPLICIT.md`, the NEDMPI design and its measurements:
  - §1–8: the solver;
  - §9: distributed elements;
  - §10: the c1_150 production test.
- `PLAN_TOPOLOGY_CHECK_COST.md`: the A1–A6 performance plan, with measurements.
- `PLAN_REBASE_AFTER_NED_MERGES.md`: **the next step**, the rebase strategy.

Machine: **rabbit**. Nothing is running. Nothing is pushed except the tie fix, which is merged.

---

## 1. Branches and worktrees (EdelweissFE)

| Worktree (`next_v2611/…`) | Branch | Base | HEAD | State |
|---|---|---|---|---|
| `EdelweissFE-mpi` | `feat/mpi-explicit-dynamics` | `6e08a2d7` | `0f404ffc` | **rebased 2026-10-02** (13 + 4 adaptation commits), verified, local only; pre-rebase backup branch `backup/mpi-explicit-pre-rebase-2026-10-02` |
| `EdelweissFE-upstream` | detached `6e08a2d7` | — | — | pristine upstream + copied `.so`, the serial reference for verification |
| `EdelweissFE-A2` | `perf/contact-facet-node-lists` | `6e08a2d7` | `a7e832c5` | done, bitwise; PR opened |
| `EdelweissFE-A3` | `perf/rigid-contact-search-batched` | `6e08a2d7` | `93d937a1` | done, bitwise; PR opened |
| `EdelweissFE-A4` | `perf/marker-halo-adjacency` | `6e08a2d7` | `f8921aa2` | done, bitwise; PR opened |
| `EdelweissFE-A5` | `perf/restart-history-coalesce` | `6e08a2d7` | `0532c30f` | done, bitwise; PR opened |
| `EdelweissFE-tiefix` | `fix/tie-adjust-element-geometry` | — | `183f76a6` | **merged as #184** (`6e08a2d7`); worktree removable |

Upstream `origin/next_v26.11` = `6e08a2d7`. It also contains #183, which is plan item A1: it makes a
topology check that only changes contact connectivity keep the system.

The four perf worktrees contain compiled `.so` files copied from `EdelweissFE-mpi`; no Cython source
changed. Run their code with `PYTHONPATH=<worktree>`, and check the origin with
`python -c "import edelweissfe; print(edelweissfe.__file__)"`. The test runs left untracked outputs
under `testfiles/` in the worktrees; they are not committed.

## 2. Environments and tools

- **conda env `next_v2611_mpi`** (clone of `next_v2611`): `mpi4py 4.1.2`, conda-forge
  `openmpi 5.0.11`, `libmetis` (bound via ctypes; pymetis has no 3.14t build). EdelweissFE is
  **editable-installed from `EdelweissFE-mpi`**. The main `next_v2611` env is untouched.
- **conda env `tools`**: `gh 2.102.0`, **not logged in**. Run
  `! /home/matthias/constitutive_modeling/miniforge3/envs/tools/bin/gh auth login` once before opening
  PRs from rabbit. x9 is not reachable from rabbit.
- **Always run with** `PYTHON_GIL=0 MKL_CBWR=AUTO,STRICT`, plus `OMP_NUM_THREADS`.
- **Launching under MPI:**
  - pure MPI: `mpirun -n N --bind-to core`;
  - hybrid: `--map-by slot:PE=T --bind-to core`, or `--bind-to none`;
  - forward settings with `-x OMP_NUM_THREADS -x PYTHON_GIL`.
  - Open MPI's default binding collapses the thread pools.
- **Benchmarks and tools:** `mpi_explicit_benchmarks/`.
  - `tools/`:
    - `dumpUPV.py`: run decks and save U/P/V bit-exactly;
    - `cmp.py`;
    - `h5cmp.py`;
    - `inject_failure.py`;
    - `run_with_peakrss.py`.
  - `gcdp_live_amr/`, `strong_scaling/` (with the results csv files), `restart/`.
  - `c1_150/`:
    - `leo5_snapshot/`: deck, mesh, `restart_0.h5`, LEO5 log;
    - `make_run.py`, `run_unverified.py`: resumed windows; fingerprint verification off, see §5;
    - `make_fresh.py`: from-scratch windows;
    - `check_tie_adjust.py`;
    - `runs/`: the comparison runs and the profiles;
    - **`verify/verify.sh WORKTREE LABEL [NPROC]`**: 250 increments of c1_150 with topology
      checks, compared bitwise to `verify/baseline.npy` (= `6e08a2d7`; override with
      `BASELINE=…`). `verify/baseline_mpibranch.npy` = the old MPI base.

## 3. What NEDMPI is (summary of HANDOFF_MPI_EXPLICIT.md)

**The model in every process.**
- Every rank builds the complete model and computes one METIS subdomain:
  - elements via `METIS_PartMeshDual` with the communication-volume objective, computed on rank 0 and
    broadcast;
  - constraints dealt out round-robin by name.
- Under a launcher, Marmot elements are `DistributedMarmotElement`:
  - it holds the topology, a replica of the converged state, a recorded setup, and the node
    coordinates as of `setNodes`;
  - the C++ Marmot element is attached only on the owner (or for a neighbour's load), and built
    transiently for single questions;
  - `EDELWEISSFE_DISTRIBUTED_ELEMENTS=0` turns this off.

**Each increment.**
- Interface element contributions are exchanged and summed **in model element order**.
- Constraint forces are shared by `Allgatherv` and added in model order.
- Consequence: **bit-identical to NED on any number of processes**, through contact, ties, restart,
  live AMR and measured-cost rebalancing.

**Where the whole model is read.**
- Output, contact searches, markers and refinement read the whole model, so they sync first:
  vectors from the owners, and element and constraint states via `getStateVars` / `getRestartData`.
- AMR is replicated on every rank and checked by a layout fingerprint; the model is then
  repartitioned, with no migration.

**Output and failures.**
- Only rank 0 writes output, and its checkpoints are ordinary ones.
- Element failures are agreed on by all ranks.

**Code:**
- `edelweissfe/domaindecomposition/`:
  - `mpienvironment`;
  - `metis`;
  - `partitioning`;
  - `subdomaininterface` (`SubdomainInterface`, `InterfaceForceAssembly`, `ConstraintForceExchange`);
  - `statesynchronization`;
  - `distributedelement`.
- `solvers/nonlinearexplicitdynamicmpi.py`.
- NED hooks: the `# --- Subdomain hooks` section.
- Driver: only rank 0 writes; non-serial solvers are refused under MPI.
- Tests: `testfiles/mpi/` (16 decks) and `tests/test_domaindecomposition.py`.
- Docs: `doc/source/documentation/domaindecomposition.rst`.

## 4. Key measurements

**192k GC3D8 GCDP block (rabbit, 32 cores), ms per increment:**

| configuration | ms per increment | memory |
|---|---|---|
| 32 threads | 118 | — |
| NEDMPI 32×1 | 108.7 | 1.8 GB per process |
| NEDMPI 8×4 | 92 | 2.6 GB per process |
| (before the distributed elements) | — | 8.2 GB per process |

**c1_150 production window (1.94M DOFs, contact, 5 ties, live AMR), resumed from LEO5:**

| configuration | ms per increment | peak memory |
|---|---|---|
| 32 threads | 180 | 14 GB |
| NEDMPI 8×4 | 155 | 8.1 GB per process |
| NEDMPI 32×1 | 238 | 6.6 GB per process (211 GB total); limited by imbalance, contention and replicated topology checks |

All configurations are bitwise identical.

**A2–A6 gains:**

| item | change |
|---|---|
| A3 | rigid contact search 1.97 → 0.12 s per call |
| A5 | resume coalesce: about 117 → 4.4 s per resume |
| A6 | NEDMPI 8×4 regular check 3.3 → 1.7 s |
| A2 | about 1 s per full rebuild |
| A4 | marking 0.31 → 0.02 s per check |

## 5. Findings worth remembering

- **Tie `adjust` bug** (fixed, #184): elements kept unsnapped geometry, 525 elements on c1_150, by up
  to 0.5 mm. The old "exact 540" of `Tie2D`/`TieHexa20Patch` came from the bug. The upper blocks are
  now 0.98 tall. **This changes results of edge-breakout runs with snapped ties**, including the LEO5
  runs made before #184.
- **Cross-machine checkpoints:** AMR node coordinates differ in their last bits between the LEO5
  container (EdelweissFE `8e63e8e9`, Marmot `20dc4ad5`) and rabbit. The fingerprint hashes raw bytes,
  so a LEO5 checkpoint only resumes on rabbit with `verifyTopologyFingerprints=False`. The plans are
  identical, and the resume reproduces LEO5's AMR events.
- **Resume cost:** replaying 372 records takes about 30 minutes; the post-resume coalesce was about
  2 minutes, now 4 s with A5.
- **cProfile misattributes worker-thread time in threaded runs.** It inflated the A2 and A4
  estimates; use direct timers.
- **Pre-existing problems, not fixed:**
  - NEDParallel with pure-Python materials at more than one thread is about 15% wrong and not
    reproducible;
  - a serial NED restart with live AMR deviates by 1.3e-3 and its first resumed reaction-force sample
    is 0; possibly addressed by #166, so re-check;
  - geostatic in NED is re-applied every increment and blows up;
  - `CDPPlane` fails on upstream.
- **Memory on rabbit:** watch it when running many ranks of big models. 32 × 6.6 GB came within 18 GB
  of the limit.

## 5b. The rebase (done 2026-10-02)

- Hooks: `_finalizeOutput` (field outputs + non-checkpoint managers, before the topology check) and new
  `_writeCheckpoints` (after it, #166); the branch's `_publishToNodeFields` dropped for upstream's
  `publishNodeFields`; `_updateConnectivityOf(..., resumed)`; NEDMPI agrees on both #183 flags (the
  old override did `bool(tuple)`, always True). Stray `klu.c` (Cython output) removed from history.
- Adaptation commits: `40653fcc` distributed element `_forgetTransient` + `_materialProperties` (#182);
  `2665e76c` NEDMPI re-syncs constraint states before a checkpoint after a topology check (owner-only
  searches; a control without it differs in contactPlateConcrete/contactShaftHole projections);
  `d826182f` **FrozenContactSearch restart data round-trips** (staged data was never reported, so rank 0
  checkpointed stale projections: NEDRestart2Resume 75 % off at 2-4 procs); `0f404ffc` TieNED U.ref.
- Verified bitwise: NED decks vs upstream; testfiles/mpi at 1/2/3/4 and 4x2 vs serial upstream; c1_150
  `verify.sh 8` vs baseline; c1_150 checkpoint at 250 NED == NEDMPI 8x4 (69573 datasets), resumed by NED
  and NEDMPI to 350 == straight NED; materialParameterFromField + matprops export on Marmot + live AMR at
  1-3 procs; serial suites/pytest identical to upstream (CDPPlane fails on both).

## 5c. Teaching-fit refactor (done 2026-10-02, see PLAN_MPI_REFACTOR.md)

Review verdict: numerically sound, but grafted onto NED via ~15 identity hooks, duplicated element
loops, an env-var class swap and an undocumented proxy element. Two agents fixed it on
`refactor/mpi-part` and `refactor/mpi-elem` (merged into `feat/mpi-explicit-dynamics`, HEAD `b07cd941`):
- `solvers/base/modelpartition.py`: `ModelPartition` (one concept, introduced once in NED's docstring),
  `WholeModel` serial; `domaindecomposition/subdomain.py:Subdomain` the MPI one. NEDMPI 1019 -> 206 lines.
- One explicit element loop (`planElements`/`ElementPlan`, `computeElementsForExplicit`,
  `computeLumpedDiagonalForExplicit`) for NED/NEDParallel/NEDMPI; dataclasses instead of tuple caches;
  rebalancing an explicit loop step.
- Elements: `BaseElement.computingObject/startComputingHere/stopComputingHere` (also on the Cython
  wrapper -> **rebuild needed**: `python setup.py build_ext --inplace -j 16`); `MarmotElementCache`;
  `MarmotElementWrapper.getResultLocation`; `marmotElementsAreDistributed()` + a journal line, env var gone.
- Verified: `mpi_explicit_benchmarks/tools/verify_refactor.sh <wt> <label> full` all BITWISE (decks
  serial/1-4/4x2, fieldmat, c1_150 NEDParallel32 + 8x4, checkpoint == NED, resume == straight); timing
  NEDParallel32 0.0978 s/inc (was 0.104), 8x4 0.0914 (0.091); suites == upstream; pytest 624 passed;
  sphinx no new warnings. Harness 4x2 compares Marmot decks only (pure-Python threaded race, pre-existing).

## 6. Next steps, in order (see PLAN_REBASE_AFTER_NED_MERGES.md)

1. ~~**Open the PRs for A2–A5**~~ OPENED 2026-10-02: A2 #185, A3 #186, A4 #187, A5 #188 (full suites = upstream control).: four independent PRs against `next_v26.11`. They merge without
   conflict against each other and the tip. Commit messages and bodies need the Claude-Session
   footer. Push to `mn`; open the PRs via `gh` once logged in.
2. ~~**Rebase `feat/mpi-explicit-dynamics` onto `6e08a2d7`**~~ DONE, see §5b. with a plain `git rebase`. The only textual
   conflict is NED's topology-check loop (#183). Adaptations:
   - **#183:** the two-flag `updateTopologyAndConnectivity`, agreed across ranks, with A6's hook
     inside it.
   - **#166:** the `resumed` / `resumeConnectivity` path together with owner-only searches; the
     frozen projection in synchronized restart data.
   - **#182:** `DistributedMarmotElement._recordAndForward` and `setNodes` must call
     `_forgetTransient()`, because sections now query `getCoordinatesAtCenter()` before
     `setMaterial`. Also `el._materialProperties` is accessed in `writeMaterialPropertiesToFile`.
   - **#184:** regenerate `testfiles/mpi/edelweiss-only/TieNED/U.ref`, and re-check all
     `testfiles/mpi` `U.ref` files against serial on the new base.
   - **Verify:** serial NED bitwise; `testfiles/mpi` at 1–4 processes and 4×2; c1_150
     `verify.sh … 8` against `baseline.npy`; an MPI restart; a field-material deck.
3. After the A2–A5 PRs merge: a trivial rebase of the MPI branch, then re-run the checks.
4. **Later.**
   - LEO5:
     - add `mpi4py`, `openmpi` and `metis` to the Apptainer image;
     - single node first (mpirun inside the container);
     - multi-node needs host PMIx / ABI work;
     - **job submission needs the user's OK**.
   - NEDMPI scaling:
     - load imbalance at 32 ranks;
     - about 6 GB replicated per process on c1_150 (facets, MPC records, AMR mirror, DofManager);
     - refinement is replicated on every rank.
   - The MPI branch also still needs a PR and docs review before merge.

## 7. Cleanup candidates (ask first)

- The `EdelweissFE-tiefix` worktree and local branch, both merged.
- Scratch outputs under `/tmp/claude-1000/...` (session scratchpad; the useful scripts were copied
  into `mpi_explicit_benchmarks/tools`).
- Large run directories under `mpi_explicit_benchmarks/c1_150/runs/`: logs and the profile `.out`
  files are small, and no checkpoints were written there.

## 5d. Transplant onto upstream 804d3817 (2026-10-05, see PLAN_TRANSPLANT_2026-10-05.md)

**The MPI work now lives on `feat/mpi-explicit-dynamics-v2`** (worktree `EdelweissFE-mpi2`, 3 commits on
origin/next_v26.11 = 804d3817: `de5f7aae` serial partition + one element loop, `6426189b` NEDMPI,
`718dbb15` docs). The old `feat/mpi-explicit-dynamics` (+ `backup/mpi-explicit-pre-transplant-2026-10-05`)
is superseded. No .pyx changes; DistributedMarmotElement not ported (removed in P1).
- Adapted to: #173 fused velocity update (restricted to integrated DOFs, views for WholeModel), #191
  base-class loads (hook `elementsLoadedHere`) and TopologyUpdate (hook `updateConnectivityOf`, flags
  agreed across ranks), #199 increment loop (hook `writeIncrementOutput`, called by StepBase.solve), #198
  restart (constraint restart data adopted immediately -> copies become their owners' after a sync;
  only owners' DOF indices are read; topology check in prepareIncrement after the checkpoint).
- Dropped as obsolete: round-trip staging (d826182f), resumeConnectivity plumbing, pre-checkpoint re-sync.
- Verified (agent + orchestrator, REF = mpi_explicit_benchmarks/refactor_reference_804d3817): every
  harness line BITWISE incl. c1_150 NEDParallel32 (0.0883 s/inc vs upstream 0.0919) and NEDMPI 8x4
  (0.0905), checkpoint NEDMPI == NED (96265 datasets), resume == straight; suites == upstream (CDPPlane
  only); pytest 1300 passed; sphinx no new warnings. New references: checkpoint format v8 (old v1 refused).
- testfiles/mpi keeps NEDRestart1Write/2Resume (upstream #201 removed the serial ones in favour of
  tests/test_contact_restart_frozen_projection.py).
- Open for review: the three small hooks in shared upstream code (NonlinearSolverBase, StepBase).
- Next: P2 (DESIGN_P2_MESH_AS_DATA.md) on top of v2, after the user's answers to its §8.

## 5e. Review fixes (2026-10-06; REVIEW_MPI_2026-10-06.md, PLAN_REVIEW_FIXES_2026-10-06.md)

`feat/mpi-explicit-dynamics-v2` now at `678c0cc1` (11 commits on 718dbb15, merged from review/mpi-v2-solver + review/mpi-v2-ci):
- ModelPartition = frozen dataclass (elements, constraints, dofs, ownedDofMask) + wholeModel(); NED has no
  communication (2424 lines; upstream 2101); all collectives are NEDMPI overrides (668 lines, table in
  domaindecomposition.rst); energy balance = NED.reportEnergyBalance; elementsLoadedHere removed (NEDMPI
  hands restricted load views to the shared assembly); addNodalForces in numerics/assembly.py.
- Risks closed: stateless elements refused under MPI; topology update in the failure agreement;
  KeyboardInterrupt aborts all ranks; excepthook installed by the driver; fingerprint hashes element DOF
  indices + node coordinates; constraint-copy currency asserted before topology updates; agreed cutback
  = smallest requested.
- CI: .github/workflows/run_tests_mpi.yml (mpirun -n 2/3, bitwise vs fresh serial --create; 12 decks,
  GCDP skipped as private); runner: --tolerance option, process-count header. NOT verified on GitHub (no push).
- Verified: harness full BITWISE (serial32 0.0887 s/inc, 8x4 0.0883), checkpoint/resume identical;
  suites == previous; pytest 1300; sphinx no new warnings; orchestrator re-ran quick + mpirun runner + pytest + lint on the merged tip.
- 2026-10-06: user: no PR for now; run_tests_mpi.yml therefore unverified on GitHub (dispatch impossible: not on the default branch).

## 5f. P2 "mesh as data" (2026-10-06) — merged into feat/mpi-explicit-dynamics-v2 (41eae721, 13 commits, local; mn is 13 behind)
Mesh (models/mesh.py) + FEModel.createElementOfMesh/createElementsOfMesh(isCreatedHere, stored predicate),
ElementSet.isComplete (derived) / requireComplete, requireCompleteMesh at whole-model readers, fields from the
mesh, all generators describe, facets cut from the mesh, AMR through the mesh, SurfaceFace. Reviewed (Fable):
P0 late-described elements fixed by mechanism (re-create after late generators + prepareYourself invariant).
Gates: harness full BITWISE (0.0871 / 0.0894 s/inc), suites == base, pytest 1310, sphinx same; one METIS part of
c1_150 builds with the identical DOF layout; build RSS +0.7 %. P3 follow-ups: DESIGN_P2_MESH_AS_DATA.md §10.
Upstream issue noticed: edelweissfe/constraints/test_tie.py has 3 failures on 804d3817 itself.

## 5g. P3 owned elements (2026-10-06) — merged locally into feat/mpi-explicit-dynamics-v2 (1da2add5), NOT pushed
Each process creates only its own elements (+ loaded neighbours) for models outside the whole-model rule
(model modifiers, any constraint, executePythonCode/cubit/surface/rigid generators, late generators,
fromExpression over elSets); one rule in domaindecomposition/distributedelements.py, logged on rank 0.
192k GC3D8 block peak RSS/process: 8.26 GB (whole, 8 procs) -> 1.77 GB (8) / 1.08 GB (32). Element field
outputs allgather; Ensight from the mesh; checkpoints gather states to rank 0 only when written. Rebalancing
off in distributed mode (migration deferred). Reviewed (Fable) + fixes; all gates bitwise; per-deck mode
check testfiles/mpi/check_element_distribution.py (8 distributed / 10 whole). Upstream defect: a load active
across a restart fails at step end (0-d array +=), see DESIGN_DISTRIBUTED_MEMORY.md §7.
Next: migration (rebalancing in distributed mode), P4 AMR, P5 contact/constraints.

## 5h. Element migration (2026-10-06/07) — merged locally into v2 (ef12738d), NOT pushed
Rebalancing in distributed mode moves elements (state vector sent; receiver recreates from mesh + section +
properties; sender drops; NED.releaseEquationSystem + rebuild). Gain guard against repeated migrations; random
thickness section is now a proper plane-section subclass (was unusable from input before). Reviewed (Fable) +
fixes; all gates bitwise; test deck NEDRebalanceDistributed moves 13/31 elements, gc proves none left alive.
Performance (192k block): no gain possible (imbalance only ~5 %); migration ~3.5 s, +80-150 MB transient; no
locality effect (moved vs unmoved cost factor 1.018 vs 1.019). Finding: per-process speed differs ~8 % for the
same element (bind-to none), biasing measured-cost repartitions. Open: normalise costs per process?
incremental repartition (from-scratch moves 20-30 % for small imbalances)? Next: P4 (AMR), P5 (contact/constraints).

## 5i. P4 distributed AMR (2026-10-07) — merged locally into v2, NOT pushed
Live AMR runs distributed: octree/marking/balance/hanging nodes/numbering replicated from mesh + node fields
+ gathered field outputs; element objects/states owner-local (parent's owner creates children, transfers state).
hAdaptivity left the whole-model rule (modifiers declare wholeModelReason; surfaceSnap stays). Gain guard after
refinement: estimate from element counts, move-only cost, horizon = min(left in step, since last change).
External work = per-increment math.fsum of the prescribed-DOF products, gathered only where read (bitwise for
any process count; references in refactor_reference_fsum, identical to 804d3817's). ElementSetOfMesh subclass.
Reviewed (Fable) + fixes; all gates bitwise; 8 AMR decks distributed, incl. recoveryError, migration, restart.
60x20x20 GC3D20R AMR model, 8 procs: wall 159 s vs 220 s fallback, peak RSS 2.7-3.2 vs 8.4 GB, increment
0.137 vs 0.132 s. c1_150 timings unchanged (still fallback: contact). Next: P5 (contact/constraints).

## 5j. P5 distributed contact/ties (2026-10-07) — merged locally into v2, NOT pushed
c1_150 runs DISTRIBUTED at 8x4: bitwise vs serial, checkpoint/resume identical, 2.55 GB/proc vs 7.57 fallback,
not slower. Facets/point masses on every process, computed beside their host element (hostElement); tie,
s2s, node-to-surface, s2rigid, nodeToDiscreteRigidBody, equalValuePenalty, nodeToRigidSurface,
directionalSpring run distributed; implicit-only constraint types say so (constraints/base/wholemodel.py).
Fixed a P3 bug (per-node output over a locally empty element set). Reviewed (Fable) + fixes.
Opus reviews (REVIEW_MPI_2026-10-07.md): CONFIRMED pre-existing load bug (stale U for configuration-dependent
loads) -> fix next (PLAN_LOAD_ASSEMBLY.md, branch fix/load-assembly), then the teaching restructure.
Follow-up: a finished model with a rigid body stays alive in the process (closure cell holding {'model': …}).

## 5k. Load assembly fix (2026-10-07) — merged into v2 and PUSHED together with P5 (mn at ac025e4f)
Owner evaluates its elements' loads; tagged entries (global ordinal: loads in deck order, faces, element position
in the mesh description) completed at the interface in "layers" = the serial left fold, bit for bit (signed zeros
incl.). Loaded-neighbour halo removed (tie/constraint + element load now works distributed). Load evaluation in
the failure agreement (test with injected failure). Regression decks: NEDFollowerPressureUL (+WholeModel),
NEDLoadsEqualValuePenalty. Reviewed (Fable): merge as is. 192k block with gravity -18 %/increment.
Next: the teaching restructure (REVIEW_MPI_2026-10-07.md, professor's 5 items).

## 5l. Teaching restructure (2026-10-07) — merged locally into v2 (d75b3723), NOT pushed
NED's increment = prescribeBoundaryMotion, updateVelocities (+imposeKinematicConstraints), advanceDisplacements,
assembleNetForce, with formulas; walk-through at the top of the module. MPI-only machinery in NEDMPI. Mesh is the
single source of truth (two-way check at prepareYourself + after topology refresh; createAuxiliaryElement).
Partial element sets raise; localElements() explicit. Standard vocabulary (owned/local/replicated/ranks,
allreduce*, allgatherOwnedValues, allRanksFailTogether). incrementsLeftEstimate non-abstract. Bit-identity claim
precise + adversarial ordering tests (element loop, interface, constraint forces, loads). Material-property
export gathered to rank 0. DofVector.copy leak fixed. Reviewed (Fable) + fixes; all gates bitwise.
UPSTREAM BUG FOUND (serial, 804d3817): changeMaterialProperty re-assigns the section every increment ->
MarmotElementWrapper.setMaterial zeroes the state arrays -> plastic/damage history wiped each increment.

## 5m. State on 2026-10-07 — MPI work PAUSED (user)
- mn: feat/mpi-explicit-dynamics-v2 at d75b3723 (no PR). Upstream next_v26.11 moved on (1870025c); a rebase/merge
  onto the current upstream will be needed when work resumes.
- env next_v2611_mpi: editable install re-pointed to EdelweissFE-mpi2 (was the superseded EdelweissFE-mpi);
  quick harness bitwise after the rebuild.
- Upstream PR #205 (changeMaterialProperty wiped element state every increment; BaseElement.updateMaterialProperty).
- NEXT ROUND when resuming (user-approved, REVIEW_MPI_2026-10-07b.md):
  1. Hang risks: no collective inside allRanksFailTogether (evaluate locally -> agree -> gather; two-phase field
     outputs); a rank-local StepFailed must agree/abort when nProcesses > 1.
  2. Lumped operators across migration: carry them like U/V/P, or assert bitwise equality.
  3. Serial stale-set trap: Mesh.removeElement scrubs element sets (or checkElementsAgreeWithMesh covers sets);
     the incomplete-set error must not speak of processes in a serial run.
  4. MPI knowledge out of FEModel: no isRootProcess import; advanceToTime back to its old signature; checkpoint
     gathering out of ElementDistribution into Subdomain/NEDMPI; child placement as one FEModel method
     (hadaptivity names no process); single-consumer base hooks into the MPI layer; provider=None ambiguity.
  5. Smaller: doc recipe for rebalance decks; bincount-order assumption pinned/documented.
  Later (architect): gathers only to the reader, contact partitioned by slave entity, P6 local DOFs/ghosts,
  retire the replicated fallback.

## 5n. Resumed (2026-10-07): upstream merge + cylinderGenerator + round 5m — local v2 at 0cdc9524, NOT pushed
- Merge commit cf0594d6 brings upstream 1870025c (cylinderGenerator #78, GCPE8UL/DP tests #164); cylinderGenerator
  converted to describe the mesh (byte-identical generator output vs upstream); new deck NEDCylinderGenerator.
  Env libMarmot lacks GCPE8UL (#164 decks skip; pass bitwise against a scratch Marmot next_v26.11 build).
- Round 5m (PLAN_ROUND_5M.md) + review (Fable) + fixes: Communicator wrapper, collectives raise inside
  withoutCommunication/allRanksFailTogether; three-phase output (read / gather / write); replicated StepFailed ->
  StepFailedOnAllRanks (graceful); lone StepFailed -> NEDMPI.endStep(step, model, failure) aborts before
  finalizeStep can gather (endStep gained a `failure` argument in StepBase/all solvers); lumped operators asserted
  bitwise after migration; Mesh/FEModel.removeElements scrub sets/surfaces; FEModel without isRootProcess,
  advanceToTime(time) restored + acceptStatesOf, checkpoint gathering in Subdomain, createChildElementOfMesh,
  ownedDofMask in Subdomain, mesh.AUXILIARY marker; loadsonsubdomain.py split out.
- Verified combined: harness quick BITWISE, CylinderGen deck, mode check, pytest 1341.
- Ordering-cost experiment (mpi_explicit_benchmarks/ordering_cost/RESULTS.md): bit-identity costs ~1 %;
  constraint-force Allgatherv costs +43 % on c1_150 32x1; "hybrid" point-to-point constraint exchange is bitwise
  and removes it (134.9 -> 94.5 ms/inc) — proposed next change, awaiting user OK.

## 5o. Marmot update in env next_v2611_mpi (2026-10-08)
Marmot e992a3f4 (#114) + nested modules at origin/master (MeshfreeCore at origin/next_v26.11) built from
worktrees at next_v2611/marmot-upstream (+ nested worktrees), installed in next_v2611_mpi; main Marmot checkout
untouched. Backup + RESTORE.md: next_v2611/marmot_backup_2026-10-08/. ctest 75/75. Numerics: only the GCDP decks
change (~1e-14 rel, CDP 14c4b3b LU instead of QR) -> NEW reference set mpi_explicit_benchmarks/
refactor_reference_marmot_e992a3f4 (now the harness default); v2 full harness BITWISE against it.
LOST from the lib: the unpushed feat/hughes-winget-wrapper meshfree module features (old lib in the backup).
EdelweissMeshfree next_v26.11 is BROKEN against current EdelweissFE (49 pytest failures, upstream #199 removed
AdaptiveTimeStepper.generateTimeStep; plus ParticleSet.isComplete from our branch) — independent of Marmot.
EdelweissFE-upstream worktree now at origin/next_v26.11 0b8fc36b.

## 5p. Point-to-point constraint forces (2026-10-08) — merged locally into v2 (ca6c1d91), NOT pushed
ConstraintForceExchange sends each owned constraint's forces only to ranks integrating its DOFs (tagged entries,
added constraint by constraint in model order) — bitwise; P defined only at integrated DOFs (NaN-poison test).
c1_150 32x1 126.8 -> 94.7 ms/inc (86.5 -> 3.1 MB/inc), 8x4 89.2 -> 88.0. Reviewed (Fable): merge as is.
Optional follow-ups: tag-uniqueness comment, rename addAllConstraintForces, poison test with AMR/topology check;
rank 31 (big contact owner) 10 ms/call vs 3.2 ms in the prototype — not investigated.

## 5q. Gather only to the reader (2026-10-08) — merged locally into v2 (81bcd7a1), NOT pushed
Output increments gather to rank 0 only unless a topology check follows / step end (markers declare
fieldOutputsRead); contact searches fetch only declared nodes (nodesReadByConnectivityUpdate) point to point;
newly integrated DOFs fetched after subdomain changes; rigid-body reference nodes always delivered (found by the
fingerprint: a second bug). Loud invariants: debug-poison-stale-solution option (in CI + harness config
mpi3poison), footprint ⊆ declared nodes check, requireSameOnAllRanks on the search decision. c1_150 32x1: output
1386 -> 43 MB, search 847 -> 0.18 MB. Reviewed (Fable) + fixes; bitwise. Next: robustness round R1
(PLAN_ROBUSTNESS_R1.md, branch fix/robustness-r1, wt EdelweissFE-r1).

## 5r. Robustness round R1 (2026-10-08) — merged and PUSHED with gather-to-reader (mn at d4d44f98)
Agreed step failure ends with the accepted model (repro deck bitwise vs NED); stepFailed hook in StepBase aborts
non-agreed failures before any finally communicates; CI compares NEDMPI 2/3 ranks with serial NED incl.
checkpoints (testfiles/mpi/compare_checkpoints.py); fingerprint covers sets/surfaces/MPC weights; Ensight partial
sets; external work = exact integer sum per increment (numerics/exactsum.py, = fsum bit for bit, verified vs
Fraction by the reviewer); serial NED = plain textbook element loop (user decision; 2.8x slower serial, NEDParallel
recommended for production, hint printed > 5000 elements), plan only in NEDParallel/NEDMPI, equivalence test with
negative control; node fields published where a process begins to integrate. Reviewed (Fable) + fixes; bitwise.
Next: contact C0 (balancer weights), then P6 (DESIGN_P6_LOCAL_DOFS.md).

## §5s C0 done, P6 starts (2026-10-14)
- C0 (constraint cost in the load balancer, contact output-results fix, review fixes: minimum share, interchangeable
  shares) fast-forwarded into feat/mpi-explicit-dynamics-v2 and pushed to mn at 1ac71f9d. Bitwise everywhere.
- User decisions: no contact split now; searches not balanced; constraints before loads OK; tolerance 0.1.
- P6 round 1 (P6.0–P6.2) planned in PLAN_P6_ROUND1.md, branch feat/p6-local-nodes, worktree EdelweissFE-p6.

## §5t P6 round 1 + empty-part fix pushed; reviews 2026-10-15 (2026-10-15)
- mn feat/mpi-explicit-dynamics-v2 at fe6d6cf8: P6 round 1 (mesh owns nodes, GlobalDofNumbering, moveNodes, field
  order) + fix(partitioning) every process keeps elements (outlier timings / METIS int32 overflow).
- Reviews REVIEW_MPI_2026-10-15_{architect,teaching}.md. User: order R2 (PLAN_ROBUSTNESS_R2.md) → T2 teaching cleanup
  (ModelPartition→ModelPart, standard vocabulary + glossary, NED seams section, kineticEnergy hook, evaluate/add split,
  serial-first docstrings) → P6.3 (incl. architect F1 O(N) rebuild work, F9 constraint ownership by DOF majority).
- 2026-10-15 naming decision (option B, two agents agreed): ModelPartition→Subdomain, big Subdomain→DomainDecomposition,
  partitionModel→defineSubdomain, "partition" = METIS division only. In PLAN_TEACHING_T2.md, starts after R2.
- §5u 2026-10-15: R2 + review fixes pushed (mn 2adfeaee). T2 started in EdelweissFE-t2 (refactor/teaching-t2).
- §5v 2026-10-15: T2 teaching cleanup (+ review doc fixes) pushed (mn 7cbfa8b4). User: kineticEnergy/momentum as exact
  sums (ExactSum) in NED and NEDMPI (branch feat/exact-kinetic-energy); then P6.3.
- §5w 2026-10-15: P6.3a (interface from partition, constraints by majority, shares follow constraints) pushed (mn 12a5288e). User: re-decide constraint owners at migrations (next), then P6.3b.
- §5x 2026-10-15: owner re-decision at migrations pushed (mn ab188f8c); balance preferred over lowest constraint traffic (user). Next: P6.3b.
- §5y 2026-10-15: P6.3b decisions (MPC global for now, [dofs] until P6.6, P6.3c = local nodes + local MPC before P6.4).
- §5z 2026-10-15: P6.3b + review fixes pushed (mn 2e1b6aa1). P6.3c started (EdelweissFE-p63c, feat/p6-3c-local-nodes).
- §5aa 2026-10-16: P6.3c + review fixes pushed (mn 4a3fb63b). P6.4a (compact AMR mirror) started: PLAN_P6_4.md, EdelweissFE-p64.
- §5ab 2026-10-16: P6.4a pushed (mn 072b1b14); compat break for level>=2 checkpoints on warped meshes accepted (user). P6.4b started (EdelweissFE-p64b).
