# Plan: transplant the MPI explicit work onto the new next_v26.11 (2026-10-05)

## Situation

- Old branch `feat/mpi-explicit-dynamics` (worktree `EdelweissFE-mpi`, 29 commits on `6e08a2d7`, HEAD
  `b06b63c0`; backup `backup/mpi-explicit-pre-transplant-2026-10-05`). Its **final state** is what we
  keep: NED with one `ModelPartition` (`solvers/base/modelpartition.py`: `ModelPartition`, `WholeModel`),
  one explicit element loop (`solvers/base/parallelelementcomputation.py`: `planElements`, `ElementPlan`,
  `computeElementsForExplicit`, `computeLumpedDiagonalForExplicit`), the MPI side
  (`domaindecomposition/`: mpienvironment, metis, partitioning, subdomaininterface,
  statesynchronization, `subdomain.py:Subdomain`), `NEDMPI` (206 lines), driver/registry/test-runner
  changes, `testfiles/mpi/` (16 decks), tests, docs. Elements are parallelization-agnostic (the
  DistributedMarmotElement was removed in P1). Every process still holds the complete model.
- Upstream `origin/next_v26.11` = `804d3817`, 22 PRs later. Relevant:
  - #185–#188 = our perf PRs A2–A5 (now upstream).
  - #191: `NonlinearSolverBase.updateTopologyAndConnectivity` returns a named `TopologyUpdate`
    (topologyChanged, meshDependentsRefreshed, constraintConnectivityChanged); load assembly
    (`computeDistributedLoads`, `computeBodyForces`, `assembleLoads`) moved into the base class (K=None).
  - #194/#195: the topology pipeline (windows, number allocators, change log, fingerprint, replay) moved
    out of FEModel into `model.topology` (TopologyPipeline).
  - #196/#198/#199: restart redesign — `utils/checkpoint.py` owns the file layout; "a resumed run performs
    exactly what the uninterrupted run would"; `step.isResumed`; the frozen contact search's staged data
    and `ConstraintBase.resumeConnectivity` are **removed** (data adopted during restore, a layout
    mismatch raises); time steppers are propose/accept/reject state machines with one increment loop;
    NED's step-start topology update no longer runs on resume; versioned checkpoint format.
  - #173: fused velocity update in NED (one formula, per-DOF coefficients, second-order mask).
  - #189 hasattr removal, #190 DisplacementElementBase, #192 `_materialize` split, #143 cylinder snap,
    #201/#204 CI/build flags (setup.py changed; no .pyx changed).
- The old branch's frozen-search round-trip fix (`d826182f`) and its `resumed`/`resumeConnectivity`
  plumbing are obsolete upstream.

## Strategy: transplant, not commit-by-commit rebase

The old history contains states that were later undone (hooks, then partition; distributed element
added, then removed). Replaying it would resolve NED conflicts ~10 times against vanished code. Instead,
port the **final design** onto `804d3817` as a few reviewable commits on `feat/mpi-explicit-dynamics-v2`
(worktree `EdelweissFE-mpi2`).

## Workstream R: new references (worktree `EdelweissFE-upstream`, detached at 804d3817)

1. Build the extensions there (`python setup.py build_ext --inplace -j 16`, conda env next_v2611_mpi;
   check `ldd … | grep Marmot` → `$CONDA_PREFIX/lib/libMarmot.so.1` and `edelweissfe/built_extensions.log`).
2. Generate `mpi_explicit_benchmarks/refactor_reference_804d3817/` with the same content and file names
   as `refactor_reference/`: `decks/*.npy` (the 15 serial NED decks + `NEDInitialStressPressure` from
   the old branch's `testfiles/mpi/marmot/`, solver NEDMPI→NED; same procedure as `verify_refactor.sh`'s
   `decks serial`), `deck__fieldmat.npy` (+`fieldmat.inp` copy), the c1_150 baseline (NEDParallel 32
   threads, `c1_150/verify/verify.sh` with the upstream worktree) saved as `c1_150_baseline.npy`, the
   checkpoint at 250 (`c1_150_ckpt250_NED.h5`) and the straight run to 350
   (`c1_150_straight350_UPV.npy`) — see the `c150` function in `verify_refactor.sh` for the deck edits,
   but run them with NED (NEDParallel, 32 threads), not NEDMPI.
3. Compare against the old references: which are bitwise equal, which changed (e.g. restart-related),
   and whether the c1_150 kit deck still runs unchanged on the new upstream (options, restart format).
4. If anything in the kit or harness must change for the new upstream, change the kit/harness (not the
   code). Report the timing of NEDParallel 32 threads (new reference for the perf gate).
Rabbit: the c1_150 runs use 32 cores; workstream T runs only `quick` until R reports.

## Workstream T: transplant (worktree `EdelweissFE-mpi2`, branch `feat/mpi-explicit-dynamics-v2`)

Read the old branch's final code (`git -C ../EdelweissFE-mpi show HEAD:<path>`, `git diff 6e08a2d7
b06b63c0`) and **re-express it on the new upstream code**, adapting to the new structure rather than
forcing the old one back. Commits, each self-contained and verified:

1. **Serial: one partition and one element loop.** `ModelPartition`/`WholeModel`, the single element
   loop, NED and NEDParallel through it. Must be bitwise to upstream NED/NEDParallel (serial decks,
   fieldmat, c1_150 32 threads) and not slower (c1_150 NEDParallel 32 threads within ±3 % of R's
   timing). Adapt to: the fused velocity update (#173; restrict its per-DOF formula to the DOFs
   integrated here without breaking bit-identity), load assembly now in `NonlinearSolverBase` (#191;
   the partition's `loadedElements` filtering must go through the shared code, not a copy),
   `TopologyUpdate` (#191), `model.topology` (#194/#195), the new increment loop / time-stepper state
   machines (#199).
2. **MPI: the domaindecomposition package, Subdomain, NEDMPI**, driver/registry/test-runner/field-output
   changes, `FEModel.advanceToTime(elements=, constraints=)`, the MPC helpers (`slaveMasterDofPairs`,
   `foldExplicitForceOperator`), tests, `testfiles/mpi/`. Adapt to the restart redesign: what
   `getRestartData`/`setRestartData` now do on a constraint replica (#198 adopts during restore — check
   whether `ModelStateSynchronization` calling `setRestartData` on non-owner copies now changes their
   footprint, and make the synchronization correct by construction); checkpoint writing in the new
   loop (rank 0 writes the whole model; constraint states of non-owners must be the owners'); resume via
   `step.isResumed`; `TopologyUpdate` flags agreed across ranks; the owner-only connectivity searches (A6).
   Drop what upstream made obsolete (round-trip staging, resumeConnectivity plumbing). Regenerate
   `testfiles/mpi/**/U.ref` only by copying the serial upstream references (every mpi deck's U.ref must
   equal its serial counterpart's; NEDInitialStressPressure: from a serial NED run).
3. **Docs**: `domaindecomposition.rst`, solvers/NED docs, restart docs if NEDMPI interacts with it.

Do not port: `DistributedMarmotElement` (removed), `MarmotElementWrapper.getResultLocation` (only served it),
the d826182f round-trip, any Cython change. The `.pyx` must stay identical to upstream.

## Gates (both: report the exact harness lines)

- `mpi_explicit_benchmarks/tools/verify_refactor.sh <wt> <label> quick|full` with
  `REF=…/refactor_reference_804d3817 BASELINE=…/refactor_reference_804d3817/c1_150_baseline.npy`
  (REF/BASELINE env vars; until R is done, compare against a run of the upstream worktree). Every line
  BITWISE. `full` only after R reports, at most twice, never concurrently.
- Full serial suites (edelweiss-only, marmot) and `pytest tests/` must give the same results as the
  upstream worktree (run both; known: CDPPlane fails upstream; pure-Python materials race at >1 thread).
- black -l 120, isort --profile black, flake8 (E203,E501,W503) on changed files.

## Rules

As in PLAN_MPI_REFACTOR.md "Rules for both": teaching-oriented names, numpydoc docstrings, no
hasattr/getattr/setattr, no duplication, polymorphism over type tests, elements stay
parallelization-agnostic, Conventional Commits with the two footer lines, stage by name, **no push**,
always `PYTHON_GIL=0 MKL_CBWR=AUTO,STRICT`, never OMP_PROC_BIND/OMP_PLACES, `--bind-to none` under mpirun.
Run code with `PYTHONPATH=<worktree>` from inside the worktree and check `edelweissfe.__file__`.
If an upstream change makes part of the old design wrong or unnecessary, follow upstream and report.
Stop and report rather than force anything that would cost bit-identity or performance.
