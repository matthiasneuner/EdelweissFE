# Plan: make the MPI explicit solver fit the teaching-oriented EdelweissFE code (2026-10-02)

Base: `feat/mpi-explicit-dynamics` at `d9d1315c` (rebased onto upstream `6e08a2d7`, verified bitwise;
see HANDOFF_SESSION_2026-10-02.md §5b). Machine: rabbit. Nothing is pushed; do not push.

## Why

A review found the NEDMPI work numerically sound (bit-identical to NED on any number of processes,
by construction) and its new package `edelweissfe/domaindecomposition/` well written, but grafted onto
the solver at too many points:

1. NED (the serial teaching solver) gained ~15 "subdomain hooks" (`nonlinearexplicitdynamic.py`,
   `# --- Subdomain hooks`), each an identity in NED. Its increment now reads `_integratedDofs()`,
   `_globalSum(psi)`, `_ownedDofMask`, `_sumOfOwnedSquaredRates`; `_synchronizeSubdomains` is called at
   5 sites. A reader learning central differences must understand subdomains.
2. Duplicated code: `planSubdomainElements`/`computeSubdomainElementsForExplicit`
   (`solvers/base/parallelelementcomputation.py`) re-implement the chunk/gather of
   `computeElementsInParallelForExplicit`; NEDMPI's `_assembleLumpedDiagonal` repeats the chunk loop;
   NEDMPI's `_publishIncrementToNodeFields` re-implements `writeDofVectorToNodeField`.
3. A hidden class swap: `config/elementlibrary.py` returns `DistributedMarmotElement` for
   `elProvider=marmot` depending on the process count and env var `EDELWEISSFE_DISTRIBUTED_ELEMENTS`.
4. `DistributedMarmotElement` is a proxy rather than an element: ~45 members without docstrings,
   module-level mutable globals (`_transientKernels` LRU, prototypes, a counter), and its detached
   `getResultArray` re-derives Marmot's per-quadrature-point state layout (`block = len // nQP`).
5. NEDMPI internals: caches as positional tuples validated by `is` and read as `layout[2]`,
   `cached[2]`; `_loadedElementsOfThisProcess` attaches elements as a side effect of a query; load
   rebalancing hidden inside the output hook `_finalizeOutput`.

(The `isinstance(…, DistributedMarmotElement)` switches are already gone: commit `d9d1315c`.)

## The contract between the two workstreams (already on the base, do not change its meaning)

`BaseElement` and the Cython `MarmotElementWrapper` (which follows the interface without deriving):

- `computingObject` (property): what computes the element's kernels; the element itself by default.
- `startComputingHere()`: prepare to compute the element in this process; no-op by default.
- `stopComputingHere() -> object | None`: stop computing it here, keep its converged state; return
  what was released (a caller keeps it alive while views may exist); None by default.

`DistributedMarmotElement` implements them (attach/detach of its Marmot element). The solver side
uses only these and `domaindecomposition.distributedelement.releaseTransientKernels()`; keep that
function's name and meaning (the elements workstream may re-implement it).

## Workstream P: "partition" (worktree `EdelweissFE-mpi-part`, branch `refactor/mpi-part`)

Owns: `solvers/nonlinearexplicitdynamic.py`, `solvers/nonlinearexplicitdynamicmpi.py`,
`solvers/nonlinearexplicitdynamicparallel.py`, `solvers/base/parallelelementcomputation.py`,
`domaindecomposition/{subdomaininterface,statesynchronization,partitioning}.py` (+ any new module
there or in `solvers/base/`), `doc/source/documentation/domaindecomposition.rst` and NED's docs,
`tests/test_domaindecomposition.py`. Do NOT edit element files, `config/elementlibrary.py`, the driver.

Goals:
1. **Replace the hooks by one concept.** NED holds one object describing which part of the model this
   process computes and how its partial results become results of the whole model (working name
   `ModelPartition`; a serial default such as `WholeModel` computes everything, sums are plain sums,
   synchronization is nothing). NED's increment must read as the central-difference method plus one
   clearly introduced partition concept, in existing FE vocabulary. The partition is introduced once,
   documented once; the serial implementation is trivial and obviously so. NEDMPI ideally reduces to
   its option schema plus constructing the MPI implementation of the partition. Collapse hooks where
   the concept allows (e.g. reductions into one method; "complete the force across the partition").
   Remove the `# --- Subdomain hooks` block from NED.
2. **One explicit element loop.** A single chunk/gather/assemble path serves NED, NEDParallel and
   NEDMPI (e.g. serial = one partition covering all DOFs). NEDMPI at one process is already bitwise to
   NED through the subdomain path, so this is attainable bitwise. Keep `NEDParallel` as a registered
   solver name (decks use it); it may become trivial. No lumped-operator or node-field duplication.
   **No performance regression**: c1_150 NEDParallel 32 threads stays at ~0.104 s/increment (±3 %),
   NEDMPI 8x4 at ~0.091 s/increment (rabbit idle; see the harness `full` stage).
3. **Readable state.** Named small dataclasses instead of positional tuple caches; no query with side
   effects (split "which elements load my DOFs" from "start computing them"); load rebalancing as an
   explicit, visible step of the increment loop, not inside output finalization.

## Workstream E: "elements" (worktree `EdelweissFE-mpi-elem`, branch `refactor/mpi-elem`)

Owns: `domaindecomposition/distributedelement.py`, `elements/base/baseelement.py`,
`elements/marmotelement/element.pyx`/`.pxd` (if needed; rebuild in the worktree with
`python setup.py build_ext --inplace -j 16`, keep `edelweissfe/built_extensions.log`, check `ldd` links
`$CONDA_PREFIX/lib/libMarmot.so.1`), `config/elementlibrary.py`, `drivers/inputfiledrivensimulation.py`,
`domaindecomposition/mpienvironment.py`, element-related docs (the "Memory"/distributed-element parts of
`domaindecomposition.rst` -- coordinate by keeping edits to clearly separate paragraphs), a new
`tests/test_distributedelement.py`. Do NOT edit the solver files listed under P.

Goals:
1. **Docstrings** on every member (numpydoc, as the rest of the code). Short is fine where the member
   just forwards: say what it answers and where the answer comes from (attached / transient / replica /
   prototype).
2. **No module-level mutable globals.** Encapsulate the transient-element LRU, the prototypes and the
   counter in one documented object. If the element-construction interface (`cls(elType, elNumber)`)
   makes a per-model instance impossible without touching generators, one module-level instance of a
   documented class is acceptable -- say why in its docstring.
3. **Don't re-derive Marmot's state layout.** The detached `getResultArray` must obtain offset/size per
   quadrature point from Marmot (e.g. a method on the wrapper, or from the layout prototype), not assume
   `len(state) // nQP` blocks. A `.pyx` addition is fine.
4. **No hidden class swap.** Remove `EDELWEISSFE_DISTRIBUTED_ELEMENTS`. Under an MPI launcher Marmot
   elements are distributed elements -- make that visible: one journal line on rank 0 at model build
   (driver) and the docs. Keep the decision in one place.
5. A unit test for the distributed element against a plain Marmot element at one process (same
   state/results/lumped mass after setup, attach/detach round trip, transient answers, getResultArray
   views follow the replica after `setStateVars`).

## Rules for both

- **Verify with** `mpi_explicit_benchmarks/tools/verify_refactor.sh <worktree> <label> quick` (~5 min,
  bitwise against frozen 6e08a2d7 references; every line must say BITWISE/PASS). Run it after every
  meaningful step. The `full` stage (c1_150, ~25 min, 64 cores): workstream P only, at most twice, and
  never while another `full` runs. Check `CODE` in its output points at your worktree.
- Always `PYTHON_GIL=0 MKL_CBWR=AUTO,STRICT`; never set OMP_PROC_BIND/OMP_PLACES. Under mpirun use
  `--bind-to none`. env: `conda activate next_v2611_mpi`; run with `PYTHONPATH=<worktree>`.
- Known pre-existing, not yours: pure-Python materials race at >1 thread (the harness compares only
  Marmot decks at 4x2); `CDPPlane` fails upstream.
- Code style: teaching-oriented names in existing FE vocabulary (no "ledger/augment/operator" jargon);
  numpydoc docstrings on everything; no `hasattr/getattr/setattr`; polymorphism over type tests;
  **no duplication** (refactor into a shared function rather than copy); black -l 120, isort --profile
  black, flake8 (ignore E203,E501,W503).
- Commits: Conventional Commits, small and reviewable, on your branch only; end every message with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01QBXKSWbY2FbWH4LpdzNTXK`. Stage files by name
  (never `git add -A`: the worktrees hold untracked build/test outputs). Do not push.
- Time cap: ~3 h wall clock. If a goal turns out to need a different design than described, or would
  cost bitwise identity or performance, stop on that goal and report rather than force it.
- Report back (short): commits, what each goal became, harness output lines, anything left undone.
