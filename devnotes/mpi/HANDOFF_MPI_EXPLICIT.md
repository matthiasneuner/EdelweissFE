# Handoff: NEDMPI, a domain-decomposed explicit dynamic solver (prototype)

Sessions of 2026-09-30 (the solver) and 2026-10-01 (distributed elements, §9), on rabbit,
unsupervised. Everything is committed locally; **nothing is pushed** and no jobs are running.

---

## 1. State in one table

| Thing | Where | State |
|---|---|---|
| Code | worktree `EdelweissFE-mpi`, branch `feat/mpi-explicit-dynamics`, 10 commits on `origin/next_v26.11` @ `da9aab63` (HEAD `e7aafe7e`) | local only, not pushed |
| Environment | conda env `next_v2611_mpi` (clone of `next_v2611` + `mpi4py 4.1.2`, conda-forge `openmpi 5.0.11`); EdelweissFE editable-installed from the worktree | ready |
| Benchmarks | `mpi_explicit_benchmarks/` (decks, scripts, `strong_scaling/results.csv`, tools) | see §5 |
| Docs | `doc/source/documentation/domaindecomposition.rst`, solver reference, parallelization page | Sphinx builds with no warnings from the new or changed files |

The main `EdelweissFE` checkout and the `next_v2611` env are untouched. `pymetis` has no build for
free-threaded 3.14 on conda-forge, so METIS is bound directly through `ctypes` (`libmetis.so`, which
the env already has for SuiteSparse). ParMETIS is not needed (see §3).

## 2. Design in one paragraph

Every process builds the complete model from the same deck and computes one **subdomain**: its own
elements (METIS `PartMeshDual`, communication-volume objective, on rank 0, broadcast) and its own
constraints (dealt out by name), plus the DOFs they touch. At the interface, neighbours exchange the
**individual element contributions** (not partial sums), and every process sums all of them at each
DOF **in model element order**, which is what the serial solver does. So NEDMPI is **bit-identical to
NED on any number of processes**. Wherever the whole model is read (field outputs and output
managers, a contact search, a marker and the refinement) the processes first synchronize: vectors
from the DOF owners, and element/constraint states through the restart interface
(`getStateVars`/`getRestartData`). AMR then runs replicated on every process (the layout fingerprint
is checked across processes) and the model is repartitioned; **nothing migrates**. Load balancing
times every element kernel and repartitions with the measured costs; that is free, and it does not
change results.

## 3. METIS / ParMETIS

METIS, serial, on rank 0 only (`edelweissfe/domaindecomposition/metis.py`). Because the model is
replicated, repartitioning after AMR or for load balance needs no migration, so the adaptive
repartitioning of ParMETIS (or Zoltan, PT-Scotch) buys nothing yet. It becomes relevant only with a
*distributed* model (§7). ParMETIS also has no maintained Python binding and a non-commercial
licence.

## 4. Verification

- **NED refactor (P0):** all 15 explicit decks are bitwise identical before and after, with
  `MKL_CBWR=AUTO,STRICT`.
- **`testfiles/mpi/`** holds the 15 explicit decks (contact, surface contact, discrete-rigid-body
  contact, tie, restart write+resume, GCDP, hyperbolic GCDP, initial and live AMR). U.ref is copied
  from serial.
  - Bitwise identical to serial NED at **1, 2, 3 and 4** processes.
  - At 4 processes × 2 threads.
  - Under `mpirun -n 3 run_tests_edelweissfe testfiles/mpi/...`: all pass.
- **GCDP tension bar with live AMR** (`gcdp_live_amr`: 256 → ~6000 GC3D20R elements, 6 live
  refinements with hanging nodes, 7740 increments): reaction history **bitwise identical** to serial
  at 8 and 16 processes, including 16–24 load-balancing repartitions.
- **Restart:**
  - A checkpoint written by NEDMPI (4 processes) is identical, dataset by dataset, to the serial
    one. Only the diagnostic external-work sum differs, in the last digit.
  - Resuming from either checkpoint with NEDMPI on 2 or 3 processes, or with serial NEDParallel,
    gives bitwise-identical runs.
- **Failures:**
  - A cutback request, or any other exception injected in the element evaluation of one process
    only: all processes end the step with the same message, and nothing hangs (`tools/inject_failure.py`).
  - A conditional stop decided on rank 0 stops every process at the same increment, bitwise equal
    to serial.
- **Unit tests:** `tests/test_domaindecomposition.py` (5 tests). The full serial suites were re-run
  at the end; see §8.

## 5. Performance (rabbit, Threadripper PRO 5975WX, 32 cores)

The 192,000-element strong-scaling case is GC3D8 GCDP, 200 increments, ms per increment.
`strong_scaling/results.csv` has everything; all runs are bitwise identical to each other.

| cores | NEDParallel (threads) | NEDMPI (processes × threads) |
|---|---|---|
| 1 | 1238 | 1300 (1×1) |
| 8 | 272 | 287 (8×1) |
| 16 | 161 | 162 (16×1) |
| 32 | 118 | **92 (8×4)**, 89 (4×8), 108 (16×2) |

The live-AMR case (6000 elements), wall time: serial 505 s. Threads: 124 s on 8, 110 s on 16. NEDMPI:
136 s on 8 processes, 103 s on 16.

**Honest reading:**
- On one node NEDMPI is on par with the thread pool, and hybrid is up to 25% faster at 32 cores.
- On this machine the element kernel itself slows 2–3× per element under full load, with threads
  and processes alike, so neither exceeds 10–15× on 32 cores.
- The AMR case is bounded by the refinement, which every process repeats (1.3 s × 16 calls).
- What MPI adds today is several nodes (not tested; §7), bitwise reproducibility at any size, and
  the removal of the serial Python parts that the thread pool cannot touch.

**Memory:** 8.2 GB per process for 192k elements, whatever the process count. That is about 30 KB
per element at mesh construction, of which state vectors are only 1.2 KB. The rest is object
overhead, mostly the Marmot elements. **16 processes = 130 GB. A 32-process run of that model does
not fit on rabbit, so it was not run.**

## 6. Known issues found (pre-existing, not caused by this work)

1. **NEDParallel with a pure-Python material and more than one thread is wrong.** For
   `testfiles/edelweiss-only/NED` (VonMises, `provider=edelweiss`) at 4 threads, the result is off
   by 15% and differs from run to run. NEDMPI with threads inherits it. Marmot materials are fine
   (bitwise at 4×2).
2. **Serial NED restart with live AMR:** the resumed run deviates 1.27e-3 (relative) from the
   uninterrupted one, and its first resumed reaction-force sample is 0. This is reproduced
   bit-for-bit with the untouched `da9aab63` code (`restart/` decks).
3. `testfiles/marmot/CDPPlane` fails on the untouched baseline too (NIST, private CDP).

## 7. Limitations and next steps, by value

1. **Memory: distribute the model.** Each process should instantiate only its own and its ghost
   elements. For AMR, keep a lightweight replicated topology (connectivity arrays), let owners do
   the state transfer, and send children to their new owners. This is where ParMETIS
   `AdaptiveRepart` would enter. Alternatively, or as well, shrink the ~30 KB per Marmot element.
2. **LEO5.**
   - Single node: `mpirun` inside the Apptainer image, using the image's own OpenMPI over shared
     memory, needs only `mpi4py`, `openmpi` and `metis` added to the image. That is the first thing
     to try.
   - Several nodes: this needs the hybrid Apptainer model (host PMIx/Slurm with a matching MPI or
     MPICH ABI). It has not been tried, and needs your go-ahead for job submission.
3. **Replicated AMR cost.** The refinement runs in full on every process.
4. **A constraint is evaluated whole by one process.** A single large contact constraint is a serial
   bottleneck, and its forces are shared by a pickled allgather every increment. Split contact by
   slave node, and exchange the indices only once per search.
5. **Leftover O(model) work per increment and process:**
   - publishing the node fields (about 4.6 ms at 192k elements);
   - the U/dU update over the whole vector;
   - the MPC fold (`T.T @ P`).

   All of these can be restricted to the subdomain. Overlapping the interface exchange with interior
   elements is not done.
6. Only NED is decomposed.

## 8. How to run

```console
conda activate next_v2611_mpi
cd EdelweissFE-mpi
PYTHON_GIL=0 OMP_NUM_THREADS=1 mpirun -n 8 --bind-to core edelweissfe deck.inp      # *solver, solver=NEDMPI
PYTHON_GIL=0 OMP_NUM_THREADS=4 mpirun -n 8 --map-by slot:PE=4 --bind-to core -x OMP_NUM_THREADS -x PYTHON_GIL edelweissfe deck.inp
mpirun -n 3 run_tests_edelweissfe testfiles/mpi/marmot
```

Code map:

- `edelweissfe/domaindecomposition/`:
  - `mpienvironment` (launcher detection; mpi4py is imported only then)
  - `metis`
  - `partitioning`
  - `subdomaininterface` (`SubdomainInterface`, `InterfaceForceAssembly`)
  - `statesynchronization`
- `edelweissfe/solvers/nonlinearexplicitdynamicmpi.py`: `NEDMPI`.
- NED subdomain hooks: the `# --- Subdomain hooks` section of `nonlinearexplicitdynamic.py`.
- Driver (only rank 0 writes): `drivers/inputfiledrivensimulation.py`.

Final suite run on HEAD (`95a39de6`, 2026-09-30):

- `testfiles/edelweiss-only`: 0 failures.
- `testfiles/marmot`: 1 failure, CDPPlane, which also fails on the baseline.
- `pytest tests`: 598 passed, 11 skipped.


---

## 9. Phase 2 (2026-10-01): distributed elements — the memory fix

### What was measured first

- A Marmot GC3D8 element takes 26.7 KB at construction, before any material, and the GCDP material
  adds 9.9 KB.
- The converged state is only 1.2 KB of that (2.4 KB counting both buffers), and a node is 0.5 KB.
- So the heavy part is the C++ element itself, and only the process computing an element needs it.

### What was built (`edelweissfe/domaindecomposition/distributedelement.py`)

Under an MPI launcher, `getElementClass(..., "marmot")` returns `DistributedMarmotElement` (it
subclasses `BaseElement`). It holds:

- the topology;
- a replica of the converged state;
- a recorded setup: `setProperties`, `initializeElement`, `setMaterial`, `assignProperty`.

Its Marmot element is built from that record in two situations:

- **Attached.** NEDMPI attaches the owned elements in `_adoptOwnership`, and lazily attaches
  neighbour elements whose distributed or body load reaches into the subdomain.
- **Transient.** One question to a detached element builds a Marmot element for it; an LRU of 64 is
  kept. This covers state-transfer geometry, lumped mass, initial conditions and the critical time
  step.

When detached:

- type-level answers come from one prototype per element type;
- the state layout comes from one prototype per (type, material);
- `getResultArray` returns a view into the replica, which every synchronization refreshes.

`computeKernels*` on a detached element **raises**.

The element starts with a zero state when its material is assigned, as Marmot does. This is
verified when the Marmot element is first built, and a mismatch raises.

### NEDMPI changes

- **Ownership changes:** attach and detach. Then re-pin the field-output result views
  (`FieldOutputController.refreshResultViews`), *then* release the dropped Marmot elements. The
  `ElementResultCollector` holds raw pointers into their states.
- **Hot loop and accept:** both call the attached Marmot elements directly. The plan has
  `computingObjects`.
- **Lumped operators:** assembled from the owned elements only, completed at the interface in model
  order, then gathered. The NED hook `_assembleLumpedDiagonal` is the plain loop in NED.
- **Per increment:** `dU = V dT`, `U += dU`, and publishing to the node fields cover the subdomain
  DOFs only (NED hooks `_advanceSolution` and `_publishIncrementToNodeFields`). This was 33 ms of a
  125 ms increment at 32 processes.

### Verification

- **Bitwise identical to NED:**
  - all 16 `testfiles/mpi` decks at 2, 3 and 4 processes;
  - the GCDP live-AMR bar at 8 processes (6 refinements, 16 rebalances);
  - every 192k-element strong-scaling configuration.
- **New deck `testfiles/mpi/marmot/NEDInitialStressPressure`:**
  - a one-time geostatic initial stress, which reaches detached elements;
  - a surface pressure on neighbour elements;
  - bitwise identical to NEDParallel at 3 processes, and the initial condition really changes the
    solution.
- **Restart with distributed elements:**
  - the 4-process checkpoint matches the serial one dataset by dataset;
  - resuming it on 3 processes is bitwise identical to the serial resume.
- **Unit tests:** 7 in `tests/test_domaindecomposition.py`.
- **Suites:**
  - `mpirun -n 4 run_tests_edelweissfe testfiles/mpi/*`: 0 failures.
  - Serial: edelweiss-only 0 failures; marmot CDPPlane only (fails on the baseline too).
  - `pytest tests`: 600 passed.

### Numbers (192 000 GC3D8 GCDP elements, 200 increments, `strong_scaling/results_distributed*.csv`)

| configuration | peak memory per process, before → now | ms per increment |
|---|---|---|
| 8 × 1 | 8.3 GB → **2.5 GB** | 287 → 292 |
| 16 × 1 | 8.2 GB → **2.0 GB** | 162 → 163 |
| 32 × 1 | did not fit (≈260 GB total) → **1.8 GB** (57 GB total) | **108.7** (32 threads: 118) |
| 8 × 4 | 8.3 GB → **2.6 GB** | 92 → 92 |

Model setup is 2.6 s at 8 processes, against 7.3 s serial: no process builds Marmot elements it does
not compute.

### What is still replicated, per element of the model (about 7 KB)

- generator sets and lists: about 1.5 KB;
- distributed element and state replica: 2.2 KB;
- `prepareYourself` overhead: about 0.7 KB;
- DofManager: 1.1 KB;
- nodes: about 0.5 KB.

On top of that, each process holds about 37 KB per element it owns. A model of 1M elements therefore
needs about 7 GB per process, plus the owned part.

### Next, if more is needed

1. **Distribute the topology itself.** Each process would hold only its own and ghost elements, sets
   and DOFs. AMR then needs:
   - distributed marking and 2:1 balancing (ParMETIS would enter here);
   - migration of distributed elements, which now only carry setup and state, so that part is easy.

   This is a large refactor of hadaptivity and the DofManager.
2. **Cheaper wins:**
   - A per-process DofManager restricted to the subdomain would save 1.1 KB per element.
   - Generators keep several per-element lists; check boxGen and its sets.
   - Each element's recorded setup currently keeps its own copy of the empty properties array; share
     it.
3. `accept state` is still about 1.5 µs per owned element. It could use the thread pool in hybrid
   runs.

---

## 10. Production test (2026-10-01): c1_150 edge breakout, from the LEO5 run

### Setup (`mpi_explicit_benchmarks/c1_150/`)

The source is LEO5 `wanwendner_edge_breakout_c1_150_explicit_amr_h8mm_meshv4_washernut_s2s_dmg02`
(job 7958248). It was copied read-only into `leo5_snapshot/`: the deck, the mesh, `restart_0.h5`
(t = 0.7642, 115,905 elements, 372 topology records) and the log.

`make_run.py` builds a resumed window of 5000 increments:

- no EnSight and no restart output;
- a conditional stop on `model.time`.

`run_unverified.py` resumes with the topology fingerprint check off. The LEO5 container
(`edelweissfe-latest.sif` = EdelweissFE `8e63e8e9`, Marmot `20dc4ad5`) creates refined nodes whose
coordinates differ in their last bits from rabbit's. The refinement plans are identical, but the
fingerprint hashes raw coordinate bytes, so checkpoints don't move between machines with
verification on. A resume replays 372 records in about 30 minutes, and reproduces LEO5's last two
AMR events exactly.

### Correctness

NEDMPI first differed from NED by 2.6e-3 in the reaction force (the displacement and damage
histories are insensitive here).

- **Isolated:** `EDELWEISSFE_DISTRIBUTED_ELEMENTS=0` (a new switch) made it bitwise identical, at
  both 1 and 8 processes.
- **Cause:** a **pre-existing NED bug**. A tie's `adjust=True` (the default) snaps slave nodes
  *after* Marmot elements have copied their coordinates in `setNodes`. In serial NED, 525 of the
  69,166 Marmot elements therefore compute with unsnapped geometry, off by up to 0.50 mm; the nut
  and washer ties use `positionTolerance=0.6`. AMR children and output see the snapped nodes. The tie
  docstring claims the opposite ("adjusting modifies the nodal coordinates before the element
  geometry is initialized"). Still present on `origin/next_v26.11`. **NOT fixed: it changes
  production results.** Script: `c1_150/check_tie_adjust.py`.
- **Fix in NEDMPI** (`f0cc9995`): a distributed element snapshots the coordinates at `setNodes`, as
  the Marmot wrapper does.
- **After it:**
  - from t = 0, 1000 increments: U, P and V are bitwise identical for 1 thread, 32 threads,
    NEDMPI 8×4 and NEDMPI 8×1;
  - resumed window: reaction force, displacement and damage are bitwise identical for 32 threads,
    NEDMPI 8×4 and NEDMPI 32×1.

### Performance, resumed window (6000 increments, 2 AMR events)

| | 32 threads | NEDMPI 8×4 | NEDMPI 32×1 |
|---|---|---|---|
| per increment | 180 ms | **155 ms** | 238 ms (309 before the fixes below) |
| elements, including waiting for the slowest process | 137 ms | 116 ms (7 ms waiting) | 170 ms (55 ms waiting) |
| topology check (replicated in NEDMPI) | 42 s | 39 s | 61 s |
| step | 1537 s | 1456 s | 2383 s |
| peak memory | 14.1 GB | 8.1 GB/process | 6.6 GB/process (211 GB total) |

Fixed along the way (`378955d9`), both bitwise:

- the MPC fold is restricted to the subdomain (38 → 2.2 ms at 32 processes);
- constraint forces are exchanged by `Allgatherv` instead of a pickled allgather.

### What limits it now, in order

1. **Topology checks** (16% for both solvers): mostly "refresh mesh dependents" after a refinement
   (facets and ties rebuilt). Every process repeats it, and it is slower under contention.
2. **Load imbalance at 32 processes** (55 ms per increment), despite measured-cost rebalancing.
   Suspects: GC3D20R and C3D8 cost differences, constraint owners, and timing noise in the measured
   weights.
3. **Replicated memory:** about 6 GB per process for this model (1.94M DOFs, 34,000 facets, 19,600
   hanging-node records, the AMR mirror, contact). It is no longer the elements.
4. On rabbit, fewer processes with more threads (8×4) beats 32×1. LEO5 is still untested.
