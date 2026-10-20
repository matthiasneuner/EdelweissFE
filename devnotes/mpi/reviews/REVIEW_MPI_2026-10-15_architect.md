# Architect's review: distributed-memory explicit dynamics (NEDMPI) in EdelweissFE

Date: 2026-10-15. Reviewed: `EdelweissFE-mpi2`, branch `feat/mpi-explicit-dynamics-v2`. The worktree was
fast-forwarded from 1ac71f9d to **728930ca** (P6 round 1: mesh owns nodes, `GlobalDofNumbering`,
`FEModel.moveNodes`) while this review was in progress; the working tree at 728930ca is what was read.
None of the eight P6-round-1 commits touch `edelweissfe/domaindecomposition/` or
`nonlinearexplicitdynamicmpi.py`; they add `numerics/globaldofnumbering.py` (read, included below).
The earlier `REVIEW_MPI_*.md` files were deliberately not read.

What I ran (env `next_v2611_mpi`, mpi4py 4.1.2, Open MPI 5.0.11, NumPy 2.5.2, `PYTHON_GIL=0`,
`OMP_NUM_THREADS=1`, `MKL_CBWR=AUTO,STRICT`): nine decks of `testfiles/mpi` (NEDContact, NEDSurfaceContact,
NEDSurfaceToDiscreteRigidBodyContact, TieNED, NEDRestart1Write, NEDWholeModel, NEDRebalanceDistributed,
NEDLiveAMRRebalanceTieDistributed, NEDRestartDistributed1Write) serially with `NED` (`--create`), then
with `NEDMPI` on 2 and 3 processes at `--tolerance 0`, plus `compare_checkpoints.py`. All 18 runs
bit-identical, all three checkpoints SAME dataset by dataset. No long runs, no c1_150.

---

## 1. Verdict

**Would one build it this way from scratch? The execution layer, yes; the data layer, no — and the
owner already knows that (P6).**

The part that decides what happens in an increment is what a clean-sheet design under the stated
constraints (bitwise == serial for any P, elements agnostic of parallelism, one rank per constraint,
rank-0 I/O, explicit only) would also arrive at:

* subdomain = the degrees of freedom a process's elements and constraints touch, closed over the MPC
  groups; interface DOFs integrated redundantly, no persistent ghost layer (`subdomaininterface.py:81-141`,
  `subdomain.py:344-383`, `:413-450`). For a lumped explicit scheme this is the right choice: the
  kinematic update is trivially cheap and redundancy removes a whole class of ghost-update bugs;
* interface completion by exchanging *individual element contributions* tagged with (DOF, model
  position) and summing them per DOF in model order (`_TaggedContributions`, `InterfaceForceAssembly`,
  `subdomaininterface.py:209-397`). This is the only way to get the same bits as the serial left fold
  for an arbitrary element numbering; it costs roughly (elements per interface node) times the message
  volume of a partial-sum exchange, which is the price of the bit-identity constraint and is cheap in
  absolute terms (2.5-3 MB/increment reported for c1_150);
* loads and constraint forces travel the same way, point to point, added in deck/model order
  (`InterfaceLoadAssembly`, `ConstraintForceExchange`);
* owner = lowest integrating rank; quantities counted once (external work, kinetic energy) are owned
  (`subdomaininterface.py:108-120`); the external work is made order-independent by an exact integer
  sum (`numerics/exactsum.py`) rather than by communicating every increment — elegant and correct;
* migration by re-creating elements from the replicated mesh and shipping only state vectors
  (`distributedelements.py:253-349`) is exactly how a clean-sheet design would do it, and it reuses the
  checkpoint contract as the element-portability contract — that is the right invariant to lean on;
* the failure protocol — a communicator that *refuses* to communicate inside an agreement region, so
  that a rank failing alone cannot leave the others blocked in a collective (`communicator.py:82-124`,
  `:444-512`) — is better engineering than what most production FE codes do (abort on any error). It
  turns the deadlock-by-omission class of bugs into loud errors at the call site.

What a clean-sheet design would not do is what the migration path from a serial code forced:

* every process holds the whole mesh, every `Node`, every node field, the whole `DofManager`,
  global-length `U/V/P`, the lumped operators at every DOF, every contact facet, every rigid body, a
  copy of every constraint, and the whole AMR mirror. Memory per process is O(N_global) + O(N_local),
  and — less visible in the docs — **work per process is O(N_global) at every rebuild of the equation
  system**, which happens at every topology change *and every contact search that changes a footprint*
  (see finding F1). That is a strong-scaling wall and makes weak scaling impossible, independent of the
  increment loop, which scales fine;
* partitioning is serial METIS on rank 0 with a pure-Python connectivity build, repartitioning from
  scratch with a renumbering heuristic to limit migration (F2, F6);
* I/O, checkpointing and field-output gathering are funnelled through rank 0 with pickled collectives
  (F5).

Soundness: I found **no deadlock, divergence or silent-staleness defect in the paths I read and ran.**
The invariant chains that protect against rank divergence are asserted (layout fingerprint, constraint
copy digests, `requireWholeSolutionHere`, `requireCoupledNodesAmongNodesRead`, the poison option) rather
than structural; they are loud, which is the right second-best. The real risks are (a) the exception
classification inside agreements turning crashes into "failed step, exit 0" (F3), (b) a mid-run
`RuntimeError` from the partitioner's empty-part check that kills a production run instead of falling
back (F6), (c) un-agreed failures during model setup hanging the other ranks (documented, F4), and (d) a
growing maintenance surface of replicated-but-stale copies (constraints, rigid-body surfaces) whose
consistency depends on call ordering inside `acceptIncrement` (F8).

---

## 2. Clean-sheet comparison, area by area

| Area | As built | Clean sheet | Change? |
|---|---|---|---|
| Data distribution | replicated mesh/nodes/DofManager/vectors; distributed element objects | local nodes + local numbering with global ids as data; replicated only the compact mesh description | **now** — P6 is exactly this; `GlobalDofNumbering` (728930ca) is the right primitive |
| Ghost/interface | no ghosts; interface DOFs integrated redundantly; `ValuesFromOwners` for the few read-only copies (search positions, RP nodes) | same | never |
| Communication pattern | p2p Isend/Irecv + Waitall per exchange, no overlap; 1-2 global `Allreduce` per increment for the failure agreement | post receives, compute interior, compute boundary, wait; no global collective per increment | later (small gain at P ≤ 32, 80 ms increments) |
| Ordering for bit-identity | lexsort-by-(DOF, model position) once, `np.bincount` per increment, relying on NumPy's sequential implementation | own sequential scatter-add kernel (Cython), so the guarantee is the code's own | **now** (cheap, removes an external implementation dependency the whole correctness claim rests on) |
| Load balancing | measured kernel times, shares beside constraints, serial METIS from scratch + `keepElementsWhereTheyWere` | diffusive/adaptive repartitioner with migration in the objective (ParMETIS `AdaptiveRepart`, Zoltan) | later; the gain check already prevents thrashing |
| Migration | state vectors via `alltoall` (pickled), re-create from mesh | same, buffer-based | later (pickle → buffers when states get large) |
| AMR | replicated topology + Python-object mirror on every rank | distributed forest (p4est-style) or at least compact arrays | later (P6.4 compaction first; distributed forest only if > ~5M roots) |
| Contact | whole constraint on one rank, owner integrates the whole footprint | partition contact by slave points, master facets ghosted | later (owner's note: beyond ~64 procs) |
| Failure handling | agreement regions; any `Exception` → `StepFailedOnAllRanks` | same mechanism; distinguish step failures from crashes; agree also during setup | **now** (F3, F4) |
| Output/checkpoint | rank-0 gather (Gatherv for vectors, pickled gather for states/results); serial HDF5 | collective HDF5 (h5py mpio) writing the identical layout | later (keeps format identity; only needed when rank 0's transient memory or I/O time bites) |

---

## 3. Findings, ranked

Severity: **S1** blocks scaling or hides failures; **S2** robustness/maintenance; **S3** polish.
Status: *confirmed* = read in code (and run where noted); *plausible* = inferred, not reproduced.

### F1 (S1, confirmed) — O(N_global) work per process at every equation-system rebuild

Every rebuild — `NED.buildEquationSystem` with or without `previous` — calls `partitionModel` →
`Subdomain.define` → `_defineInterface` → `SubdomainInterface.__init__`, which:

* `Allgatherv`s the subdomain DOF lists of *all* processes to *every* process and `lexsort`s them
  (`subdomaininterface.py:89-115`): O(Σ|subdomain DOFs|) ≈ O(N_global) memory and O(N log N) time per
  process, per rebuild;
* allocates several `nDof`-length arrays (`touched`, `owner`, `isLocalDof`, `_gatherBuffer`, `:96-126`);
* `_closeOverMultiPointConstraints` builds an nDof×nDof COO and runs `connected_components` (`subdomain.py:442-450`);
* on `topologyChanged`, `layoutFingerprint` iterates every element, node and field in Python
  (`subdomain.py:1240-1288`), and `_nodeFieldSlotsOf` does `np.arange(nDof)` per field
  (`nonlinearexplicitdynamicmpi.py:1265-1278`).

Rebuilds happen at every topology change *and* at every periodic contact search whose outcome changes a
footprint (`nonlinearexplicitdynamic.py:797-810`), i.e. potentially every `contact-update-frequency`
increments in a sliding-contact run. The docs report 0.2 s per output increment on c1_150 at 8 and 32
processes — note it does *not* fall with P, which is this effect. This, not the increment loop, is the
strong-scaling wall. **Change now, as part of P6.3**: the interface should be built from the
neighbour-local DOF sets (each process already knows its own touched DOFs; the owner of a DOF can be
derived from the replicated `GlobalDofNumbering` plus the element partition without a global gather),
and the fingerprint should digest the replicated *data* (`GlobalDofNumbering.digest()` already exists,
`globaldofnumbering.py:159-175`) instead of walking Python objects.

### F2 (S1 for large meshes, confirmed) — serial partitioner with Python connectivity build

`partitionElementsOfMesh` builds `eptr/eind` in a Python loop over every node of every element on rank 0
(`partitioning.py:129-132`), calls `METIS_PartMeshDual` serially, and `Bcast`s. Repartitions gather a
pickled `{elementNumber: cost}` dict to rank 0 (`subdomain.py:1011-1017`). Fine to a few million
elements; a wall beyond. **Later** (ParMETIS / array-based connectivity from the mesh arrays P6.1 now
provides). Also note `keepElementsWhereTheyWere` (`partitioning.py:194-264`) is a renumbering band-aid for
what an adaptive repartitioner does natively; the doc's own limitation ("even a small imbalance can move a
fifth of the elements") is this.

### F3 (S1, confirmed) — any exception inside an agreement becomes a "failed step" with exit status 0

`Communicator.allRanksFailTogether` catches `Exception` (`communicator.py:476`) and raises
`StepFailedOnAllRanks` for everything that is not a cutback or conditional stop (`:509-512`). The driver
treats that as a graceful step failure (`inputfiledrivensimulation.py:310-312`) and returns normally; no
`sys.exit` anywhere sets a non-zero status. Serially, a `KeyError`/`AttributeError`/`MemoryError` in an
element kernel propagates as an unhandled exception (`:320-323`) and the process exits non-zero; under
MPI the same bug ends the job "normally". A job script or CI step cannot tell a crash from a material
failure. **Now**: inside the agreement classify `StepFailed`-family exceptions as step failures and
everything else as a crash-on-all-ranks (a distinct exception the driver re-raises after the collective
cleanup), so that exit status semantics equal the serial ones.

### F4 (S2, confirmed, documented) — un-agreed failures before the first step hang the other ranks

Documented in the Limitations (`domaindecomposition.rst:909-913`). `fillFEModelFromInputFile`,
`prepareYourself`, restart restore and the field-output/output-manager setup run outside any agreement
(`inputfiledrivensimulation.py:137-232`); an exception caught by a caller — the test runner catches
`Exception` and goes on (`_run_tests_edelweissfe.py:185-188`) — leaves the others in their next
collective (first one: `partitionElementsOfMesh`'s `Bcast`, or `SubdomainInterface`'s `Allgatherv`).
**Now, cheap**: wrap the setup phase of the driver in `allRanksFailTogether` once the communicator exists
(the partition `Bcast` is the only communication in setup and can be moved after the agreement), and make
the test runner abort all ranks on a per-test failure under a launcher.

### F5 (S2, confirmed) — rank-0 funnels with pickled collectives

Element states (`subdomain.py:903-909`, `gather` of a dict of arrays), field-output results
(`distributedelements.py:509-510`), migration states (`distributedelements.py:294-298`, `alltoall`),
constraint states (`statesynchronization.py:198`) and the external work (`nonlinearexplicitdynamicmpi.py:836`)
all use mpi4py's pickle path. Rank 0 transiently holds every element state of the model at a checkpoint
— by design (D2) equal to a serial run. Costs: pickling O(size) on both ends, one extra copy, and
(*plausible*, not tested) the 2 GiB-per-message ceiling of the pickle path unless mpi4py's large-count
support is active for this MPI. **Later**: buffer-based `Gatherv` with per-rank counts, as
`ModelStateSynchronization.synchronizeElementStates` already does (`statesynchronization.py:158-176`);
collective HDF5 only if rank 0's transient memory becomes the limit.

### F6 (S2, confirmed) — a partitioner artefact kills a production run mid-step

`partitionElementsOfMesh` raises `RuntimeError` when METIS leaves a part empty under unequal shares
(`partitioning.py:154-160`). It is reached from `Subdomain.rebalance` at an output increment
(`subdomain.py:1037`), outside any agreement — the exception is raised on every rank identically (same
broadcast partition), so no hang, but the run ends because of a partitioner heuristic. The three magic
constants (`MINIMUM_SHARE_OF_AN_EQUAL_SHARE`, `ELEMENTS_PER_PROCESS_FOR_SHARES`, `INTERCHANGEABLE_SHARES`,
`:60-72`) are empirical guards around the same behaviour. **Now, cheap**: on an empty part, fall back to
equal shares (or keep the current partition), log at level 1, and continue; keep the raise for tests via
an option.

### F7 (S2, confirmed) — the bit-identity of the whole scheme rests on `np.bincount` being a left fold

`ElementPlan.assembleInto` (`parallelelementcomputation.py:225`) and `InterfaceForceAssembly.assemble`
(`subdomaininterface.py:395-397`) sum contributions with `np.bincount(weights=...)`; the docs and tests
acknowledge that this is an implementation detail, not a contract (`domaindecomposition.rst:271-279`), and
pin it with a test and a NumPy version. NumPy could vectorize or parallelize `bincount` in a future
release and every bitwise claim would fail at once — loudly in CI, but at the wrong moment. The
repository already builds Cython extensions; a 20-line sequential scatter-add kernel removes the
dependency. **Now** (cheap; it also removes the `minlength`/`positions` indirection).

### F8 (S2, confirmed) — replicated copies that are stale by design, protected by call ordering

Three kinds of replicated objects are intentionally stale between synchronizations:

* every non-owner's copy of a constraint keeps the footprint of the last sync (`statesynchronization.py:46-53`);
* every process moves the rigid-body surfaces from its *own* possibly stale RP displacement in
  `acceptIncrement` (`nonlinearexplicitdynamic.py:874` via `NEDMPI.acceptIncrement:905`), corrected by
  `_receiveRigidBodyReferences` only at output increments (`nonlinearexplicitdynamicmpi.py:1575-1578`);
* the node fields outside the integrated slots.

The guards (`requireConstraintCopiesCurrent`, the layout fingerprint that digests node coordinates incl.
rigid-body surface nodes, `requireWholeSolutionHere`, the poison option) are loud, and the ordering in
`acceptIncrement` (sync → rigid bodies → rebalance → migrate → full rebuild → fingerprint) makes the
fingerprint pass. But the correctness of the fingerprint check after a migration depends on a migration
*only ever* happening right after a sync; nothing structural prevents a future caller from migrating
elsewhere. A constraint whose footprint can change but which returns empty `getRestartData()` would not
be synchronized at all (`statesynchronization.py:195-196`) — caught by the digest, but as a confusing
"processes disagree" error. **Later (P6.5 direction)**: do not hold copies at all — non-owners hold no
constraint object, and rigid-body surfaces are moved only where read. Until then, state the two
invariants as assertions next to the code that relies on them (a `requireJustSynchronized` flag set by
`_synchronizeModel` and consumed by `_moveElements`).

### F9 (S2, confirmed) — constraint ownership shapes subdomains non-geometrically

A constraint is owned whole by one rank chosen round-robin by sorted name (`partitioning.py:360-378`),
and the owner integrates every DOF the constraint touches plus their MPC closure
(`subdomain.py:356-362`). A large contact surface spanning many subdomains makes its owner a neighbour of
all of them, and `ConstraintForceExchange` sends to all of them every increment. The load balancer
compensates (shares beside constraints), but the attainable time is bounded below by the costliest
constraint and the owner's element share has a floor of half an equal share (`partitioning.py:311-320`):
beyond P ≈ T_elements / (2 T_contact) no further speedup is possible. The owner's decision (C1-C3 deferred
beyond ~64 processes) is reasonable for the current target. **Later.** Two cheap improvements now: assign
constraints by the rank that owns most of their DOFs (reduces the owner's interface and the force
traffic) instead of by name; and report, at level 1, the interface growth a constraint causes.

### F10 (S3, confirmed) — no overlap of communication and computation; global sync per increment

Each exchange is Isend/Irecv immediately followed by `Waitall` (`subdomaininterface.py:326-337`,
`:640-646`); nothing is computed in between. The agreement's `Allreduce` runs on every increment before
the interface exchange (`nonlinearexplicitdynamicmpi.py:664`, `:734`), converting neighbour-wise waiting
into global waiting. The contribution-buffer design makes overlap easy (interface entries are known;
order boundary chunks first, start the exchange, compute interior). **Later** — on one node at P ≤ 32
with 80 ms increments the gain is a few percent; on multi-node runs it matters.

### F11 (S3, confirmed) — one message tag for three successive exchanges

`_INTERFACE_TAG = 7` serves the element-force, load and constraint-force exchanges between the same
pairs within one increment (`subdomaininterface.py:57`). Correct today because every exchange `Waitall`s
before the next starts and MPI guarantees non-overtaking; a future overlap (F10) would silently cross
messages. Distinct tags cost nothing. **Now, trivial.**

### F12 (S3, confirmed) — threads × processes

`NEDMPI` inherits `NEDParallel`'s threaded element loop; MPI calls are issued from the main thread only.
Sound. The binding guidance in the docs (`--map-by slot:PE=n`) is correct and necessary.

### F13 (S3, plausible) — `PYTHONHASHSEED`

Documented (`domaindecomposition.rst:594-599`): model construction order from a `set` of strings differs
between ranks unless the seed is fixed; the fingerprint catches it at the first build. A clean-sheet
design forbids set-ordered construction. Cheap mitigation: at start-up, `requireSameOnAllRanks` on a
digest of the generated model *before* any element is created, so the error names the generator rather
than "the processes hold different models".

### F14 (S3, confirmed) — the test-time harness

The CI workflow (`.github/workflows/run_tests_mpi.yml`) does the right thing: fresh serial references on
the runner, `--tolerance 0` at 2 and 3 processes, checkpoint comparison, poison run, distribution check.
The pytest file covers the ordering claims with adversarial values and the failure paths with its own
`mpirun`. I reproduced the bitwise claim on 9 decks (see top). The only gap: nothing runs at P ≥ 4 or
with threads > 1 under MPI in CI (the docs claim both); one such job would close it.

---

## 4. Recommendations, prioritized

**Now (cheap, before the branch is merged or P6.3 lands):**
1. F3 — distinguish step failures from crashes inside `allRanksFailTogether`; non-zero exit on crash.
2. F4 — agreement around model setup in the driver; abort-all in the test runner under a launcher.
3. F6 — fall back instead of raising when the share partition leaves a part empty.
4. F7 — own sequential scatter-add kernel instead of `np.bincount`.
5. F11 — distinct tags per exchange.
6. F8 — assert "migration only right after a synchronization" explicitly.

**Now, as part of P6.3 (the design is right; this is scope guidance):**
7. F1 — build the interface from neighbour-local data, not an `Allgatherv` of all subdomain DOFs; digest
   `GlobalDofNumbering` and the mesh arrays for the fingerprint, not Python objects. Without this, P6 fixes
   memory but not the rebuild-time wall.
8. F9 — assign constraints by DOF majority, not by name.

**Later (when the target moves beyond ~64 processes or ~5M elements):**
9. F2 — adaptive repartitioning (ParMETIS/Zoltan) and array-based connectivity.
10. F5 — buffer-based gathers; collective HDF5 keeping the serial layout.
11. F10 — overlap communication with the interior element chunks; drop the per-increment global agreement
    in favour of a status word carried with the interface messages.
12. Contact partitioning (C1-C3), as the owner already scheduled.

**Never (deliberate, and correct under the constraints):**
* exchanging individual contributions instead of partial sums;
* redundant integration at the interface instead of a ghost layer;
* re-creating migrated elements from the mesh instead of pickling objects;
* rank-0 I/O with the serial format (until F5's limits are actually hit).

---

## 5. Maintenance-burden note

`NEDMPI` overrides 15 methods of `NED` and relies on NED's internal call order (constraints evaluated in
`assembleInternalForces`, added in `assembleConstraintForces`; `partitionModel` reached from inside
`buildEquationSystem`; `acceptIncrement`'s ordering). The guard at
`nonlinearexplicitdynamicmpi.py:779-784` and the `ModelPartition` abstraction keep this honest, and the
docs' method table (`domaindecomposition.rst:222-245`) is the contract. That is an acceptable coupling for
a teaching code whose serial reading path must stay clean; the alternative — an explicit increment
pipeline with hook points — would move complexity into NED. Keep the table in sync with the overrides; it
is the first thing the next person will read.
