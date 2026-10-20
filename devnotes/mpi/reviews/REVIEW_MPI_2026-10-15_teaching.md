# Review: the MPI explicit solver (NEDMPI) read from a student's side

Reviewed: `EdelweissFE-mpi2`, branch `feat/mpi-explicit-dynamics-v2`, working tree at **728930ca**
(the fast-forward during the review that merged P6 round 1: the mesh owns the nodes,
`GlobalDofNumbering`, `moveNodes`). The solvers did not change between 1ac71f9d and 728930ca;
`subdomain.py`, `mesh.py`, `femodel.py`, the sets and `GlobalDofNumbering` did, and were read at the new tip.
Date: 2026-10-15. Independent view; the earlier `REVIEW_MPI_*.md` files were not opened.

What was read, in a student's order: the driver, `inputfilehelpers` (MPI hooks), `models/mesh.py`,
`models/femodel.py`, `models/elementdistribution.py`, `numerics/globaldofnumbering.py`, `sets/*`,
`solvers/nonlinearexplicitdynamic.py` (NED, whole file), `nonlinearexplicitdynamicparallel.py`,
`solvers/base/modelpartition.py`, `solvers/base/parallelelementcomputation.py`,
`nonlinearexplicitdynamicmpi.py` (whole file), all ten modules of `domaindecomposition/`,
`numerics/exactsum.py`, the base-solver hooks NEDMPI overrides, the constraint/model-modifier base
hooks, `doc/source/documentation/{domaindecomposition,mesh,sets,solvers,parallelization}.rst`, the
test list of `tests/test_domaindecomposition.py`, two decks of `testfiles/mpi`, and the MPI workflow.

What was run (env `next_v2611_mpi`, `PYTHONPATH` = this worktree, verified by
`edelweissfe.__file__`, `PYTHON_GIL=0 OMP_NUM_THREADS=1`): `NED`, `NEDEqualValuePenalty`,
`NEDSurfaceContact` from `testfiles/mpi/edelweiss-only`, each once with `solver=NED` serially, once
with `NEDMPI` on `mpirun -n 2`, once on `mpirun -n 3` with `debug-poison-stale-solution=True`.
All six `U.ref` pairs compare byte for byte (`cmp`). The distributed path was taken
(`Distributed model: each of the 2 processes creates only the elements it computes`).

---

## Verdict

**Designed, not bolted on -- but the seams have not been sanded on the serial side.** The
architecture is the one a code meant for domain decomposition would have from day one, and it is
the standard one (deal.II/PETSc shape): the mesh is data and the element objects are made from it;
an `ElementDistribution` null object decides what is local; sets know whether they are complete and
refuse to be read otherwise; the increment runs over a `ModelPartition` that is the whole model
serially; the bulk element loop keeps a per-contribution buffer so interface sums can be formed in
model order; the external work is an exact sum. Bit-identity is not a promise but a mechanism
(ordered folds + exact sums + a layout fingerprint + a poison mode + a CI byte-compare), and it holds
on this tip. Elements know nothing (confirmed: zero MPI vocabulary in `elements/`), and so does the
`DofManager`.

What a student will feel as bolted on is not the mechanism but its *vocabulary and its residue in
serial code*:

* NED's kinematic update is no longer textbook: every hot-path line indexes with
  `self.partition.dofs` (= `slice(None)`), and "integrated here / computed here / evaluated here /
  this process" appears 46 times in NED alone (0 at the merge base). The element loop itself is still
  the plain loop the owner wanted (NED:1449-1460).
* The deictic house style ("here", "held here", "the elements computed here") is used in ~60 serial
  files instead of the standard words (locally owned / ghost / subdomain / replicated), and the one
  mapping to that standard vocabulary is a half sentence in `orderedset.py:207`.
* `FEModel` carries a dozen MPI-motivated members; the driver starts with `isRootProcess()`.
* `NEDMPI` (1662 lines, ~50 private attributes) and `Subdomain` (1288 lines, 35 methods) are the two
  classes a student must understand and the two hardest; `Subdomain` is a definition + exchange +
  load-balancing-policy + migration object.

Soundness: the loud checks are unusually good (collective-inside-agreement refused, fingerprint,
constraint-copy digest, lumped-operator migration check, set completeness, poison). The silent class
of mistakes is "new per-process state": a new carried vector or node-field entry, a new whole-vector
diagnostic between outputs. Nothing at runtime catches those; only the CI byte-compare does, and only
if a deck exercises them.

Recommendation: merge after the cheap renames/rewrites in section 4 (half a day), and schedule the
two structural items (vector list in one place; load balancer out of `Subdomain`) as the first
follow-up, before more features land on NEDMPI.

---

## 1. Is this how one would design it from scratch? (Q1)

**Yes for the structure, no for the surface.** Structural choices that are right and well
explained:

| Choice | Where | Why it is right |
|---|---|---|
| Mesh as data, elements made from it; DOF layout from the mesh | `models/mesh.py`, `femodel.py:435-459`, `globaldofnumbering.py` | Lets a partition exist before elements, and makes the numbering identical by construction |
| `ElementDistribution` null object, every method trivial serially, documented as a table | `models/elementdistribution.py:29-69` | The serial reader can skip it with confidence; the table says exactly where the model meets the decomposition |
| `LocalOrderedSet`: complete-or-raise | `sets/orderedset.py:202-351` | Fails safe; a partial result cannot be silently produced |
| `ModelPartition` with `dofs = slice(None)` serially | `solvers/base/modelpartition.py` | One concept, no branch, zero cost |
| Contribution buffer in `ElementPlan` so interface sums are left folds in model order | `parallelelementcomputation.py:153-226`, `subdomaininterface.py:340-397` | The bit-identity mechanism is one idea, stated once, tested with values whose sum is order-sensitive |
| `Communicator.withoutCommunication` + `allRanksFailTogether` | `communicator.py:82-124, 444-512` | The rule "no collective inside an agreement" is a mechanism, not a convention; good teaching material |
| `ExactSum` | `numerics/exactsum.py` | Order-independent external work with a 40-line, well-documented class |
| Opt-in claims with safe defaults (`replicatedElementsReason`, `nodesReadByConnectivityUpdate`, `fieldOutputsRead`) | `constraintbase.py:64-67, 170-190`, `modelmodifierbase.py:76-79, 104-120` | A new constraint is correct by default (whole-model fallback) and must earn distribution with a test |

Signals of accretion (each is small; together they are what a student sees):

1. **Vector triple enumerated by hand in nine places** (confirmed): `nonlinearexplicitdynamicmpi.py:500, 948, 1109-1116, 1221, 1329, 1422-1423, 1661-1662`; `nonlinearexplicitdynamic.py:1588, 2123-2125`. See finding F2.
2. **`ExplicitSystem` reused as a half-empty carrier** (confirmed): `releaseEquationSystem` returns `ExplicitSystem(Minv=None, dU=None, ...)` (`nedmpi:1109-1116`), while the dataclass is documented as "everything an explicit increment operates on" (NED:299-336).
3. **`IncrementPlan.elementLoop: list[tuple] | ElementPlan`** (confirmed, NED:387): a union with no common interface; NEDMPI reads `.elements` on it (`nedmpi:547, 592`), which only `ElementPlan` has. Works by inheritance; reads as ill-typed.
4. **Hooks that exist only to be overridden**, with no serial purpose: `advanceModelToTime` (NED:884-896, one line), `halfMassTimesSquaredRate` (NED:1394-1411), `partitionModel` (NED:1674-1695). Acceptable, but they should be gathered in one labelled section of NED ("Seams a domain-decomposed solver overrides") instead of being spread through the file.
5. **The driver** opens with the MPI role (`inputfiledrivensimulation.py:107-118`) and returns to it at 138-148, 222-230, 269-274, 310-318, 350-353.
6. **`FEModel.elementStatesFromElsewhere`**, a context manager that sets a hidden attribute read by `writeRestart` (`femodel.py:1136-1158, 1131-1134`), used only by `NEDMPI.writeIncrementOutput:1180`. A pass-through argument was not possible through the output-manager API, so this is the workaround; it must be named as such.
7. **`FEModel.materialPropertiesToExport` is collective** in a distributed run (`femodel.py:937-971`), the only collective method of the model. A model method that may block on other processes is a surprise for a serial reader and a deadlock for a student who calls it from `if rank == 0:`.

---

## 2. Can a student read the serial path without meeting parallel concepts? (Q2)

Walk: input → mesh → nodes/elements → DofManager → NED increment.

| Stage | File | Meets MPI mechanisms? | Meets MPI vocabulary? | Grade |
|---|---|---|---|---|
| driver | `drivers/inputfiledrivensimulation.py` | yes: `isRootProcess`, `abortAllProcessesOnUncaughtException`, `elementDistributionOfThisJob`, `supportsDomainDecomposition` check, two `abortAllProcesses` | 6 places | moderate leak |
| input → mesh | `helpers/inputfilehelpers.py:378-381` | one line: `decideLocalElements` | 2 docstring mentions | fine |
| mesh | `models/mesh.py` | none | "held in this process" twice (module doc 41-45) | clean |
| nodes/elements | `models/femodel.py` | `elementDistribution`, `createChildElementOfMesh` returning `None`, `dropElementOfMesh`, `putElementsInMeshOrder`, `requireCompleteMesh`, `wholeElementSet`, `acceptStatesOf`, `elementStatesForCheckpoint`/`FromElsewhere`, `materialPropertiesToExport`, `createNodesOfMesh(isHeldHere)` | 25 hits | moderate leak |
| sets | `sets/*.py`, `sets.rst` | `isComplete`, `localElements()`, `elementNumbersOfWholeSet`, `requireComplete` | 12 of the 30 lines of `sets.rst` are about processes | vocabulary leak |
| DofManager | `numerics/dofmanager.py` | none | none | clean |
| elements | `elements/` | none | none | clean (owner's rule holds) |
| NED increment | `solvers/nonlinearexplicitdynamic.py` | `self.partition.dofs` in `updateVelocities:1124-1135`, `advanceDisplacements:1178-1181`, `assembleNetForce:1232-1233`, `planIncrement:1792-1794`; `partitionModel`; `advanceModelToTime`; fsum in `addExternalWork` | 46 hits; module docstring has a section on NEDMPI (138-144) | the main leak |

Where exactly MPI leaks into serial code, naming, base classes, sets, the model:

* **Naming/deixis** (confirmed, pervasive): "integrated here", "computed here", "evaluated here",
  "held here", "this process", "the owner". The serial reader must learn that "here" means "in this
  process" and that in a serial run "here" is everywhere. Standard words exist: *locally owned*,
  *ghost / locally relevant*, *subdomain*, *interface*, *replicated*. They are mentioned exactly once
  (`orderedset.py:207`, "in the terms of deal.II and PETSc").
* **Base classes**: `ConstraintBase.replicatedElementsReason`, `outputResults`,
  `nodesReadByConnectivityUpdate`, `getOutputResults/setOutputResults`;
  `ModelModifierBase.replicatedElementsReason`, `fieldOutputsRead`;
  `GeneratorBase.replicatedElementsReason`, `describesElementsOfMesh`;
  `NonlinearSolverBase.supportsDomainDecomposition`. Each is a class attribute or a one-line default
  with a safe value; the cost is one docstring paragraph per base class and one line in ~25 subclass
  files. This is the right place for them; the docstrings should lead with the serial meaning.
* **Sets**: `len()`, iteration, indexing are now guarded (`orderedset.py:242-280`). Serially invisible;
  the docstrings are not.
* **Model**: see table. The `ElementDistribution` table (`elementdistribution.py:43-53`) is the one
  place that lists all seven contact points, and it is good; `FEModel`'s own docstrings repeat the
  MPI story in every affected method.

Bottom line for Q2: a student never has to *execute* a parallel concept to follow the serial path,
and never meets one in `mesh.py`, `dofmanager.py` or the elements. They meet the *words* at every
step, and in NED they meet an increment written for a part of the model.

---

## 3. Can a student understand NEDMPI from code + docs? (Q3)

**Yes, with effort proportional to its size; the invariants are stated where they are enforced.**

Where each invariant lives and is enforced:

| Invariant | Stated | Enforced |
|---|---|---|
| Bit-identity with NED | `subdomaininterface.py:29-50`; `nedmpi:99-108`; `domaindecomposition.rst:258-290` | `ElementPlan.assembleInto` (bincount left fold, `parallelelementcomputation.py:206-225`), `InterfaceForceAssembly.assemble:375-397`, `ConstraintForceExchange` model-order additions, `ExactSum`; tests `test_assembly_sums_every_degree_of_freedom_from_left_to_right_in_element_order`, `test_interface_forces_are_summed_in_element_order_on_every_process`, CI `--tolerance 0` + checkpoints |
| Ownership (one owner per DOF, lowest rank; rank 0 owns untouched DOFs) | `subdomaininterface.py:45-49, 94-103` | `SubdomainInterface.__init__:108-120`; `ownedDofMask` docstring `subdomain.py:709-723` |
| Same layout everywhere | `subdomain.py:453-489` | `_checkReplicatedLayout` + `layoutFingerprint:1220-1285` + `GlobalDofNumbering.requireAgreementWith` |
| Synchronization points / what may be read | `nedmpi:50-78`; `rst:316-394` (table) | `requireWholeSolutionHere:1580-1606`, `Subdomain.requireConstraintCopiesCurrent:680-707`, `_requireCoupledNodesAmongNodesRead:1451-1483`, field-output `_raiseHeldElsewhere` (`fieldoutput.py:242-258`), `debug-poison-stale-solution` |
| No collective inside an agreement | `communicator.py:29-44` | `_requireCommunicationAllowed:103-124` |
| Migration keeps the lumped operators | `nedmpi:1057-1088` | `_requireSameLumpedOperators` (bytes compare) |

Vocabulary (standard or not):

* Standard and well used: subdomain, interface, owner, collective, point to point, partition (METIS
  sense), dual graph, migration, replicated.
* Non-standard and should be mapped once: "integrates" (= locally relevant/ghosted DOFs), "touched"
  (= adjacent), "complete" (= set entirely local), "whole model on every process" (= replicated /
  serial-redundant), "agreement" / "fail on all ranks together" (= collective error propagation),
  "synchronization" (= ghost update / gather), "publish" (= write vectors into node fields).
* One word carries two meanings (confirmed): *partition* is both "the assignment of elements to parts"
  (`Subdomain._partition`, `partitionElementsOfMesh`, "repartition") and "the part this process
  computes" (`ModelPartition`, `NED.partition`). `modelpartition.py:31-34` even says "split it into
  *parts*". Rename `ModelPartition` → `ModelPart`, `NED.partition` → `NED.part` (finding F5).

Duplication between NED / NEDParallel / NEDMPI: code duplication is low (NEDParallel is 194 lines,
every NEDMPI override calls `super()` or the shared function). Three places where the same thing
exists twice:

* Kinetic energy `0.5 Σ m v²` is written out at NED:638-640 and 2507-2509 and as the hook
  `halfMassTimesSquaredRate:1394-1411`; only the hook is owner-restricted by NEDMPI (`nedmpi:841-859`).
  The two literal copies are right only because they run after a synchronization that reached every
  process. A student who deduplicates them through the hook silently turns the topology-change
  diagnostic into a per-process number in NEDMPI (F4).
* Publishing into node fields: `NED.publishNodeFields:1565-1595` (field loop) and
  `NEDMPI._publishIntoNodeFields:1226-1247` (slot loop) are two implementations of one operation.
* The constraint evaluation: `NED.assembleConstraintForces` evaluates and adds (2152-2199);
  `NEDMPI.assembleInternalForces` evaluates (670-674) and `NEDMPI.assembleConstraintForces` only
  adds (745-786), with a `_constraintForcesOfIncrement` handshake and a `RuntimeError` if the
  handshake is skipped. This is the single most surprising control flow in NEDMPI for a reader who
  has just learned NED's `assembleNetForce` order; it is documented (`nedmpi:636-643`) and the reason
  (evaluate the costly contact before the interface exchange, while others compute the elements given
  to them instead) is good. Make the split visible in NED itself (F6).

Clever mechanisms a simpler design would avoid, or at least must name:

* Lazy receive plans with a "forget" protocol: `_searchesPlanned`, `_solutionReadBySearches`,
  `_rigidBodyReferences`, `_forgetReceivePlans` (`nedmpi:354-364, 1393-1400`). Each is planned on
  first use and forgotten on `planIncrement`/`releaseEquationSystem`. Correct, but three booleans and
  Nones encode one state machine ("plans valid for the current subdomain"). A single
  `ReceivePlans` object created in `planIncrement` and discarded with it would have no lazy state.
* `_ENTRIES_PER_CONSTRAINT = 2**32` as an order key `position * 2**32 + index`
  (`subdomaininterface.py:59-61, 520-536`). Fine, guarded, but a `(position, index)` lexsort is the
  same thing without the magic constant.
* `FEModel.elementStatesFromElsewhere` context manager (above).
* `ExplicitSystem` with `None` members as a carrier (above).
* NEDMPI's `_wholeSolutionHere` flag is set in three places and cleared in one (`nedmpi:353, 907, 1566`)
  -- the whole "what may be read" story hangs on this one boolean; it is correct and tested
  (`test_a_process_reads_no_solution_it_did_not_receive`), and it deserves to be the first thing the
  class docstring explains, with its three transitions.

Docs: `domaindecomposition.rst` (952 lines) is complete and honest, and the two tables
(synchronization points 326-351; what a constraint reads 481-498) are exactly what a student needs.
But the conceptual model, the invariants, the mechanisms and the performance anecdotes (c1_150
timings, MB tables, rabbit measurements, "8x4 base / 32x1 now") are interleaved, so the page reads as a
lab notebook rather than a chapter. `parallelization.rst` still says `python edelweiss.py` and is
OpenMP-only with one sentence pointing to MPI. `solvers.rst` gives NEDMPI four lines. The glossary is
missing entirely.

---

## 4. Ranked findings

Severity scale: **H** = will mislead or will be broken by a student; **M** = costs understanding;
**L** = polish. Status: *confirmed* (read and, where stated, run) or *plausible* (read, not run).

### F1 (H, confirmed) — NED's kinematic update is written for a part of the model
`nonlinearexplicitdynamic.py:1122-1135, 1178-1181, 1228-1233`. Every line of the central-difference
update gathers with `[dofs]` and scatters back, where `dofs` is `slice(None)`. The owner's goal
"serial NED keeps a plain textbook element loop" holds for the element loop (1449-1460) and not for
the integrator. The integrator formula and the partition indexing are fused, so NEDMPI cannot
override the indexing without duplicating the formula, which is why NED carries it.
**Fix** (keeps one copy of the formula, removes `dofs` from NED): extract the update rule as a pure
function on plain arrays and let NED call it on whole vectors, NEDMPI on `[dofs]` views
(rewrite R2 below). Same for `advanceDisplacements`. The MPC fold in `assembleNetForce` can stay
`[dofs]`-free in NED if NEDMPI overrides the fold.

### F2 (H, confirmed) — The carried vectors (U, V, P) are enumerated by hand in nine places
`nedmpi:500, 948, 1109-1116, 1221, 1329, 1422-1423, 1661-1662`; `ned:1588, 2123-2125`. The driver
already creates an `"A"` node-field entry (`inputfiledrivensimulation.py:183-188`) that no explicit
solver publishes. The day a student publishes an acceleration from NED (a natural extension), NEDMPI
will publish it only at integrated DOFs, leave it stale elsewhere, and nothing raises: not the poison
(it poisons U, V, P only, `nedmpi:946-950`), not a check, only the CI byte-compare of a checkpoint
that happens to include `A`. **This is the most likely silent break.**
**Fix**: one tuple of `(entry, vector)` owned by `ExplicitSystem` (e.g. `publishedEntries()`), consumed
by both `publishNodeFields` implementations, `_synchronizeModel`, `_synchronizeAcceptedModel`,
`_poisonStaleSolution`, `_receiveRigidBodyReferences`, `_receiveNewlyIntegratedValues`,
`releaseEquationSystem` and the rebuild path; and a test that NED's published entries equal NEDMPI's.

### F3 (M, confirmed) — Deictic vocabulary instead of the standard one, everywhere
~60 serial files; NED alone 46 occurrences of "here/process/partition" (0 at the merge base
a59ae166). `orderedset.py:207` is the only sentence mapping to deal.II/PETSc terms. A student who
has read any parallel FE text expects *locally owned*, *ghost*, *subdomain*, *replicated*; one who
has not expects nothing in a serial docstring. **Fix**: a glossary at the top of
`domaindecomposition.rst` (R5), and in serial classes move the MPI sentence to the end of the
docstring, phrased "In a domain-decomposed run, ..." (R1, R6).

### F4 (M, confirmed) — Kinetic energy computed three ways, only one through the overridable hook
`ned:638-640, 2507-2509` (literal) vs `halfMassTimesSquaredRate:1394-1411` (hook, owner-restricted in
`nedmpi:841-859`). Correct today because the literal copies run after a full synchronization; wrong
the day someone routes them through the hook, with no failure. **Fix**: one `kineticEnergy(V)` on NED
(sum over `ids_mechanicalEnergy`) that NEDMPI overrides as "owned part, then `allreduceSum`" so the
method means "the model's kinetic energy" in both; use it in `energyBalanceTerms`,
`prepareIncrement` and `reportTopologyChangeConservation`. `secondOrderMomentum:2243-2263` has the
same shape and should get the same treatment.

### F5 (M, confirmed) — "partition" means two things
`ModelPartition`/`NED.partition` ("the part this process computes") vs `Subdomain._partition`,
`partitionElementsOfMesh`, "repartition" ("the assignment of elements to parts"), both in the same
docstrings (`modelpartition.py:31-34`). **Fix**: rename `ModelPartition` → `ModelPart`,
`NED.partition` → `NED.part`, `partitionModel` → `partOfModelComputedHere` (or `defineModelPart`).
Cheap now, expensive after merge.

### F6 (M, confirmed) — The constraint evaluate/add split is invisible in NED and surprising in NEDMPI
`ned:2152-2199` (evaluate+add in one method) vs `nedmpi:670-674` (evaluate inside
`assembleInternalForces`) and `nedmpi:745-786` (add only, with a `_constraintForcesOfIncrement`
handshake and `RuntimeError`). **Fix**: in NED, split `assembleConstraintForces` into
`evaluateConstraintForces(...) -> dict[str, ConstraintForce]` and `addConstraintForces(forces, P)`,
called one after the other in `assembleNetForce`. NEDMPI then overrides `assembleInternalForces` to
call `evaluateConstraintForces` early and keep the result, and `evaluateConstraintForces` to return it.
The reordering becomes a visible override of a visible step, and the handshake disappears.

### F7 (M, confirmed) — `Subdomain` is four objects
`subdomain.py`: definition/partition (201-451), layout check (453-528), load restriction
(532-650), search/ownership queries (652-723), vector/state exchange (727-915), load-balancing policy
with six cost attributes (917-1170), migration (1172-1217). The policy part (`rebalance`,
`_imbalance`, `recordRepartitionCost`, `rebalanceAfterTopologyChange`, `_lastRepartitionCost`,
`_lastMigrationCost`, `_meanElementCost`, `_constraintCosts`, `_processShares`) is a separable
`LoadBalancer` that takes a `MeasuredLoad` and answers "repartition? with which owners?". NEDMPI is
likewise ~50 private attributes. **Fix**: extract `LoadBalancer` (pure decisions, unit-testable
without MPI) and `ReceivePlans` (F-"clever" above) out of `Subdomain`/`NEDMPI`.

### F8 (M, confirmed) — The driver and `FEModel` carry the MPI role inline
`inputfiledrivensimulation.py:107-118, 138-148, 222-230, 269-274, 310-318, 350-353`;
`femodel.py` (25 mentions; `materialPropertiesToExport` collective; `elementStatesFromElsewhere`
context). **Fix**: a small `JobProcesses` object in `domaindecomposition/` with `writesOutput`,
`elementDistribution(inputfile)`, `refuseUnlessDecomposed(step)`, `stopAllOnFailure(failure)`,
so the driver reads as serial with one object standing in for "the processes". In `FEModel`, lead every
affected docstring with the serial sentence and end with "In a domain-decomposed run, ...".
Mark `materialPropertiesToExport` **Collective** as the first word of its summary.

### F9 (L, confirmed) — `IncrementPlan.elementLoop` union and `ExplicitSystem` as carrier
`ned:387`; `nedmpi:1109-1116`. **Fix**: a `CarriedState(U, V, P, criticalTimeStep)` dataclass used by
NED's rebuild path (`previous` → `carried`) and by `releaseEquationSystem`; give NED's plain loop a
tiny class with `.elements` and `__iter__` so both loops share an interface, or type the field as a
`Protocol`.

### F10 (L, confirmed) — Docs: structure and staleness
`domaindecomposition.rst` mixes concept, invariant, mechanism and measurement; no glossary;
`parallelization.rst:1-15` says `python edelweiss.py`; `solvers.rst:192-200` gives NEDMPI four lines;
`sets.rst` is 40 % about processes. **Fix**: R5; move every timing/MB table to a "Performance notes"
section at the end; one paragraph in `solvers.rst` saying what NEDMPI adds to NED and pointing at the
override table.

### F11 (L, plausible) — Silent per-process diagnostics
Any new reduction over a whole vector between output increments in NED (`np.linalg.norm(P)`, a new
momentum line) is per-process in NEDMPI and nothing says so; with `debug-poison-stale-solution` it
becomes NaN, otherwise a plausible-looking number. The guards cover reading the whole *solution*
(`requireWholeSolutionHere`), not reductions. **Fix**: name the rule once in NED's docstring section
"Seams a domain-decomposed solver overrides": "a sum over all degrees of freedom belongs in a method
NEDMPI overrides (`kineticEnergy`, `energyBalanceTerms`); a sum written inline is a sum over one
process". Consider running `testfiles/mpi` with the poison by default (CI already does, as a third
pass), so a student sees NaN rather than a number.

### F12 (L, confirmed) — Setup-time failure in one process hangs the others
`domaindecomposition.rst:922-926` states it; `stepFailed` covers steps only. Under the test runner a
deck that fails on one rank during model building leaves the others in `Bcast` of the partition
(`partitioning.py:148`). Rare (all ranks read the same input) but a student will hit it with a per-rank
file path. **Fix**: wrap `fillFEModelFromInputFile` + `prepareYourself` in the driver in one
`allRanksFailTogether("Setting up the model")` after the partition broadcast, or document the symptom
("the job hangs after `Setting up finite element model`") next to the limitation.

### F13 (L, confirmed) — Good, keep: the loud checks
For the record, what a student *cannot* break silently: a collective inside an agreement
(`communicator.py:117-124`), reading a partial set as a whole (`orderedset.py:242-280`), processes
building different models (`subdomain.py:478-489`), a constraint copy diverging before a topology
update (`subdomain.py:700-707`), a constraint coupling a node it did not name (`nedmpi:1451-1483`),
a state-dependent lumped mass across a migration (`nedmpi:1057-1088`), elements described after the
partition (`distributedelements.py:451-460`), a field output read where it was not gathered
(`fieldoutput.py:242-258`). This is a strong net; it is why F2/F4/F11 (the silent class) are the ones
to close.

---

## 5. Concrete rewrites for the worst passages

### R1 — NED module docstring, section "The part of the model computed here" (`ned:138-144`)
Move it to the *end* of the module docstring, after "Restart", under a heading that names it as a
seam, and drop the deixis:

```
**Seams a domain-decomposed solver overrides.** The increment runs over one object,
:class:`~edelweissfe.solvers.base.modelpartition.ModelPart` (:attr:`NED.part`): the elements,
the constraints and the degrees of freedom of the model computed in this process. For ``NED``
that is the whole model, and ``part.dofs`` is ``slice(None)``. A domain-decomposed solver
(:class:`~edelweissfe.solvers.nonlinearexplicitdynamicmpi.NEDMPI`) gives each process a
subdomain instead and overrides exactly these methods: :meth:`defineModelPart`,
:meth:`assembleInternalForces`, :meth:`assembleLumpedDiagonal`, :meth:`assembleLoads`,
:meth:`evaluateConstraintForces`, :meth:`kineticEnergy`, :meth:`energyBalanceTerms`,
:meth:`addExternalWork`, :meth:`publishNodeFields`, :meth:`acceptIncrement`,
:meth:`advanceModelToTime`, :meth:`updateConstraintConnectivity`, :meth:`updateTopology`,
:meth:`writeIncrementOutput`, :meth:`applyStepActionsAtStepEnd`, :meth:`buildEquationSystem`.
Rule for this file: a sum over all degrees of freedom belongs in one of them; written inline,
it is a sum over one process.
```

### R2 — `NED.updateVelocities` / `advanceDisplacements` (`ned:1089-1135, 1162-1181`)
Pure functions carry the formula once; NED calls them on whole vectors; NEDMPI on views.

```python
def centralDifferenceVelocityUpdate(velocity, force, Minv, halfRateStep, secondOrderMask, dTAverage):
    """v_{n+1/2} = ((1 - h) v_{n-1/2} + dT M^{-1} f_n) / (1 + h), h = alpha dT / 2, in place.
    At h = 0 bit-identical to the undamped update, which is why there is no branch."""
    velocity *= 1.0 - halfRateStep
    velocity += Minv * force * (secondOrderMask * dTAverage)
    velocity /= 1.0 + halfRateStep


class NED:
    def updateVelocities(self, V, P, Minv, dTAverage):
        """Advance the velocity of every degree of freedom by the net nodal force (see
        :func:`centralDifferenceVelocityUpdate`); forward Euler on the first-order ones."""
        centralDifferenceVelocityUpdate(
            V.asPlainArray(), P.asPlainArray(), Minv.asPlainArray(),
            self._halfDampingRate * dTAverage, self._secondOrderMask, dTAverage,
        )
        firstOrder = self.ids_1st
        V.asPlainArray()[firstOrder] = Minv.asPlainArray()[firstOrder] * P.asPlainArray()[firstOrder]


class NEDMPI:
    def updateVelocities(self, V, P, Minv, dTAverage):
        """The update of :meth:`NED.updateVelocities` at the degrees of freedom of this subdomain."""
        dofs = self.part.dofs
        velocity = V.asPlainArray()[dofs]          # a copy: dofs is an index array
        centralDifferenceVelocityUpdate(velocity, P.asPlainArray()[dofs], Minv.asPlainArray()[dofs],
                                        self._halfDampingRate[dofs] * dTAverage,
                                        self._secondOrderMask[dofs], dTAverage)
        V.asPlainArray()[dofs] = velocity
        firstOrder = self._incrementPlan.firstOrderDofs
        V.asPlainArray()[firstOrder] = Minv.asPlainArray()[firstOrder] * P.asPlainArray()[firstOrder]
```

`firstOrderDofs` then leaves `IncrementPlan` (NED uses `ids_1st`; NEDMPI intersects once in its
own `planIncrement`). Bit-identity is preserved because the arithmetic per entry is unchanged; it
must be re-verified with the CI byte-compare after the refactor.

### R3 — `NEDMPI.acceptIncrement` docstring (`nedmpi:886-903`)
The current summary is one 70-word sentence. Replace with:

```
Commit the increment (:meth:`NED.acceptIncrement`); then, on an output increment, three things.
Collective.

1. Make the whole model current where it will be read: on rank 0, which writes the output and
   the checkpoint, or in every process if a topology check follows
   (:meth:`outputSynchronizationReachesEveryProcess`).
2. Rebalance the subdomains from the measured costs (:meth:`Subdomain.rebalance`).
3. If that moved elements between processes, build the equation system again for the elements
   now held here (:meth:`_moveElements`); if it changed only which degrees of freedom this
   process integrates, receive the vectors at the new ones from their previous owners.

After any increment the vectors are current at the degrees of freedom integrated here only
(``_wholeSolutionHere = False``); see :meth:`requireWholeSolutionHere`.
```

### R4 — `Subdomain.constraintsSearchedHere` docstring (`subdomain.py:652-678`)
The second paragraph explains, in one breath, why a stale copy's footprint is harmless. Split:

```
Those of the given constraints whose connectivity search runs in this process: the ones it
owns -- or, right after a topology change, is about to own.

A search runs on the owner only; the other processes' copies keep the footprint of their last
synchronization. That stale footprint is read by exactly one thing, the activation of fields
on nodes when the next equation system is built, and it is harmless there because every copy
equals its owner's whenever the mesh changes -- asserted by :meth:`requireConstraintCopiesCurrent`.
```

### R5 — `domaindecomposition.rst`: open with a glossary and the invariants (new, before "Running a job")

```
Vocabulary
----------

================================  ======================================================================
here                              in this process, which holds a *subdomain* of the model
subdomain degrees of freedom      the DOFs the elements and constraints of this process touch:
("integrated here")               *locally relevant* DOFs in deal.II's terms (owned + ghost)
owned DOF                         exactly one process per DOF, the lowest rank integrating it:
                                  *locally owned*; it counts what must be counted once
interface DOF                     integrated by more than one process (a ghost in every other)
local element / node              has an object in this process
complete set                      every member of the set is local; reading a partial set raises
replicated                        held identically in every process: the mesh, the nodes, the sets,
                                  the facets, the rigid bodies, the DOF numbering
whole model on every process      every element object replicated (the fallback rule)
synchronization                   every entry of a vector, and every state, from its owner
agreement                         every process raises the failure any one of them met
================================  ======================================================================

Invariants
----------

1. **Bit-identity.** Every sum that decides the solution is a left fold in model order; the
   external work is an exact sum. A run on any number of processes equals ``NED``'s run.
2. **One owner per degree of freedom**, and the owner's value is authoritative in every gather.
3. **Same model everywhere.** Every process builds, refines and numbers the model identically;
   checked by a fingerprint after every build.
4. **Read only what you received.** Between synchronizations a process reads vectors at its
   subdomain DOFs only; the whole-model readers are listed in the table below and refuse otherwise.
5. **No communication inside an agreement.** Enforced by the communicator.
```

Then the existing sections, with every measurement table moved under a final "Performance notes".

### R6 — `FEModel.createNodesOfMesh` parameter (`femodel.py:197-202`)
`isHeldHere` is passed by no production caller (grep: only `tests/test_mesh.py:503` passes it), and the
docstring says "(for now)". Either drop the parameter until P6 round 2 uses it, or phrase it as the
seam it is: "``isHeldHere``: which nodes of the mesh get an object in this process; ``None`` for
all of them (every run today). The hook a process holding only the nodes of its subdomain will use."

---

## 6. Prioritized recommendations

Before merging (each < 2 h, no behaviour change, verified by the existing CI byte-compare):

1. **F5** rename `ModelPartition`/`partition` → `ModelPart`/`part` (one word, two meanings).
2. **R1 + F11** one "seams" section at the end of NED's docstring, with the inline-sum rule; drop
   the NEDMPI paragraph from the top of NED.
3. **R5 + F10** glossary and invariants block at the top of `domaindecomposition.rst`; move
   measurements to a "Performance notes" appendix; fix `parallelization.rst` (`edelweiss.py`).
4. **F3/R6/F8-docstrings** in `femodel.py`, `sets/*.py`, `sets.rst`, `constraintbase.py`,
   `modelmodifierbase.py`: serial sentence first, "In a domain-decomposed run, ..." last.
5. Make "Collective." the first word of every collective method's summary line (currently trailing
   in ~40 docstrings), and mark `FEModel.materialPropertiesToExport` so.

First follow-up PR (behaviour-preserving refactors, each with the byte-compare as the gate):

6. **F2** one tuple of published entries / carried vectors; a test that NED and NEDMPI publish the
   same entries. Closes the most likely silent break.
7. **F4** `kineticEnergy(V)` and a momentum counterpart as the only reductions, overridden once.
8. **F6** split `assembleConstraintForces` into evaluate + add in NED; NEDMPI moves the call.
9. **F1/R2** pure-function integrator; NED without `[dofs]`.
10. **F9** `CarriedState` instead of a half-empty `ExplicitSystem`; a common interface for the two
    element loops.

Second follow-up (structure):

11. **F7** `LoadBalancer` out of `Subdomain`; `ReceivePlans` out of `NEDMPI` (removes the three
    lazy flags and `_forgetReceivePlans`).
12. **F8** `JobProcesses` in `domaindecomposition/`, so the driver reads serially.
13. **F12** one agreement around model setup, or the documented hang symptom.

What not to change: the `ElementDistribution` null object and its table; `LocalOrderedSet`'s
complete-or-raise; the `Communicator` guard; the contribution-buffer assembly; `ExactSum`; the
opt-in claims with safe defaults; the poison mode and the three-pass CI. These are the parts a
student should be shown as *how to do it*.

---

## Appendix: numbers behind the claims

* `git diff a59ae166...HEAD -- edelweissfe doc`: 164 files, +19 263 / -5 762 (branch includes
  upstream merges; MPI-specific additions are `domaindecomposition/` 4 207 lines,
  `nonlinearexplicitdynamicmpi.py` 1 662, `modelpartition.py` 93, `elementdistribution.py` 230,
  `globaldofnumbering.py` 231, `exactsum.py` 161, `domaindecomposition.rst` 952).
* Occurrences of `NEDMPI|process|partition|computed here|integrated here|evaluated here` in NED:
  46 at HEAD, 1 at the merge base.
* Files outside `domaindecomposition/` and NEDMPI that call a decomposition hook
  (`localElements()`, `elementNumbersOfWholeSet`, `requireComplete`, `elementDistribution`,
  `replicatedElementsReason`, `nodesReadByConnectivityUpdate`, `fieldOutputsRead`, ...): 47;
  `femodel.py` 25 hits, `elementset.py` 16, `hadaptivity.py` 9, `marking.py` 7.
* Zero hits in `numerics/dofmanager.py`, `elements/base/baseelement.py`,
  `elements/displacementelement/element.py`, `elements/marmotelement/element.pyx`.
* Bitwise runs at 728930ca: `NED`, `NEDEqualValuePenalty`, `NEDSurfaceContact`; serial `NED` vs
  `NEDMPI` on 2 processes and on 3 processes with `debug-poison-stale-solution=True`: 6/6 `cmp` equal.
