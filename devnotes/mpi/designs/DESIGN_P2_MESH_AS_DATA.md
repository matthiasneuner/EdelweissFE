# P2 design note: the mesh as data (DECIDED 2026-10-06, see §9; drafted 2026-10-02; updated for upstream 804d3817, 2026-10-05; revised after the review, 2026-10-06)

Part of DESIGN_DISTRIBUTED_MEMORY.md (§3 layers 1–3, §5 P2). **Plan only — no code yet.** P2 changes
how every model is built, but must leave every serial result bit-identical; it is the enabler for P3
(each process creates only its own elements).

## 1. The problem in one paragraph

Today a model is built straight from the input text into objects: `abqmodelconstructor.py:102-142`
creates a `Node` per node line and an element object per element line (`ElementClass(type, label)`,
`setNodes`), and six generators do the same (boxgen, planerectquad, pipegen, microstructure,
cuboidlattice; cubit goes through the Abaqus constructor). Everything downstream refers to those
objects: an `ElementSet` is an ordered set of element objects (`*elset` by number raises `KeyError` on a
missing element), a surface maps a face number to an `ElementSet`, a section holds `ElementSet`s, and the
DOF layout follows from which fields each *element object* switches on at its nodes
(`FEModel._populateNodeFieldVariablesFromElements`, `femodel.py:719`). So a process that does not create
an element loses the element's set memberships and, worse, the fields at its nodes: the DOF layout would
differ between processes. P2 removes the dependence on element objects from all of this.

## 2. The idea

**The mesh is data; elements are made from it.** Model construction gets two clearly separated stages:

1. **Describe the mesh.** The Abaqus constructor and the generators fill a `Mesh` held by the model
   (`model.mesh`): the nodes (as today, see §5), and per element its number, type, provider and node
   labels; element sets and node sets as ordered label lists; surfaces as (face, element-set name) lists;
   element numbers reserved from the mesh, not from created objects. (Upstream #194/#195 moved the
   number allocators, change log and fingerprint into `model.topology`, `models/topologypipeline.py`;
   the mesh draws its numbers from those allocators instead of owning new ones.)
2. **Make the elements.** `model.createElementsOfMesh(isCreatedHere)` creates an element object for every
   element of the mesh for which `isCreatedHere(number)` holds, in mesh order, and resolves the sets and
   surfaces to those objects. Serially the predicate is "every element", and the model is exactly
   today's model.

This is the textbook picture (mesh → elements), so it makes the construction easier to teach, not harder.

## 3. What each consumer sees

| Thing | Meaning after P2 | Serial | Under P3 |
|---|---|---|---|
| `model.mesh` | the whole mesh, as data | – | identical on all processes |
| `model.elements` | the element objects **created here**, by number, in mesh order | all | own elements |
| `ElementSet` | the elements of the set **created here** (same class, same API) | unchanged | own part of the set |
| `model.mesh.elementSets[name]` | all element numbers of the set | – | for gathers to rank 0 |
| `NodeSet`, `model.nodes` | unchanged (nodes stay replicated, §5) | unchanged | unchanged |
| `EntityBasedSurface` | faces of the created elements of its sets | unchanged | own part |
| sections, element properties | assigned to the created elements of their sets | unchanged | own elements |

The one semantic shift is "an element set is the part of the set that is created here". Serially that is
the whole set, so no serial code path changes behaviour; every consumer that iterates a set or
`model.elements` keeps working.

**A partial set must never be read as if it were whole** (revision after the 2026-10-06 review,
REVIEW_MPI_2026-10-06.md). Silent partial results on rank 0 — an output, a marker or a checkpoint
computed from "the part created here" while believing it is the whole set — are the typical failure of
this design, so it is excluded by a loud invariant rather than by care:

- An `ElementSet` knows whether it is complete (`isComplete`: serially always true; under P3 true only
  if every element of `model.mesh.elementSets[name]` was created here).
- Every consumer that needs the whole set — field outputs over sets, Ensight geometry and results,
  checkpoints, markers, mesh/plot outputs — states it by calling `elementSet.requireComplete(reader)`
  (raises with the reader's name on a partial set), or reads through the explicit gather path instead.
- The gather path is one API on the partition (P3): whole-model element results by mesh element
  number, gathered to rank 0 in mesh order (`gatherElementResults(setName, result)`); node results are
  already global-length. Consumers are converted one by one in P3; one that is not converted fails
  loudly at its first use under MPI, never silently.

## 4. Fields and DOF layout without element objects

Which fields an element has at which of its nodes depends on its **type** only (Marmot:
`getNodeFields` of the element type; Python elements: their class). The mesh asks one prototype element
per (type, provider) once, and `FEModel` activates the field variables of every node from the mesh —
every element of the model, created here or not. Constraints keep activating fields as today (they are
built after the elements; surface-sized, replicated). The DOF layout is then a function of the mesh and
the constraints alone: identical on every process **by construction**, which retires the layout
fingerprint as a correctness device (it can stay as a cheap assertion).

**Checked exception:** the Marmot single-quadrature-point element (`marmotsingleqpelement/`) gets its
fields from its *material driver*, i.e. only once a section assigns the material. For it the mesh asks the
section's material instead of the type (same answer on every process, since sections are replicated
definitions); if that turns out awkward, these elements are simply created on every process (they are
used in small material-point tests, not in production meshes).

The `DofManager` keeps its global numbering; its per-element index arrays are built for the created
elements only (that part is distributed, ~1.1 KB/element in the census).

## 5. Deliberately unchanged in P2

- **Nodes stay `Node` objects with `FieldVariable`s** (~300 MB on c1_150, replicated). They are core
  teaching vocabulary and are referenced everywhere (elements, sets, constraints, AMR, outputs). Turning
  them into arrays is P6 (distributed nodes), not P2.
- **Contact facets and point masses** are created on every process (surface-sized, no kernels); the
  facet generator reserves numbers from the mesh, so they are the same everywhere.
- **AMR** keeps creating and removing element objects as today, now through the mesh: a refinement
  adds the children to the mesh (numbers from the mesh counter) and creates them where their parent was
  created. Compacting the AMR mirror into arrays (334 MB on c1_150) belongs to P4.
- Input syntax, generator options, results: unchanged.

## 5b. Element migration falls out of P2 (designed up front, before P3)

Rebalancing (and AMR children landing on a different process, P4) must move an element between
processes. With the mesh as data, an element is completely described by data every process already has
plus its state:

- **from the mesh:** number, type, provider, node labels — so the receiving process can create it;
- **from replicated definitions:** its section, material (incl. `materialParameterFromField`, evaluated
  at the element centre, deterministic) and element properties — resolved through its set memberships;
- **sent:** only the state vector (`getStateVars`), plus the number.

So migration = `createElementOfMesh(number)` + `assignSectionToElement` (exists since upstream #182) +
`setStateVars(state)` on the receiver, and dropping the object on the sender. P2 must therefore provide
element creation **per element, at any time** (not only the bulk `createElementsOfMesh`), and section
assignment per element — both are the natural shape of the stage-2 code anyway. Elements that expose no
state are refused for domain decomposition (review risk 1). Constraint state is not migrated: constraints
are replicated definitions whose evaluated state travels through `getRestartData` from their owner.

## 6. Work items

1. `Mesh` (new, `edelweissfe/models/mesh.py` or similar): element records (number, type, provider,
   node labels) in insertion order, element/node set label lists, surface definitions; numbers drawn from
   `model.topology`'s allocators (upstream #194), so that AMR, facets and the mesh share one counter.
2. Abaqus constructor and the six generators describe instead of create; set operations (`generate`,
   boolean union/difference/intersection, name unions) work on label lists — they become simpler.
3. `FEModel.createElementsOfMesh(isCreatedHere)` built on a per-element `createElementOfMesh(number)`
   (§5b); `ElementSet.isComplete` / `requireComplete` (§3); the driver calls it right after the mesh is described
   (serial: all), before sections/properties/constraints/modifiers.
4. Field activation from the mesh (§4); `_populateNodeFieldVariablesFromElements` goes.
5. AMR and the facet generator add to the mesh as well as creating objects.
6. Tests: a mesh-level unit test (sets, booleans, surfaces from labels; a predicate that skips elements
   leaves sets/surfaces/fields/DOF layout consistent); the full serial suites, pytest, and the bitwise
   harness must be unchanged.

## 7. Gate

Bitwise harness (`verify_refactor.sh … full`), the full edelweiss-only and marmot suites and pytest with
results identical to the current branch — P2 is a pure restructuring for serial runs. Plus one new check
that matters for P3: building the c1_150 model with a predicate that keeps a METIS part gives the same
DOF layout as building it whole.

## 8. Questions for you

1. **Name**: `model.mesh` / `Mesh` and `createElementsOfMesh` — or other vocabulary you prefer?
2. **Set semantics**: OK that `ElementSet` means "the part of the set created here" (§3), guarded by
   `isComplete`/`requireComplete` and an explicit gather path, rather than a separate local-set class?
   It keeps all serial code untouched, and the guard makes every unconverted whole-model reader fail
   loudly under MPI. (Recommendation revised after the review: yes, but only with the guard.)
3. **Generators**: change all six (and the Abaqus constructor) to describe the mesh in P2, or keep a
   compatibility path for generators that are rarely used (microstructure, cuboid lattice, pipe) and
   refuse them under MPI?
4. **Memory now or later**: P2 as written keeps nodes and the AMR mirror as objects (~0.6 GB of the
   1.6 GB replicated floor). Compacting them is P4/P6. Agree with deferring?

## 9. Decisions (2026-10-06)

1. Names: `model.mesh` / `Mesh`, `createElementsOfMesh(isCreatedHere)`, per element `createElementOfMesh(number)`.
2. `ElementSet` = the part of the set created here, guarded by `isComplete` / `requireComplete(reader)`
   plus one explicit gather path to rank 0.
3. All generators (Abaqus constructor + six) are converted to describe the mesh in P2.
4. Node objects and the AMR mirror stay objects in P2; compacting them is P4 (AMR) / P6 (nodes).

## 10. P3 follow-ups (from the P2 implementation and its review, 2026-10-06)

Left unchanged in P2 on purpose; each must be done (or decided) in P3:

1. **Partition from the mesh.** `domaindecomposition/partitioning.partitionElements` and the subdomain's
   `_checkReplicatedLayout` read `model.elements`; under P3 they must key on `model.mesh.elements`
   (connectivity from the records, weights from the type), since no process holds every element object.
   `mpi_explicit_benchmarks/p3_readiness/check_dof_layout.py` shows the mesh-based partition.
2. **Restart read of a not-created element.** `FEModel.readRestart` raises "the replayed model does not
   match the one checkpointed" for a checkpointed element that is merely not created in this process —
   misleading under P3; it must skip (or route) elements declined by the creation predicate and keep the
   error for elements missing from the mesh.
3. **`executePythonCode` / `cubit` create every element** described so far (they read element objects
   while the mesh is still being described). Under MPI that is correct but defeats the memory saving:
   refuse these generators under MPI, or accept the whole model for such decks.
