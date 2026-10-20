# Design note P6: local nodes and local DOF numbering (draft, 2026-10-08)

Base: `EdelweissFE-mpi2` at `ca6c1d91`. Plan only, no code. Context: DESIGN_DISTRIBUTED_MEMORY.md
§3-§6 (D1 replicated mesh, D2 I/O through rank 0, D3 bitwise), DESIGN_P2_MESH_AS_DATA.md,
PLAN_GATHER_TO_READER.md (in progress), DESIGN_CONTACT_BY_SLAVE.md.

## 1. The floor to remove

Census (§4, c1_150 unrefined: 283,760 nodes, 1.14M DOFs, 69k roots), replicated per process:

| Component | MB |
|---|---|
| AMR mirror | 334 |
| Nodes + FieldVariables | ~300 (≈ 1 KB/node) |
| Imports | 256 |
| Global vectors, lumped operators, MPC, NED buffers | 220 |
| Constraints, facets, rigid bodies | 190 |
| DofManager per-node maps | 110 |
| Parsed input | 100 |
| Sets, field outputs | 50 |

- That totals about **1.6 GB** per process.
- With distributed elements (P3-P5), c1_150 measures 2.55 GB per process at 8x4.
- In the refined production window (1.94M DOFs), every term that grows with the node count is about
  1.7 times larger.

## 2. The key observation: local = integrated, no ghost layer

NEDMPI already computes only on `partition.dofs`, the sorted subdomain DOFs: everything the elements,
pieces and constraints of a process touch, plus the MPC closure (`subdomain.py:299-338`,
`subdomaininterface.py:105`).

- **Interface DOFs are integrated redundantly, bit for bit.** So nothing in an increment reads a DOF
  outside the subdomain.
- **What the global length serves**: only the global numbering, the whole-model readers (output,
  checkpoints, AMR, searches) and the full allgather before them (`nonlinearexplicitdynamicmpi.py:1110ff`).

So P6 needs no classical ghost layer:

- **Local DOFs = the subdomain DOFs.**
- **Owned** = the lowest rank integrating a DOF (exists: `ownedDofMask`, `subdomain.py:685`).
- **Received copies**, read-only, are needed only where a process reads values it does not integrate:
  - the rigid-body RP in option (b) of the contact note;
  - master-surface positions for a search;
  - transient gathers for whole-model readers.

These live in `domaindecomposition`, under the name "received from the owner".

**Order-preserving numbering.** The local numbering is the order-preserving compression of the global
one (sorted global ids), not "owned first".
- The MPC fold already runs on such a sorted subset and is bitwise today
  (`nonlinearexplicitdynamic.py:1216`, `mpcForceFold @ PPlain[dofs]`): the CSR columns keep their
  order.
- With local vectors, `partition.dofs` becomes `slice(None)` in NEDMPI too. That removes the `V[dofs]`
  indirection the teaching review criticised from NED's reading path.

## 3. Replicated and local

**Replicated, as compact data:**
- **The mesh**: today's `MeshElement` records (`mesh.py:124-170`). Converting them to arrays is
  optional (≈ 25 MB).
- **New: the mesh's nodes** (labels, reference coordinates, component). The Mesh becomes the single
  source of truth for nodes, as the professor's review asked.
- **The global DOF numbering as data**: per field, a node-membership array and prefix offsets; scalar
  variables at the end.
- **The AMR topology** (§7).
- Set and surface definitions as label arrays, sections, constraint definitions.
- Master-surface facets and rigid bodies.

**Local:**
- `Node` objects and their FieldVariables;
- NodeFields;
- the DofManager;
- DofVectors;
- the lumped operators;
- the MPC transformation;
- node sets, as their part held here.

## 4. Global ↔ local maps, and the DofManager

**Today's numbering** (`dofmanager.py:256-301`):
- field-major: each field takes a contiguous slice (`idcsOfFieldsInDofVector[field]`);
- within a field, nodes in NodeField order, which is model node order;
- then `dim` components per node.

**As replicated data**:
`globalDof(f, node, c) = fieldOffset[f] + rankInField[f][node]·dim_f + c`.
This needs about 4 B per node per field, computed from the mesh types plus the constraints'
`fieldsOnNodes`, the same way `_activateNodeFieldsFromMesh` does (`femodel.py:520-552`).

**The DofManager stays unchanged.** Given the local nodes in global order, it numbers them in exactly
that order-preserving way.
- `globalOfLocal` is a sorted int64 array; `localOfGlobal` is a `searchsorted` on it.
- One requirement: local nodes must be inserted in mesh order. A node acquired later (migration, AMR)
  is placed with `putNodesInMeshOrder`, analogous to `putElementsInMeshOrder` (`femodel.py:292`).
- The per-field slices and `writeDofVectorToNodeField`'s reshape (`dofmanager.py:724-750`) then stay
  valid locally.
- `idcsOfFieldsOnNodeSetsInDofVector` is built only for node sets held here (it is per node, part of
  the 110 MB).

**Exchanges stay keyed by global DOF.** `_TaggedContributions` (interface, loads, constraints) and
`SubdomainInterface` exchange tags `(globalDof, order)`; the local index is derived once per rebuild.
The layout check (`subdomain.py:407ff`) compares a digest of the replicated numbering.

## 5. Serving every whole-model read

| Reader | Under P6 |
|---|---|
| Node-field output, Ensight, monitors | Owned rows are gathered to rank 0 in global field order (PLAN_GATHER_TO_READER provides the path). Rank 0 holds one gathered field at a time (~15 MB). |
| Checkpoint `nodeFields/<field>/<entry>` (`femodel.py:982-986`) | The same gather; the file stays identical to a serial one. |
| Resume (`femodel.py:1017ff`) | Each process reads its rows by global position. The topology history is replayed on the replicated data, and Node objects are made for local nodes only. |
| AMR | Topology from the replicated arrays. New node values are interpolated by the owner of the parent element from local values; interface nodes are computed redundantly from identical bits. "fields resize & restore" (2.15 s) becomes local. |
| Field-output marker | Already reads gathered element results. |
| Recovery-error marker | Interim: a transient allgather of its one node field at topology checks (O(N), rare). Later: a per-element indicator from the element owners. |
| Contact search | Master-surface positions gathered, surface-sized; contact-note pieces search only local slave points. |
| Dirichlet, node forces | Node sets as their part held here, guarded like `ElementSetOfMesh` (`isComplete`/`requireComplete`). |
| Reaction or per-node output over a set | The set's rows gathered to rank 0, then summed exactly as serially (bitwise). |
| Rigid bodies | The RP node is held where its DOFs are read; its U is received each increment (contact note §4b). `updateRigidBodies` reads only that. |
| Lumped operators | Local (today they are shared to every DOF, `nonlinearexplicitdynamicmpi.py:460-487`). |

## 6. The biggest obstacles

### 1. The Node object graph

These hold or read Node objects:
- `model.nodes` (`femodel.py:97,147-167`): 68 reads in 25 modules, mostly generators and output;
- every NodeSet;
- elements (`setNodes`);
- `constraint.nodes`: the replicated copies hold whole surfaces, about 32k nodes for contactBottom
  alone;
- the replicated facets;
- `FieldVariable.values` views (`femodel.py:580-602`);
- DofManager dictionaries keyed by FieldVariable (`dofmanager.py:296-299`).

The rule, as P2 did for elements:
- Generators describe nodes into the mesh arrays.
- `createNodesOfMesh(isHeldHere)` makes Node objects after the partition. Serially it makes all of
  them.
- Readers of non-local nodes move to the mesh arrays.

The surface-sized nodes of replicated facets and constraint copies stay held until the contact pieces
exist (§9, P6.5). Until then that is about 11-16 % of the nodes on c1_150.

### 2. The DofManager's global assumptions

These are field slices over all nodes, per-node maps for every node set, scalar variables after all
node fields, and `nDof` used as the global size. All of them hold once the DofManager is given the
local set in order (§4). The global facts move to the numbering data.

### 3. The restart format

It is unchanged: per field, global node order, written by rank 0 (D2).
- The cost is a transient gather per field on rank 0.
- Per-process files are a later option, and only if rank 0's transient memory or the I/O time becomes
  a problem.

## 7. The AMR mirror as compact arrays

Today the mirror is about 4.4-4.7 KB of Python objects per root element:
- per element, a dict with 11 keys (`refinement.py:209`);
- NodeKey NamedTuples;
- `_keyOfPoint` (`rootentities.py:115`);
- a coordinate dictionary (`refinement.py:85`).

As a struct of arrays:
- int32 connectivity, n × 20;
- level, parent, first child, root and active flags;
- the lattice box origin and edge as int64 arrays;
- NodeKeys packed into int64 keys, held in a dict of ints or a sorted array;
- coordinates as an n × 3 float array.

That is about **0.6-0.9 KB per root**, so 334 MB becomes ~55 MB on c1_150.

The algorithms stay as they are. The gate is byte-identical plans, labels, histories and fingerprints.

## 8. Bit-identity

P6 relabels; it computes nothing differently.
- Per-DOF sums are ordered by tags, not by DOF number.
- The external work is an `fsum` (order-independent), and the critical step is a minimum.
- The MPC fold keeps its column order (§2).
- Checkpoints and outputs are gathered in global order.

Gates:
- a unit test that local and global numbering give bitwise-equal NED increments on one process;
- the harness;
- `h5cmp` of checkpoints;
- `--tolerance 0` at `-n 2/3/4`.

## 9. Memory per process (c1_150 census model, MB)

| Component | Today (8 procs) | P6 at 8 | P6 at 32 |
|---|---|---|---|
| Elements, wrappers, element maps (distributed) | 612 | 612 | 153 |
| AMR mirror | 334 | 55 | 55 |
| Nodes + FieldVariables (subdomain ×1.15) | 300 | 43 | 11 |
| Surface nodes until contact pieces | – | ≤ 50 | ≤ 50 |
| Mesh node arrays + numbering (new) | – | 15 | 15 |
| DofManager per node | 110 | 16 | 4 |
| Vectors, lumped operators, MPC, buffers | 220 | 32 | 8 |
| Constraints, facets, rigid bodies | 190 | 190 | 190 |
| Imports | 256 | 256 | 256 |
| Parsed input | 100 | 0 | 0 |
| Sets, field outputs | 50 | 15 | 10 |
| **Sum** (before allocator overhead) | **2,170** (2.55 GB measured) | **≈ 1,280** | **≈ 750** |

- Expect about **1.4-1.5 GB at 8** and **0.85-0.9 GB at 32**, against 2.55 and ≈ 1.75-2.0 today.
- The node-proportional rows grow about 1.7 times when refined.
- The new floor is the imports, the facets and the AMR topology (≈ 0.5 GB). The next levers are lazy
  imports on ranks other than 0, and slave facets held only by their pieces.

## 10. Phases, each with a bitwise gate

**Every gate**: `verify_refactor.sh` quick and full, `testfiles/mpi` `--tolerance 0` at `-n 2/3/4`
against fresh serial, checkpoints == serial, resume == straight, full pytest and suites, sphinx, lint,
and memory plus c1_150 timing alternated base/head.

| Phase | Content | Extra gate |
|---|---|---|
| P6.0 | Free the parsed input; lazy imports on non-root ranks | measure only |
| P6.1 | Mesh owns node arrays; `createNodesOfMesh(isHeldHere)`; `NodeSetOfMesh` with guard. Serial unchanged. | full serial suites |
| P6.2 | Global numbering as replicated data, asserted equal to the DofManager's on every deck and on c1_150 | assertion everywhere |
| P6.3 | NEDMPI local nodes, DofManager and vectors; exchanges keyed by global id; gathers for output and checkpoint; restart by rows. AMR and contact models keep global length, one logged rule as in P3. | 192k block memory |
| P6.4 | Compact AMR mirror (its own commit series), then local nodes under AMR, local interpolation, the recovery marker's transient gather | AMR decks; c1_150 replay byte-identical history |
| P6.5 | Contact and rigid bodies after contact-note C2/C3: pieces hold local nodes, master positions gathered, RP received | c1_150 8x4 and 32x1 bitwise, memory, no slowdown |
| P6.6 | Retire global length in distributed mode | – |

The whole-model fallback stays "every node held here".

## 11. Serial readability

A serial run holds every node, so `isHeldHere` is always true and `globalOfLocal` is never built.
DofManager, NodeField and NED are unchanged. NodeSet gains the ElementSet guard, and NED loses
`partition.dofs`. No ghost or owner word enters the serial reading path.

## 12. Decisions for the user

1. **Local = integrated DOFs, no persistent ghost layer**; read-only received values only where §5
   names them. Recommended: yes.
2. **Order-preserving local numbering** rather than owned-first: needed for the bitwise MPC fold.
   Recommended: yes.
3. **Keep `Node` objects for local nodes, and make the Mesh own node arrays** (P2 pattern), rather
   than arrays everywhere. Recommended: yes.
4. **NodeSet = the part held here**, guarded like ElementSet. Recommended: yes.
5. **Restart format unchanged**, through a rank-0 gather of owned rows; per-process files deferred.
   Recommended: yes.
6. **Order of work**: contact pieces (C2/C3) before P6.5, and a temporary global-length rule for
   AMR/contact models during P6.3. Acceptable?
7. **AMR mirror compaction inside P6.4** with its own byte-identity gate, or as a separate project?
8. **Recovery-error marker**: the interim transient allgather, or redesign it now as a per-element
   indicator?


## Decisions (user, 2026-10-08)

Core concept accepted: local DOFs = integrated ones in global order (no classical ghost layer), DofManager
unchanged, Node objects created only where held (nodes described in the mesh), node sets = the part held here
with the same guard as element sets, restart format unchanged via rank-0 gather. AMR mirror compaction inside
P6 (P6.4). Order: gather-to-reader -> contact C0 -> P6 (P6.5 waits for a contact decision).

## Decisions (user, 2026-10-14)

C0 merged and pushed (mn feat/mpi-explicit-dynamics-v2 at 1ac71f9d). Contact split C1–C3 NOT done now (C0 removed the
owner bottleneck; revisit beyond ~64 processes). Consequence for P6.5: there are no contact pieces; a whole contact
constraint stays with its evaluating process, which holds that constraint's surface nodes (received copies of the
ones it does not integrate). Searches stay out of the balance; constraints are evaluated before loads in NEDMPI;
load-balance tolerance stays 0.1.

## Decisions (user, 2026-10-15), after P6 round-1 review
1. Rigid-body STL float32 coordinates: left as is in P6 ("auxiliary nodes" kept, docstring says plainly: only to keep
   the float32 arithmetic bitwise); float64 loading later as its own upstream PR (then auxiliary nodes go).
2. Coordinates: one `FEModel.moveNodes` (mesh + held Node objects) used by tie and surfacesnap now; in P6.4 the AMR
   registry stops storing coordinates of model nodes and reads them from the mesh (syncNodeCoordinates disappears).
3. Field order: the model always keeps node fields in phenomena order (also after later activation).
4. Duplicate node labels raise.
