# Design: real distributed memory for the explicit MPI solver (draft, 2026-10-02)

Status: **decided 2026-10-02** (D1 replicated mesh arrays, D2 output/restart via rank 0, D3 bitwise with NED is a hard gate, D4 explicit only, D5 interim memory regression after P1 accepted). **P1 DONE** (feat `b06b63c0`): stand-in element removed, elements parallelization-agnostic; full harness bitwise; c1_150 8x4 peak RSS 7.5 GB/rank (handoff had 8.1 GB/rank *with* the stand-in — P0 to reconcile). P0 census done (§4). **P2 DONE 2026-10-06** (feat/mesh-as-data merged into feat/mpi-explicit-dynamics-v2; reviewed, review fixes applied; bitwise; one-METIS-part build gives the identical DOF layout). Next: P3 (see DESIGN_P2_MESH_AS_DATA.md §10 follow-ups). Base: `feat/mpi-explicit-dynamics`
(after the teaching-fit refactor, handoff §5c). Facts below come from four code maps made today
(model construction, run-time readers, AMR, constraints); file references are to that branch.

## 1. Where we are, and why it cannot scale

NEDMPI today: **every process builds and holds the complete model**, computes a METIS subdomain, and
synchronizes vectors and all element/constraint states (Allgatherv) before anything reads the whole
model (output, markers, refinement, contact search, step end). Memory per process is therefore ≈ the
whole model, independent of the number of processes. `DistributedMarmotElement` trimmed the largest
item (the C++ Marmot elements: 8.2 → 1.8 GB/process on the 192k block) by faking elements a process
does not compute. That put parallelization into the element hierarchy (`BaseElement.computingObject`,
`startComputingHere`, the Cython wrapper) — which is wrong: **an element must not know about
parallelization.**

Target: **each process creates only the elements it computes.** Memory then falls with the number
of processes, and the stand-in element has no reason to exist.

## 2. What depends on the whole model today (summary of the maps)

| Area | Whole-model assumption | Inherent? |
|---|---|---|
| Model construction | Input parsed into strings, then **objects** directly (`abqmodelconstructor.py:102-247`, boxgen); no array stage. Sets/surfaces/sections hold **element objects**; `*elset` by label raises on a missing element. Numbers from global monotonic counters (`femodel.py:179-258`). | No — but it is the deepest change |
| DOF layout | Global numbering over all nodes; a field is activated on a node by a local element/constraint (`femodel.py:719-742`), so layouts would differ per rank. Layout fingerprint assumes identity. | No |
| Increment (elements, loads, Dirichlet, geostatic, material change) | Whole sets, but trivially filterable to local entities; loads already use the elements touching the subdomain. | No |
| Field outputs, monitor, conditional stop, Ensight, restart | Every rank evaluates every field output; rank 0 writes; checkpoint = every element state + every node field + topology history. | **Yes**: needs a gather to rank 0, collective reductions, or per-rank files |
| AMR | Initialization over all elements; global Doerfler ranking/budget; 2:1 balance and hanging-node detection over the whole active mesh (unbounded reach); new node/element numbers from global counters in creation order; restart replays the history on the whole root mesh. | Partly: the *topology* logic is global by nature; element objects and states need not be |
| Contact / tie / MPC | Data is **surface-sized**; searches read current coordinates of all candidates (global rMax broadphase); constraint evaluated whole by one owner; MPC operator `nDof×nDof`, closure keeps a slave and its masters on one process; facets live in `model.elements` and draw global numbers. | Surface-sized gathers suffice |
| Rigid bodies | Reference-node DOFs read everywhere. | Tiny broadcast |

## 3. Proposed architecture (one sentence per layer)

1. **Mesh as data, replicated.** Every process parses the input and holds the mesh as lightweight
   arrays — node labels and coordinates, element numbers, types and connectivity, set and surface
   membership as label arrays. Order 100–200 bytes per element and node, instead of today's
   kilobytes of Python and C++ objects. Global numbering stays meaningful because it is computed from
   the same data everywhere, by construction rather than by a fingerprint check.
2. **Partition before elements exist.** METIS runs on the connectivity arrays; each process then
   creates element objects — ordinary, parallelization-agnostic Marmot or Python elements — only for
   its own elements. Sets, sections and surfaces resolve labels to the local objects; a missing label
   means "not here", not an error.
3. **Nodes and DOF layout from the mesh data.** The DOF layout is derived from element types and
   connectivity arrays (which fields a type puts on its nodes), identical on all processes without
   any element object. Vectors stay global-length in the first stage (c1_150: 1.94 M DOFs ≈ 15 MB per
   vector); local numbering with ghost nodes is a later stage (§5, P6).
4. **The increment** is today's `Subdomain` partition, minus the whole-model synchronization: owned
   elements, interface force exchange in model element order (bitwise as today), loads and boundary
   conditions filtered to local entities.
5. **Whole-model reads become explicit collectives.** Field outputs over sets gather (or reduce) to
   rank 0, which alone writes Ensight, monitors and checkpoints (D2). A checkpoint keeps today's format:
   rank 0 receives element states and node fields at write time — transient, streamed per chunk.
6. **AMR: replicated topology, owner-local elements.** The octree / hanging-node / balance logic runs
   on the replicated mesh data (it is global by nature and cheap as data); marker values are
   allgathered (one number per element); each owner creates its children, transfers state from its
   own parent, and drops the parent. Rebalancing **migrates** element states (state vector + the
   section that recreates the element on the receiving process).
7. **Contact and ties: surface-sized collectives.** Master surface coordinates are gathered for a
   search (surface, not volume); the constraint owner evaluates with an Allgatherv of its footprint's
   U, as forces are shared today. MPC groups stay closed on one process.

What this deliberately **keeps**: one input file read by every process; global labels and DOF
numbers; bitwise identity with serial NED (the strongest test we have, and the existing harness
proves it every step); rank 0 as the only writer.

What it deliberately **defers**: distributed node storage and local DOF numbering (P6), parallel
output files, distributed AMR topology, the implicit solvers.

## 4. Memory per process — P0 census (c1_150, one process; mpi_explicit_benchmarks/memory_census/)

Model: 65,686 GC3D20R + 3,480 C3D8 + 25,622 facets; 283,760 nodes; 1.14 M DOFs. Total ~7.2 GB (peak 7.8).

| Component | MB | Per unit | Under §3 |
|---|---|---|---|
| **Marmot C++ elements** (construction 3,700 + material init 425) | **~4,100** | ~56–65 KB/GC3D20R | distributed |
| Marmot wrapper, Python side (incl. state vectors 80) | 723 | 10.5 KB/el | distributed |
| DofManager per-element index arrays + element plans | ~75 | 1.1 KB/el | distributed |
| **AMR mirror** (octree, root tables, coordinate registry; Python objects) | **334** | 4.4 KB/root el | replicated |
| Nodes + FieldVariables + field linking | ~300 | ~1 KB/node | replicated (until P6) |
| Imports (libMarmot, numpy, scipy, sympy, pyvista) | 256 | – | replicated |
| Global vectors, lumped operators, MPC sparse, NED buffers | ~220 | – | replicated (global-length) |
| Constraints + facets + rigid bodies (surface-sized) | ~190 | – | replicated |
| DofManager per-node index arrays | ~110 | 400 B/node | replicated |
| Parsed input (kept alive by the driver for the whole run) | 100 | – | replicated (can be freed) |
| Sets, field outputs | ~50 | – | replicated |
| Allocator retention / unattributed | ~670 | – | – |

Consequences:
1. Owning only computed elements (P3) removes ~68 % (≈ 4.9 GB, i.e. 0.6 GB/process at 8, 0.15 at 32).
2. With today's replicated structures the floor is **~1.6 GB/process**, not a few hundred MB: per process
   ≈ 2.2 GB at 8 and 1.75 GB at 32 processes (today 7.2–7.5 GB). Reaching a few hundred MB needs the
   AMR mirror and nodes/FieldVariables as compact arrays — the same "mesh as data" move as P2.
3. ~56–65 KB of C++ per GC3D20R element is a Marmot-side lever (likely fixed-size per-QP matrices),
   independent of MPI and worth it for serial runs too — out of scope here, worth an issue.
4. Unreconciled: the handoff recorded 8.1 GB/process at 8x4 *with* the stand-in, P1 measured 7.5 GB
   without, although the stand-in should have saved ~3 GB at 8 ranks. Likely a transient setup peak
   (temporary Marmot elements during section assignment) or a different measurement; P3 measures the
   real thing anyway.

## 5. Phases, each closed by the bitwise harness

| Phase | Content | Gate |
|---|---|---|
| P0 | Memory census of c1_150 per process by component (elements, nodes/fields, DofManager, sets, AMR mirror, facets, MPC). Confirms where the bytes are before anything moves. | numbers in this doc |
| P1 | Remove `DistributedMarmotElement`, the `BaseElement`/wrapper additions and the class swap: elements agnostic again. Memory temporarily back to plain replication. | `verify_refactor.sh full` |
| P2 | **Mesh as data** in the model construction: generators and the Abaqus constructor fill mesh arrays first; element objects are created from them through one predicate (serial: all). Sets/sections/surfaces resolve labels. DOF layout from mesh data. Serial behaviour unchanged. | harness + full serial suites (this touches every input path) |
| P3 | Partition before creation; NEDMPI creates owned elements only; whole-model synchronization replaced by gathers/reductions to rank 0 for field outputs, output managers, checkpoints; loads/BCs filtered. **No AMR, no contact yet** (refused under MPI with a clear message). | harness without AMR/contact decks; memory measured |
| P4 | AMR with owner-local elements on replicated topology; state migration on rebalance; restart replay. | AMR decks + c1_150 AMR |
| P5 | Contact, ties, rigid bodies via surface-sized collectives. | contact decks + c1_150 full |
| P6 (later) | Distributed nodes, local DOF numbering with ghosts; per-rank output if rank 0 becomes the bottleneck. | — |

P2 is the deep, risky one: it changes how every model is built. It is also the one that makes the
code *more* teaching-friendly — "the mesh is data, elements are made from it" is the textbook
picture — so it should be designed in its own short document before code.

## 6. Decisions needed

- **D1 Replicated mesh data** (arrays on every process) as the first target, versus a fully
  distributed mesh from the start (parallel input reading, distributed AMR topology à la p4est).
  Recommendation: replicated arrays — orders of magnitude smaller than today, keeps AMR tractable.
- **D2 Output and restart through rank 0** (gather at output/checkpoint time; checkpoint format
  unchanged; rank 0 needs transient memory for one gathered field) versus per-rank files now.
  Recommendation: rank 0 first.
- **D3 Bitwise identity with serial NED stays a hard requirement.** Recommendation: yes.
- **D4 Scope: explicit only** (NEDMPI); implicit solvers untouched. Recommendation: yes.
- **D5 Interim memory regression** after P1 (until P3 lands) is acceptable on the unpushed branch.

## 7. Known defects found on the way

### 7.1 A load step action resumed from a checkpoint fails at the end of its step (serial, upstream)

Found in P3 (2026-10-06) while writing a distributed restart deck; **not MPI-related**: it reproduces
serially with `NED` on upstream `804d3817` (worktree `EdelweissFE-upstream`) and on
`feat/owned-elements`. A `>>bodyForce` (and likewise a `>>distributedLoad`) that is restored from a
checkpoint raises when its step ends:

```
  File ".../edelweissfe/stepactions/bodyforce.py", line 260, in applyAtStepEnd
    self._forceAtStepStart += self._delta * self._amplitude(1.0)
ValueError: non-broadcastable output operand with shape () doesn't match the broadcast shape (2,)
```

(`distributedload.py`, line 377, `self._magnitudeAtStepStart += self.delta * self.amplitude(1.0)`:
same error with shape `(1,)`.) Cause, as far as read: the accumulated load at step start starts as
the Python float `0.0`, is checkpointed (`checkpointedState`, declared `np.ndarray`) and restored as a
0-d array; the in-place `+=` of a vector into a 0-d array fails, where `float += vector` rebinds.
An uninterrupted run never restores it and is unaffected.

Minimal reproduction (edelweiss-only, ~2 s), two decks in sibling directories `write/` and `resume/`:

```
*material, name=VonMises, id=le, provider=edelweiss
210000, 0.3, 550, 1000, 200, 1400, 1
*section, name=section1, thickness=1.0, material=le, type=plane
all
*job, name=defect, domain=2d
*solver, solver=NED, name=theSolver
output-frequency=10
*modelGenerator, generator=planeRectQuad, name=gen
l=10
h=10
nX=2
nY=2
elType=CPE4
elProvider=edelweiss
*output, type=restart, name=restart
writeInterval=1, baseName=restart, numberOfFilesToKeep=1
*step, type=adaptiveForExplicitSimulations, solver=theSolver
maxInc=1, minInc=1e-12, maxNumInc=50, maxIter=25, stepLength=1.0
>>dirichlet, name=bottom, nSet=gen_bottom, field=displacement, 1=0, 2=0
>>bodyForce, name=gravity, elSet=all, forceVector='0.0, -1.0', f(t)=t
```

`resume/test.inp`: the same, with the two `*output, type=restart` lines replaced by
`*restart, readFrom=../write/restart_0.h5` and `maxNumInc=1000` (the write run stops at its
increment limit, time 0.376, before the step ends; the resumed run reaches the step end). Run
`PYTHON_GIL=0 edelweissfe test.inp --noplot` in `write/`, then in `resume/`: the second fails as
above. Because of it, the distributed restart pair in `testfiles/mpi` (`NEDRestartDistributed*`)
carries no load.
