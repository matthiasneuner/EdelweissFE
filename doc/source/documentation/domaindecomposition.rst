Domain decomposition over MPI processes
=======================================

The explicit dynamic solver ``NEDMPI`` computes one model in several MPI processes, each process
computing one *subdomain* of it. It is the explicit solver of
:mod:`~edelweissfe.solvers.nonlinearexplicitdynamic` -- same options, same input decks, same
results -- distributed over processes, which may sit on several nodes of a cluster, instead of over
the threads of one process.

Running a job
-------------

Replace ``NED``/``NEDParallel`` by ``NEDMPI`` in the ``*solver`` keyword and start the job with an
MPI launcher:

.. code-block:: edelweiss

    *solver, solver=NEDMPI, name=theSolver
    courant-number=0.8
    output-frequency=1000

.. code-block:: console

    PYTHON_GIL=0 OMP_NUM_THREADS=1 mpirun -n 16 --bind-to core edelweissfe input.inp

``mpi4py`` and the ``metis`` library (both on conda-forge) are required, and nothing else changes:
a serial run never imports ``mpi4py``
(:mod:`~edelweissfe.domaindecomposition.mpienvironment` recognises an MPI launcher by the environment
variables it sets). Started without a launcher, ``NEDMPI`` runs as a single subdomain. Every other
solver refuses to run under a launcher with more than one process, since each process would compute
the whole model and write the same files.

Each process may in addition run the element loop on several threads, exactly as ``NEDParallel``
does (``OMP_NUM_THREADS``). The launcher must then give each process as many cores: Open MPI binds
a process to a single core by default, which confines all of its threads to that core -- the same
failure as described in :ref:`parallelization_thread_pinning`. For example, 4 processes with 8
threads each:

.. code-block:: console

    PYTHON_GIL=0 OMP_NUM_THREADS=8 mpirun -n 4 --map-by slot:PE=8 --bind-to core \
        -x PYTHON_GIL -x OMP_NUM_THREADS edelweissfe input.inp

Only rank 0 reports progress and writes output. An error is reported by every process that sees
it.

Which process creates which element
-----------------------------------

Every process reads the same input file and builds the same mesh, nodes, node sets and definitions
(materials, sections, constraints, steps). The element objects -- with their materials and states,
most of a model's memory -- are created in one of two ways, decided once per job, in one place
(:func:`~edelweissfe.domaindecomposition.elementdistribution.elementDistributionOfThisJob`), and
reported once on rank 0:

* **Distributed** -- ``Distributed model: each of the N processes creates only the elements it
  computes``. The mesh is partitioned *before* any element exists, and each process creates the
  elements of its own subdomain, plus the few others described below. The memory of the elements is
  divided among the processes.
* **The whole model on every process** -- ``Whole model on every process because: ...``. Every
  process creates every element, as a serial run does, and keeps the states of the elements other
  processes compute current by synchronizing them whenever the whole model is read. This is the
  fallback for what still reads, or changes, the whole model during a run.

The fallback rule
~~~~~~~~~~~~~~~~~

:func:`~edelweissfe.domaindecomposition.elementdistribution.reasonsForTheWholeModel` reads the input
file and names every reason to hold the whole model on every process; the job is distributed only if
there is none:

* a **model modifier** (``*modelModifier`` -- adaptive refinement above all): it changes the mesh
  during the run, and its topology logic, marker and state transfer read the whole mesh;
* a **constraint** (``*constraint`` -- contact, ties, and the like): each is evaluated whole by one
  process, searches its whole surface, and may couple nodes of any subdomain;
* a **generator that does more than describe the mesh**: ``executePythonCode`` and ``cubit`` act on
  element objects while the mesh is being described, the contact-facet generator
  (``surfaceElementGenerator``) and the discrete rigid body generator make elements in every process
  themselves. Every generator says so through
  :attr:`~edelweissfe.generators.base.generatorbase.GeneratorBase.wholeModelReason`; a generator
  that does not say it only describes the mesh is assumed to need the whole model.

Moving these readers to gathers of their own -- refinement on replicated mesh data with owner-local
elements, contact through surface-sized exchanges -- removes them from the rule one by one.

What a distributed process creates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~edelweissfe.domaindecomposition.elementdistribution.DistributedElements` partitions the mesh
with the same function and the same weights the subdomain uses
(:func:`~edelweissfe.domaindecomposition.partitioning.partitionElementsOfMesh`), right after the
mesh is described and before the elements are made
(:func:`~edelweissfe.helpers.inputfilehelpers.fillFEModelFromInputFile`). A process then creates

* the elements it **computes** -- its part of the partition; and
* every element carrying a **load** -- on the surface of a distributed load, or in the element set
  of a body load, of any step -- that shares a node with one of its own elements.

The second kind exists for its loads only: a load is not exchanged between processes, each process
adds the loads at the degrees of freedom it integrates itself, in the order of the load's elements,
which is what keeps the result bit-identical to a serial run (see `An increment`_). Such an element
is neither computed nor reported by the process. A subdomain reaching a loaded element in another way
-- through a multi-point constraint, say -- would miss its load, and is refused
(:meth:`~edelweissfe.domaindecomposition.subdomain.Subdomain.distributedLoadsOnSubdomain`). An
element described after the partition, which no process would compute, is refused as well.

The subdomain of the solver adopts this partition instead of computing its own.

.. _domaindecomposition_mesh_to_elements:

Mesh to elements
~~~~~~~~~~~~~~~~

Every model is built in two stages (see :doc:`mesh`): the input file and the mesh generators
*describe* the mesh as data -- ``model.mesh``, a :class:`~edelweissfe.models.mesh.Mesh` with every
element's number, type, provider and node labels, the element sets as lists of element numbers and
the surfaces by their element sets -- and the element objects are then *made* from it by
:meth:`~edelweissfe.models.femodel.FEModel.createElementsOfMesh`, for every element the model's
:class:`~edelweissfe.models.elementdistribution.ElementDistribution` creates here. Four properties
make a distributed model possible:

* **The layout of the degrees of freedom follows from the mesh.** Which fields an element has at
  which node depends on its type only, so the fields at the nodes are activated from the mesh, not
  from element objects (:meth:`~edelweissfe.models.mesh.Mesh.typeOf`). The global numbering of the
  degrees of freedom is therefore the same in every process *by construction*, whichever elements it
  created. Vectors stay global-length: a process holds the solution at every node, and receives it
  from the owners of the degrees of freedom whenever the whole model is read.
* **An element set is the part of the set created here.** Each
  :class:`~edelweissfe.sets.elementset.ElementSet` of the model holds the created elements of its set
  in the mesh, and knows whether that is all of them (``isComplete``). So do surfaces and sections.
  The nodes of a set (``extractNodeSet``) are taken from the mesh where only part of it was created,
  and so are the same in every process.
* **Whole-model readers gather, or refuse.** A reader that needs every element of a set or of the
  model either goes through the gather path of the element distribution (see `Where the whole model is
  read`_), or says that it needs the whole set
  (:meth:`~edelweissfe.sets.elementset.ElementSet.requireComplete`) or the whole model
  (:meth:`~edelweissfe.models.femodel.FEModel.requireCompleteMesh`) and fails loudly on a distributed
  model, instead of giving a silently partial result.
* **An element can be created at any time, anywhere.** An element is completely described by the
  mesh, its section (assigned through its sets) and its state vector, so
  :meth:`~edelweissfe.models.femodel.FEModel.createElementOfMesh`,
  :meth:`~edelweissfe.sections.base.sectionbase.Section.assignSectionToElement` and
  ``setStateVars`` recreate it in another process -- which is what moving an element between
  processes needs (see `Load balancing`_).

Contact facets and the point masses of rigid bodies are made by their owners in every process; they
are added to the mesh as well (:meth:`~edelweissfe.models.mesh.Mesh.addElementMadeByOwner`), and the
facets are cut from the surface as described in the mesh, not from element objects. A refinement
adds its children to the mesh and creates them from it. Both happen in models that hold the whole
model on every process.

Subdomains, interface, ownership
--------------------------------

The elements are partitioned by `METIS <https://github.com/KarypisLab/METIS>`_
(:mod:`~edelweissfe.domaindecomposition.metis`, :func:`METIS_PartMeshDual`): the dual graph of the
mesh -- elements adjacent when they share a face, or an edge in 2D -- is split into parts of balanced
weight, minimising the total communication volume. The partition is computed on the mesh -- element
numbers, connectivity, and the size of each element type -- not on element objects, on rank 0, and
broadcast. The constraints are dealt out by name, one process each.

A process *integrates* every degree of freedom its elements and constraints touch -- its subdomain
degrees of freedom. A degree of freedom touched by several processes lies on their *interface*. Each
degree of freedom is also *owned* by exactly one process, the lowest-ranked one integrating it; the
owner is the process that counts a quantity which must be counted once, such as a kinetic energy
(:class:`~edelweissfe.domaindecomposition.subdomaininterface.SubdomainInterface`).

A multi-point constraint -- a tie, a hanging node -- folds a slave's force onto its masters and
interpolates the slave's velocity from theirs. A process integrating any degree of freedom of such
a group therefore integrates all of them.

An increment
------------

An increment is the central-difference increment of ``NED``, unchanged. ``NED`` runs it over the
elements, constraints and degrees of freedom of its
:class:`~edelweissfe.solvers.base.modelpartition.ModelPartition` -- four members: the elements
computed here, the constraints evaluated here, the degrees of freedom integrated here, and which of
those are owned here. For ``NED`` that is the whole model, and the degrees of freedom are
``slice(None)``, so that ``V[dofs]`` is the whole vector; for ``NEDMPI`` it is the subdomain of the
process, as a :class:`~edelweissfe.domaindecomposition.subdomain.Subdomain` defines it, and the
degrees of freedom are a sorted index array.

Everything a process must exchange with the others, ``NEDMPI`` adds in overrides of a few methods of
``NED``, each calling the ``NED`` method or the shared function it extends:

==========================================  =========================================================
``NEDMPI`` method                            adds
==========================================  =========================================================
``partitionModel``                          the subdomain, instead of the whole model
``computeElements``                         the interface exchange of the element forces
``assembleLumpedDiagonal``                  the same for the lumped inertia and damping, once per mesh
``assembleLoads``                           restricts the loads to the elements reaching the subdomain
``assembleConstraintForces``                evaluates the own constraints, shares all forces
``getCriticalTimeStepForExplicitDynamics``  the minimum over the subdomains
``energyBalanceTerms``                      the sums over the subdomains
``acceptIncrement``                         the synchronization and rebalancing of an output increment
``updateConstraintConnectivity``            the synchronization before a contact search
``updateConnectivityOf``                    runs a search on the constraint's process only
``updateTopology``                          the agreement on a topology update
``writeIncrementOutput``                    the agreement on writing the output
``applyStepActionsAtStepEnd``               the synchronization at the end of a step
==========================================  =========================================================

Each process evaluates the kernels of its own elements. At an interior degree of freedom of its
subdomain that is every contribution there is. At an interface degree of freedom the contributions
of the neighbouring subdomains' elements are missing, and the neighbours exchange them -- the
individual element contributions, not their partial sums
(:class:`~edelweissfe.domaindecomposition.subdomaininterface.InterfaceForceAssembly`). Every process
then sums all contributions at the degree of freedom in the order of the elements in the model,
which is the order a single process computing the whole model sums them in.

Floating-point addition is not associative, so this order is what makes the result independent of
the decomposition: the force at every degree of freedom is the same bits as without decomposition,
and so is everything computed from it. The loads are added in the same way -- a distributed load or
body force on a neighbour's element is evaluated wherever it reaches, rather than exchanged -- and
the constraint forces, each computed by one process, are shared with all and added in model order.
A run on any number of processes is bit-identical to ``NED`` on the same input.

The volume exchanged per increment is that of the interface and the constraints. The lumped mass
and damping are assembled once per mesh by each process from its own elements, completed at the
interface like the forces and then shared from the owners, so that every process holds them, the
same bits as without decomposition, at every degree of freedom of the model. The critical time step
is the minimum over the subdomains.

Where the whole model is read
-----------------------------

A process keeps current only what it computes. Field outputs, output managers, restart checkpoints,
the marker of an adaptive refinement and the refinement itself, and a contact search all read more
than one subdomain. Before each of them every process receives the current solution at every degree
of freedom from its owner (the vectors are global-length in every process). What happens to the
element states depends on how the model is held.

**The whole model on every process.** Every process receives the current state of every element and
stateful constraint from the process that computed it -- except before a contact search, which reads
positions only (:mod:`~edelweissfe.domaindecomposition.statesynchronization`). The states travel
through the same interface restart checkpoints use: whatever a checkpoint must carry to resume a run
is what another process must receive to continue it. Every reader then reads the model as a serial
run does. This synchronization happens every ``output-frequency`` increments, at every
``contact-update-frequency`` search, and at the end of a step.

**Distributed.** No element state is synchronized -- a process holds no element it does not compute,
apart from the loaded elements above, which nothing reads -- and each whole-model reader goes through
the *gather path* of the element distribution
(:class:`~edelweissfe.models.elementdistribution.ElementDistribution`, whose serial base answers with
what is here):

======================================  =================================================================
Reader                                  How it reads a distributed model
======================================  =================================================================
element field output over a set         ``resultsOfWholeSet``: each process collects the results of the
(``>>perElement``)                      elements of the set it computes, and they are gathered to every
                                        process, by mesh element number, in set order -- so a field
                                        output is the same in every process, as without decomposition
node field output over an element set   the nodes of the whole set, from the mesh; the node fields are
(``>>perNode, elSet=``)                 global-length
Ensight                                 rank 0 draws the geometry of a partial set from the mesh
                                        (``visualizedElementsOf``) and writes the gathered results
monitor, conditional stop               read field outputs
restart checkpoint                      before the output of an output increment, every process sends
                                        the states of the elements it computes to rank 0
                                        (``gatherStatesForCheckpoint``, only if rank 0 writes
                                        checkpoints); rank 0 writes them in the format of a serial
                                        checkpoint and releases them
======================================  =================================================================

Still guarded, and refused loudly on a distributed model: an expression field output over an element
set (``>>fromExpression, elSet=``, which reads the elements themselves), the mesh plot,
``meshDataToFile``, the element set marker, adaptive refinement, ``surfaceSnap`` and the P1 topology
classification. None of them can run in a distributed job today that the fallback rule did not
already send to the whole model, except the first three, which fail at their setup.

The gathers of the field outputs happen inside the output step that every process agrees on: each
process reaches them in the same order, before anything that runs in one process only (the output
managers of rank 0), so no process can wait for one that already left.

The contact search itself, at a contact update and at a topology check alike, runs on the process
that evaluates the constraint only: nothing but that evaluation reads its outcome. The constraint
forces are shared with the degrees of freedom of the owner's search. Whether a search changed a
footprint is decided by the owners and agreed on by all processes.

Restoring a constraint's state restores it completely -- a contact constraint adopts the owner's
search, and with it the nodes it couples -- so after a synchronization every process' copy of a
constraint is its owner's. Between two synchronizations a copy keeps that footprint, with the mesh
refreshes since applied to it as to the owner's. Its degree-of-freedom indices are never read, since
a constraint is evaluated, and its forces exchanged, with its owner's indices only. The footprint
itself is read in one place: a topology change activates fields on the nodes of every constraint
copy. That is harmless because a topology update always follows a synchronization -- a topology
check is due only on an output increment, the start of a step follows the end of the last one or a
checkpoint -- and it is asserted rather than assumed: before every topology update the nodes and
fields every constraint couples are compared across processes
(:meth:`~edelweissfe.domaindecomposition.subdomain.Subdomain.requireConstraintCopiesCurrent`).

Adaptive refinement
-------------------

Adaptive refinement holds the whole model on every process (see `The fallback rule`_). Because every
process holds the complete, synchronized model at a topology check, every process runs
the same refinement -- the same marker on the same field output, the same state transfer from the
same parent states -- and arrives at the same refined model. That is verified, not assumed: after
every build of the equation system, a fingerprint of the mesh (element numbers and connectivity),
the degree of freedom of every node of every field, the node coordinates and the size of the system
is compared across all processes, and the run stops if any two differ. It is made from the mesh and
the nodes, which every process holds whole, so it serves a distributed model as well. The refined model is then partitioned afresh. Nothing migrates, because every process already
holds every element.

Load balancing
--------------

The first partition weighs an element by its number of degrees of freedom. A softening material
costs more where it softens -- a return mapping needs more iterations -- so a partition balanced at
the start drifts out of balance as damage localizes. Every element kernel is therefore timed, and on
an output increment, as a step of the increment loop of its own right after the synchronization of
the output, the model is repartitioned with the measured
costs as weights whenever the slowest process has fallen more than ``load-balance-tolerance``
(default 0.1) behind the mean. The partition then depends on measured timings; the result does not,
since it does not depend on the partition at all.

A **distributed** model keeps its first partition, and says so once (``Load balancing is off``):
an element changing process would have to be *migrated* -- created by its new process from the mesh
(:meth:`~edelweissfe.models.femodel.FEModel.createElementOfMesh`), given its section and element
properties, and sent its state (``setStateVars``) by the old one, which drops it -- and the
subdomain's degree-of-freedom indices, lumped operators and loaded elements rebuilt for the new
elements. That is designed, but not implemented yet. The kernels are not timed then.

Output and restart
------------------

Only rank 0 creates output managers. A restart checkpoint is the last output of an output
increment, written after the synchronization of that output, so rank 0's copy of the model holds
every element and every constraint as the process computing it left it, and the external work of
the whole model. A distributed model gathers the element states to rank 0 for it instead (see
`Where the whole model is read`_); the file is the same, bit for bit, as a serial run's. It is an ordinary checkpoint of the whole model, so a run can be resumed by
``NEDMPI`` on any number of processes, or by ``NED``. The topology check due after that increment
runs at the start of the next one, after the checkpoint, so a resumed run performs it exactly as the
uninterrupted one does (see :doc:`restart`).

On a resume every process restores the whole model from the checkpoint -- every constraint adopting
the checkpointed state of its owner -- and so starts from the same model; rank 0 continues with the
external work of the whole model, the others with none of it. A distributed process restores the
elements it created and skips the others
(:meth:`~edelweissfe.models.femodel.FEModel.readRestart`); a checkpoint written by a serial run, by
the whole model on every process, or by a distributed run can be resumed in either way.

Failures
--------

A failure while evaluating the elements or the constraints -- a material that cannot integrate at
the stable step, above all --, in a constraint's connectivity search, in a topology update (the
marker, the refinement, the mesh refresh), or in finalizing the output (a conditional stop, too) is
agreed on by all processes before any of them raises
(:meth:`~edelweissfe.domaindecomposition.subdomain.Subdomain.agreedOnByAllParts`), and every process
ends the step the same way; a cutback requested anywhere is raised everywhere, with the smallest
size requested. A failure anywhere else aborts all processes, and so does an interrupt (``Ctrl+C``)
of any one of them: the others would otherwise wait forever for the one that stopped
(:mod:`~edelweissfe.domaindecomposition.mpienvironment`).

A model with an element that exposes no state (``getStateVars``) is refused on more than one
process: its state could not be sent to the process writing the output and the checkpoints, nor to
the one computing it after a repartition.

Which way a job held its model does not show in its result. ``testfiles/mpi/check_element_distribution.py``
runs every deck of ``testfiles/mpi`` under the launcher and checks, from the elements each process
created and computed, that it ran distributed or with the whole model as expected; the MPI workflow
runs it over 2 and 3 processes.

Verifying bit-identity yourself
-------------------------------

The decks in ``testfiles/mpi`` run the same models as their serial counterparts, with
``solver=NEDMPI``. Run each set once serially with ``NED`` and once on several processes, let the
test runner write the final solution of every deck (``--create`` writes ``U.ref`` with 18
significant digits, which reproduces a double exactly, and only rank 0 writes), and compare the
files byte for byte:

.. code-block:: console

    cp -r testfiles/mpi serial && cp -r testfiles/mpi decomposed
    sed -i 's/solver=NEDMPI/solver=NED/' serial/*/*/test.inp
    export PYTHON_GIL=0 OMP_NUM_THREADS=1
    run_tests_edelweissfe serial/edelweiss-only --create
    mpirun -n 3 --bind-to none -x PYTHON_GIL -x OMP_NUM_THREADS \
        run_tests_edelweissfe decomposed/edelweiss-only --create
    for reference in serial/*/*/U.ref; do
        cmp "$reference" "decomposed/${reference#serial/}" || echo "DIFFERS: $reference"
    done

Nothing printed by the loop means every deck is bit-identical; ``marmot`` in place of
``edelweiss-only`` runs the decks that need Marmot (some need its private materials). Compare against
a serial run on the same machine rather than against the committed ``U.ref`` files: those were
written on another one, and a serial run differs from them in the last digits wherever the two
machines' math libraries do. ``run_tests_edelweissfe testfiles/mpi/edelweiss-only`` without
``--create`` compares against the committed files within an absolute tolerance of 1e-6.

Limitations
-----------

* Memory: a distributed process holds its own elements, but every process holds the whole mesh,
  every node with its fields, and global-length vectors. Jobs under the fallback rule hold the whole
  model on every process.
* A distributed model is not rebalanced (no migration yet).
* Adaptive refinement is computed by every process, in full, on the whole model.
* A constraint is evaluated whole, by one process; a single large contact constraint is not split.
* Only the explicit dynamic solver is decomposed.

Package reference
-----------------

.. automodule:: edelweissfe.domaindecomposition
   :members:

.. automodule:: edelweissfe.domaindecomposition.mpienvironment
   :members:

.. automodule:: edelweissfe.domaindecomposition.metis
   :members:

.. automodule:: edelweissfe.domaindecomposition.subdomain
   :members:

.. automodule:: edelweissfe.domaindecomposition.elementdistribution
   :members:

.. automodule:: edelweissfe.domaindecomposition.partitioning
   :members:

.. automodule:: edelweissfe.domaindecomposition.subdomaininterface
   :members:

.. automodule:: edelweissfe.domaindecomposition.statesynchronization
   :members:
