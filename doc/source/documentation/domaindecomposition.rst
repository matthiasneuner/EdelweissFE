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
(:func:`~edelweissfe.domaindecomposition.distributedelements.elementDistributionOfThisJob`), and
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

:func:`~edelweissfe.domaindecomposition.distributedelements.reasonsToReplicateElements` reads the input
file and names every reason to hold the whole model on every process; the job is distributed only if
there is none:

* a **model modifier that reads element objects of the whole model** (``*modelModifier``, e.g.
  ``surfaceSnap``): it changes the mesh during the run from what it reads. Every model modifier says
  so through :attr:`~edelweissfe.modelmodifiers.base.modelmodifierbase.ModelModifierBase.replicatedElementsReason`;
  one that does not say it reads only the mesh is assumed to need the whole model. Adaptive
  refinement (``hAdaptivity``) reads only the mesh and runs distributed (see `Adaptive refinement`_);
* a **constraint not known to read only what every process holds** (``*constraint``): every
  constraint says so through
  :attr:`~edelweissfe.constraints.base.constraintbase.ConstraintBase.replicatedElementsReason`
  (:mod:`~edelweissfe.constraints.base.wholemodel`). Ties (``tie``), penalty contact
  (``surfaceToDeformableSurfacePenalty``, ``nodeToDeformableSurfacePenalty``,
  ``surfaceToDiscreteRigidBodyPenalty``, ``nodeToDiscreteRigidBodyPenalty``,
  ``nodeToRigidSurfacePenalty``) and the other forces-only penalty constraints
  (``equalValuePenalty``, ``directionalSpringPenalty``) run distributed (see `Contact, ties and
  rigid bodies`_). The constraints with Lagrange multipliers (``rigidBody``, ``linearizedRigidBody``,
  ``equalValueLagrangian``) and the indirect load control (``penaltyIndirectControl``) are for the
  implicit solvers and keep their reason, as does any constraint not verified by a distributed test
  case;
* a **generator that does more than describe the mesh**: ``executePythonCode`` and ``cubit`` act on
  element objects while the mesh is being described. Every generator says so through
  :attr:`~edelweissfe.generators.base.generatorbase.GeneratorBase.replicatedElementsReason`; a generator
  that does not say it only describes the mesh is assumed to need the whole model. The contact-facet
  generator (``surfaceElementGenerator``) and the discrete rigid body generator are not among them:
  they read the mesh and make elements of their own in every process;
* a **generator run after the keywords** (``executeAfterManualGeneration=True``) **that describes
  elements of the mesh**
  (:attr:`~edelweissfe.generators.base.generatorbase.GeneratorBase.describesElementsOfMesh`): it
  would describe elements after the mesh was partitioned, which no process would compute. The
  facet and rigid body generators, which describe none, may run late;
* an **expression field output over an element set** (``>>fromExpression, elSet=``): the expression
  reads the element objects of the whole set itself, which cannot be gathered.

Moving these readers to gathers of their own removed them from the rule one by one: adaptive
refinement on replicated mesh data with owner-local elements, then contact and ties, which read
only replicated, surface-sized data. ``testfiles/mpi/edelweiss-only/NEDWholeModel`` keeps the
fallback itself under test.

What a distributed process creates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~edelweissfe.domaindecomposition.distributedelements.DistributedElements` partitions the mesh
with the same function and the same weights the subdomain uses
(:func:`~edelweissfe.domaindecomposition.partitioning.partitionElementsOfMesh`), right after the
mesh is described and before the elements are made
(:func:`~edelweissfe.helpers.inputfilehelpers.fillFEModelFromInputFile`). A process then creates

* the elements it **owns** -- those the partition assigns to it; and
* every **auxiliary element** -- the contact facets and the point masses of rigid bodies
  (see `Contact, ties and rigid bodies`_).

A load acting on an element -- a distributed load on its face, a body force -- is evaluated by the
process computing the element and exchanged like its forces (see `An increment`_), so no process
needs another one's element for it. An element described after the partition, which no process
would compute, is refused.

The subdomain of the solver adopts this partition instead of computing its own.

.. _domaindecomposition_mesh_to_elements:

Mesh to elements
~~~~~~~~~~~~~~~~

Every model is built in two stages (see :doc:`mesh`): the input file and the mesh generators
*describe* the mesh as data -- ``model.mesh``, a :class:`~edelweissfe.models.mesh.Mesh` with every
element's number, type, provider and node labels, the element sets as lists of element numbers and
the surfaces by their element sets -- and the element objects are then *made* from it by
:meth:`~edelweissfe.models.femodel.FEModel.createElementsOfMesh`, for every element the model's
:class:`~edelweissfe.models.elementdistribution.ElementDistribution` creates here (the model owns its
distribution, ``model.elementDistribution``). Four properties
make a distributed model possible:

* **The layout of the degrees of freedom follows from the mesh.** Which fields an element has at
  which node depends on its type only, so the fields at the nodes are activated from the mesh, not
  from element objects (:meth:`~edelweissfe.models.mesh.Mesh.typeOf`). The global numbering of the
  degrees of freedom is therefore the same in every process *by construction*, whichever elements it
  created. Vectors stay global-length: a process holds the solution at every node, and receives it
  from the owners of the degrees of freedom whenever the whole model is read.
* **An element set holds its local elements, and fails safe.** Each element set of the model, an
  :class:`~edelweissfe.sets.elementset.ElementSetOfMesh`, holds the elements of its set in the mesh
  that are *local* to this process (created here), and is *complete* if that is all of them
  (``isComplete``); a serial run is always complete. So do the faces of surfaces, and with them
  sections. Reading a set that is not complete as a whole -- iterating it, indexing it, ``len()`` --
  raises a :class:`~edelweissfe.utils.exceptions.TopologyError`; a reader of the local part asks for
  it explicitly (:meth:`~edelweissfe.sets.elementset.ElementSet.localElements`), e.g. the section
  assignment, the restriction of the loads to a subdomain, the initial conditions and the result
  collector of a per-element field output.
  What the mesh describes is known for the whole set in every process: its element numbers
  (:meth:`~edelweissfe.sets.elementset.ElementSetOfMesh.elementNumbersOfWholeSet`) and its nodes
  (:meth:`~edelweissfe.sets.elementset.ElementSet.extractNodeSet`), read from the mesh.
* **Whole-model readers gather, or refuse.** A reader that needs every element of a set or of the
  model either goes through the gather path of the element distribution (see `Where the whole model is
  read`_), or reads it whole and so fails loudly on a distributed model, instead of giving a silently
  partial result -- by iterating the set, or up front, naming itself, through
  :meth:`~edelweissfe.sets.elementset.ElementSet.requireComplete` or, for the whole model,
  :meth:`~edelweissfe.models.femodel.FEModel.requireCompleteMesh`.
* **An element can be created at any time, anywhere.** An element is completely described by the
  mesh, its section (assigned through its sets) and its state vector, so
  :meth:`~edelweissfe.models.femodel.FEModel.createElementOfMesh`,
  :meth:`~edelweissfe.sections.base.sectionbase.Section.assignSectionToElement` and
  ``setStateVars`` recreate it in another process -- which is what moving an element between
  processes needs (see `Load balancing`_).

Contact facets and the point masses of rigid bodies are auxiliary elements: the surface or rigid body
they belong to makes them itself, in every process; they
are added to the mesh as well (:meth:`~edelweissfe.models.mesh.Mesh.addAuxiliaryElement`), with
their host element -- the element of the mesh they lie on --, and the facets are cut from the surface as described in the
mesh, not from element objects. A refinement adds its children to the mesh and creates them from it.

Subdomains, interface, ownership
--------------------------------

The elements are partitioned by `METIS <https://github.com/KarypisLab/METIS>`_
(:mod:`~edelweissfe.domaindecomposition.metis`, :func:`METIS_PartMeshDual`): the dual graph of the
mesh -- elements adjacent when they share a face, or an edge in 2D -- is split into parts of balanced
weight, minimising the total communication volume. The partition is computed on the mesh -- element
numbers, connectivity, and the size of each element type -- not on element objects, on rank 0, and
broadcast. The auxiliary elements are not given to METIS: each is computed by the process
of its host element -- a facet by that of the solid element whose face it tiles, so that its degrees
of freedom are already in that subdomain -- or, if it has none, as a point mass, by the process given
part 0 of the partition (rank 0, unless a repartition renumbered the parts)
(:func:`~edelweissfe.domaindecomposition.partitioning.processOfAuxiliaryElement`). The constraints
are dealt out by name, one process each.

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
:class:`~edelweissfe.solvers.base.modelpartition.ModelPartition` -- three members: the elements
computed here, the constraints evaluated here, and the degrees of freedom integrated here. For
``NED`` that is the whole model, and the degrees of freedom are ``slice(None)``, so that ``V[dofs]``
is the whole vector; for ``NEDMPI`` it is the subdomain of the process, as a
:class:`~edelweissfe.domaindecomposition.subdomain.Subdomain` defines it, and the degrees of freedom
are a sorted index array. Which of them a process owns -- counts once for the model, e.g. in the
external work -- is the subdomain's (``Subdomain.ownedDofMask``).

Everything a process must exchange with the others, ``NEDMPI`` adds in overrides of a few methods of
``NED``, each calling the ``NED`` method or the shared function it extends:

==========================================  =========================================================
``NEDMPI`` method                            adds
==========================================  =========================================================
``partitionModel``                          the subdomain, instead of the whole model
``assembleInternalForces``                  the interface exchange of the element forces
``assembleLumpedDiagonal``                  the same for the lumped inertia and damping, once per mesh
``assembleLoads``                           the loads of the own elements, completed at the interface
``assembleConstraintForces``                evaluates the own constraints, sends forces to neighbours
``getCriticalTimeStepForExplicitDynamics``  the minimum over the subdomains
``energyBalanceTerms``                      the sums over the subdomains
``halfMassTimesSquaredRate``                counts a degree of freedom where it is owned
``addExternalWork``                         keeps the owned work until it is read (below)
``publishNodeFields``                       publishes the degrees of freedom integrated here
``acceptIncrement``                         the synchronization and rebalancing of an output increment
``advanceModelToTime``                      accepts the states of the own elements and constraints only
``updateConstraintConnectivity``            the positions a contact search reads, from their owners
``updateConnectivityOf``                    runs a search on the constraint's process only
``updateTopology``                          the agreement on a topology update
``writeIncrementOutput``                    reads locally, agrees, gathers, then writes (below)
``applyStepActionsAtStepEnd``               the synchronization at the end of a step
``buildEquationSystem``                     completes the vectors at newly integrated degrees of freedom
``releaseEquationSystem``                   (its own) releases the system before elements move
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
and so is everything computed from it. The constraint forces, each computed by one process, are sent
to the processes integrating their degrees of freedom and added there in model order (see `Contact,
ties and rigid bodies`_). A run on any number of processes is bit-identical to ``NED`` on the same
input.

**What is bit-identical, exactly.** The solution, the velocity and the net nodal force at every
degree of freedom; the state of every element and every constraint; the external work; and
therefore every dataset of a restart checkpoint and every field output computed from them. Not
bit-identical: the kinetic and the internal energy printed in the energy table, which each process
sums over its own subdomain and the processes then add in rank order; they enter nothing but the
table. Timings, and the load balancing that depends on them, change the speed of a run, never its
result.

**What the order rests on.** Within one process the contributions at each degree of freedom are
summed by :func:`numpy.bincount` (:meth:`~edelweissfe.solvers.base.parallelelementcomputation.ElementPlan.assembleInto`,
:class:`~edelweissfe.domaindecomposition.subdomaininterface.InterfaceForceAssembly`), which adds the
weights into each bin one after another, in the order given -- a left fold, the sum a loop over the
elements forms. That is how NumPy implements it, not a documented guarantee, so it is tested rather
than assumed: ``tests/test_domaindecomposition.py`` sums values of very different magnitudes and
signed zeros, whose sum differs in any other order, through the element loop and -- over three
processes -- through the interface exchange, and compares the bits with the left fold. It was last
verified with NumPy 2.5.2; run these tests after upgrading NumPy.

**Loads.** A distributed load or a body force acting on an element is a contribution of that element,
and is assembled like its forces
(:meth:`~edelweissfe.domaindecomposition.subdomain.Subdomain.loadsOnSubdomain`): the process
computing the element evaluates its load, with its own solution -- current at every degree of
freedom of the element, since the process integrates all of them -- and sends the contributions at
the degrees of freedom a neighbour integrates too, tagged with their place in the order a single
process adds the loads in: the loads in deck order, the faces of each surface, the elements of each
face or set as the mesh describes them, distributed loads before body forces
(:class:`~edelweissfe.domaindecomposition.subdomaininterface.InterfaceLoadAssembly`). Serially, the
loads are not part of the element sum: ``NED`` sums the element forces, negates them, adds the
concentrated loads, and then adds every load contribution onto that, one after another. Every
process does the same at each of its degrees of freedom -- the net force of the elements and the
concentrated loads, the same bits everywhere, then the first load contribution, the second, and so
on -- so the result is the same bits as serially. A concentrated load acts on degrees of freedom,
not on elements, and is added by every process integrating them. A configuration-dependent load --
a follower pressure on a finite-strain element, say -- therefore reads the current solution, not the
one of the last synchronization, which a load evaluated on another process' element would.

The volume exchanged per increment is that of the interface and of the constraint forces at it. The lumped mass
and damping are assembled once per mesh by each process from its own elements, completed at the
interface like the forces and then shared from the owners, so that every process holds them, the
same bits as without decomposition, at every degree of freedom of the model. The critical time step
is the minimum over the subdomains.

Where the whole model is read
-----------------------------

A process keeps current only what it computes: the solution, the velocity and the net force at the
degrees of freedom it integrates, and the states of its own elements and constraints. Field outputs,
output managers, restart checkpoints, the marker of an adaptive refinement and the refinement itself,
and a contact search read more than one subdomain. Each of them receives what it reads from the
processes computing it -- every entry of a vector from its owner -- and only the processes that read
it receive it:

=================================  ==================================================================
Synchronization point              What is gathered, to whom
=================================  ==================================================================
output increment                   to **rank 0**, which writes the output and the checkpoints: the
                                   solution, the velocity and the net force at every degree of freedom
                                   (published into its node fields), the state of every stateful
                                   constraint, the results of every element field output, and -- if a
                                   checkpoint is written -- the element states; and to **every
                                   process** the solution at the reference nodes of the rigid bodies,
                                   whose surfaces every process moves (they are nodes of the model)
output increment the topology      to **every process**: all of the above but the element states, and
check follows (every               of the element field outputs those the markers read; every process
``topology-check-frequency``-th)   refines the same mesh, interpolates its node fields onto the refined
                                   mesh, and builds its equation system from them
end of a step                      to **every process**, as before a topology check: the next step
                                   starts with a topology update and an equation system built from the
                                   node fields; a field output whose last result went to rank 0 only is
                                   read again, so that every process holds it (a step-start marker or
                                   ``setField`` may read it)
periodic contact search            to the **process evaluating the constraint**: the positions of the
(``contact-update-frequency``)     nodes the search reads that it does not integrate itself, point to
                                   point (see `Contact, ties and rigid bodies`_)
the subdomain changes: a contact   to **each process**, at the degrees of freedom it integrates from
search moved a constraint, or a    now on: the solution, the velocity and the net force, from their
repartition moved elements         previous owners, point to point
=================================  ==================================================================

Which processes read what is decided from the model, not by a blanket rule. A topology check follows
an output increment when the solver says so (:meth:`~edelweissfe.solvers.nonlinearexplicitdynamic.NED.topologyCheckDueAfter`,
:meth:`~edelweissfe.solvers.nonlinearexplicitdynamicmpi.NEDMPI.outputSynchronizationReachesEveryProcess`); a
marker names the field outputs it reads
(:meth:`~edelweissfe.adaptivity.marking.MarkerBase.fieldOutputsRead`, gathered for its model modifier
by :meth:`~edelweissfe.modelmodifiers.base.modelmodifierbase.ModelModifierBase.fieldOutputsRead`); a
constraint names the nodes its search reads
(:meth:`~edelweissfe.constraints.base.constraintbase.ConstraintBase.nodesReadByConnectivityUpdate`).
Each says None where it cannot name them -- a marker or a modifier of another package, a constraint
not declaring its nodes -- and is then sent all of it: every field output, or the whole solution to
every process before a search.

**What a process holds, and what refuses to be read.** Between two synchronizations that reached it,
a process holds the vectors and the node fields current at the degrees of freedom it integrates only
-- elsewhere they hold what the last synchronization left there --, its copies of the constraints
owned elsewhere in the state of their last synchronization, and the field outputs gathered to rank 0
not at all. None of this is read as if it were whole:

* the readers of the whole solution -- a topology update, an equation system built from the node
  fields -- first require it to be here
  (:meth:`~edelweissfe.solvers.nonlinearexplicitdynamicmpi.NEDMPI.requireWholeSolutionHere`), and
  raise otherwise;
* a field output that was gathered to rank 0 only records the time of the increment (so that every
  process decides alike whether the end of a step stores another result), and reading its last
  result or its history in another process raises, until a result is stored there again
  (:meth:`~edelweissfe.utils.fieldoutput.FieldOutputController.gatherResultsOfWholeSet`);
* before a topology update, the constraint copies are compared with their owners
  (:meth:`~edelweissfe.domaindecomposition.subdomain.Subdomain.requireConstraintCopiesCurrent`);
* after every connectivity update, a constraint that names the nodes its search reads must couple no
  other node -- else the search read a position it may not have received -- and the run stops
  naming the constraint.

``debug-poison-stale-solution=True`` (a check, not for production runs) makes the rest loud as well:
after every increment it sets the solution, the velocity, the net force and the node fields to NaN
outside the degrees of freedom a process integrates, wherever the process does not hold the whole
solution. Nothing may read them there, so the result must stay the same bits; a reader of data it did
not receive -- a search reading a node it does not name, say -- turns it into NaN. The MPI workflow and
the verification harness run ``testfiles/mpi`` with it.

``tests/test_domaindecomposition.py`` checks with it that a poisoned run gives the same bits, and
that the poison would notice: without the positions received for the contact search, it does not.

What it saves, on the c1_150 edge-breakout model (about 1 million degrees of freedom; output and a
contact search every 50 increments, a topology check every 250; summed over all processes):

==============================  ========================  ========================
                                8 processes x 4 threads   32 processes x 1 thread
==============================  ========================  ========================
output increment, before        313 MB, 0.30 s            1386 MB, 0.50 s
output increment, now           39 MB, 0.20 s             43 MB, 0.22 s
contact search, before          191 MB, 0.26 s            847 MB, 0.32 s
contact search, now             0.13 MB, 0.18 s           0.18 MB, 0.17 s
==============================  ========================  ========================

An output increment a topology check follows costs what it did. What is left of a contact search
is the search itself; an increment without output or search, and the peak memory of a process, are
unchanged.

What happens to the element states depends on how the model is held.

**The whole model on every process.** Every process receives the current state of every element
from the process that computed it on every output increment, since a repartition may give any
element to any process (:mod:`~edelweissfe.domaindecomposition.statesynchronization`). The states
travel through the same interface restart checkpoints use: whatever a checkpoint must carry to resume
a run is what another process must receive to continue it.

**Distributed.** No element state is synchronized -- a process holds no element it does not compute
-- and each whole-model reader goes through
the *gather path* of the element distribution
(:class:`~edelweissfe.models.elementdistribution.ElementDistribution`, whose serial base answers with
what is here):

======================================  =================================================================
Reader                                  How it reads a distributed model
======================================  =================================================================
element field output over a set         ``resultsOfWholeSet``: each process collects the results of the
(``>>perElement``)                      elements of the set it computes, and they are gathered, by mesh
                                        element number, in set order, to rank 0 -- and to every process
                                        if a marker reads it before a topology check --, so a field
                                        output is the same wherever it is read, as without decomposition
node field output over an element set   the nodes of the whole set, from the mesh
(``>>perNode, elSet=``)                 (``extractNodeSet`` of the set, also of a partial set, and
                                        again after a refinement); the node fields are global-length
Ensight                                 rank 0 draws the geometry of a partial set from the mesh
                                        (``visualizedElementsOf``) and writes the gathered results
monitor, conditional stop               read field outputs
restart checkpoint                      on an increment rank 0 writes one
                                        (``writesCheckpointAtNextIncrement``, broadcast), every process
                                        reads the states of the elements it computes and they are
                                        gathered to rank 0 (``Subdomain.gatherElementStatesToRoot``);
                                        rank 0 writes them in the format of a serial checkpoint
                                        (``FEModel.elementStatesFromElsewhere``) and releases them,
                                        however the output ended
======================================  =================================================================

Still guarded, and refused loudly on a partial model: an expression field output over an element
set (``>>fromExpression, elSet=``, which the fallback rule already sends to the whole model), the
mesh plot and ``meshDataToFile`` (which fail at their setup), ``surfaceSnap`` and the P1 topology
classification (which only exist with a model modifier the fallback rule sends to the whole model).

The gathers of the field outputs and of the element states happen between the two output steps every
process agrees on -- reading its own part, and writing -- in the same order in every process, so no
process can wait for one that already left (see Failures below).

The contact search itself, at a contact update and at a topology check alike, runs on the process
that evaluates the constraint only: nothing but that evaluation reads its outcome. The constraint
forces are sent with the degrees of freedom of the owner's search. Whether a search changed a
footprint is decided by the owners and communicated to all processes.

Restoring a constraint's state restores it completely -- a contact constraint adopts the owner's
search, and with it the nodes it couples -- so after a synchronization that reached it a process'
copy of a constraint is its owner's. Between two such synchronizations a copy keeps that footprint, with the mesh
refreshes since applied to it as to the owner's. Its degree-of-freedom indices are never read, since
a constraint is evaluated, and its forces exchanged, with its owner's indices only. The footprint
itself is read in one place: a topology change activates fields on the nodes of every constraint
copy. That is harmless because a topology update always follows a synchronization of every process
-- a topology check is due only on an output increment, whose synchronization then reaches every
process, the start of a step follows the end of the last one or a checkpoint -- and it is asserted rather than assumed: before every topology update the nodes and
fields every constraint couples are compared across processes
(:meth:`~edelweissfe.domaindecomposition.subdomain.Subdomain.requireConstraintCopiesCurrent`).

Contact, ties and rigid bodies
------------------------------

A constraint is surface-sized: it couples the nodes of one or two surfaces, or of a surface and a
rigid body. It is evaluated whole, by one process (dealt out by name), and runs distributed because
nothing it reads is an element object of the solid mesh:

======================================  =================================================================
What a constraint reads                 Where it comes from in a distributed model
======================================  =================================================================
contact facets of its surfaces          made by every process, from the surface as described in the mesh
                                        (``surfaceElementGenerator``); the whole facet set is in every
                                        process, and a constraint reads it through
                                        :meth:`~edelweissfe.models.femodel.FEModel.wholeElementSet`, which
                                        refuses a partial set; a facet's parent face (for the
                                        surface-to-surface quadrature) is stamped on the facet itself
node sets                               every process holds every node set
nodes, their coordinates and fields     every process holds every node; the positions of the nodes a
                                        search reads are current on the constraint's process before the
                                        search (below)
rigid bodies                            made by every process (``discreteRigidBodyGenerator``), with the
                                        point mass of the reference node; its surface follows the
                                        reference node, moved from the complete solution in every process
the degrees of freedom it couples       global numbers, the same in every process
======================================  =================================================================

**What a search receives.** A contact search runs on the process evaluating the constraint, and
reads the current positions of every node it may couple -- the slave nodes (or the nodes of the
faces carrying the contact points), the nodes of every master facet, the reference node of a rigid
body --, of which that process integrates only those the constraint couples now. The constraint
names them (:meth:`~edelweissfe.constraints.base.constraintbase.ConstraintBase.nodesReadByConnectivityUpdate`);
before every periodic search the process receives their displacement from the owners of the
degrees of freedom, point to point, and publishes it into its node fields
(:meth:`~edelweissfe.solvers.nonlinearexplicitdynamicmpi.NEDMPI.updateConstraintConnectivity`).
Which entries go where is agreed once per subdomain, at its first search; a search then moves the
values alone. If the search changes what the constraint couples, the process integrates further
degrees of freedom, and receives the solution, the velocity and the net force there from their
previous owners when the equation system is rebuilt. A constraint naming no nodes has the whole
solution sent to every process before a search instead.

A tie also moves (snaps) slave nodes when it is constructed: every process constructs every
constraint and so moves its nodes identically, before any element is initialized; an element created
later -- after a migration, or as the child of a refinement -- is made from the moved nodes.

Where a tie couples nodes of several subdomains, the multi-point-constraint closure (see
`Subdomains, interface, ownership`_) makes the process integrating any of its degrees of freedom
integrate the whole group; the forces at those degrees of freedom are completed by degree of freedom,
whichever element objects a process holds. A refinement of a contact or tie surface retiles it with
new facets, made by every process and computed by the process of their host element, a child, and
the constraint projects onto them afresh. A restart checkpoint carries the constraint states as
before: they are synchronized from their owners before every output.

**How the constraint forces travel.** The owner of a constraint integrates every degree of freedom
it acts on, so another process integrating one of them shares it with the owner. The owner sends each
such neighbour, point to point, the forces of its constraints at the degrees of freedom the two share,
and nothing else; which entries go where, and their place in the order of addition -- the position of
the constraint in the model and the index of the entry within it -- is exchanged once per rebuild of
the equation system, an increment sends the forces alone. Every process then adds the forces it holds
constraint by constraint, in model order, each as ``NED`` adds it
(:func:`~edelweissfe.numerics.assembly.addNodalForces`): at a degree of freedom it integrates, those
are the forces of every constraint acting there, added in the same order onto the same bits, so the
sum is the same bits as serially
(:class:`~edelweissfe.domaindecomposition.subdomaininterface.ConstraintForceExchange`).

The net force vector of a process is therefore defined at the degrees of freedom it integrates only.
Elsewhere it holds neither the element forces of the other subdomains nor the constraint forces, and
nothing reads it there: the vectors are gathered from the owners before anything reads the whole
model (`Where the whole model is read`_). ``tests/test_domaindecomposition.py`` asserts both -- the
exchange writes nothing outside the subdomain, and a run with the net force set to NaN there gives the
same bits.

What it costs: the constraint owners evaluate their constraints while the others wait. On the c1_150
edge-breakout model (five penalty contacts, five ties, 8 processes of 4 threads) the largest contact
(the support under the slab, surface to rigid body) costs its process about 7 ms of a 90 ms increment,
the others 0.5--1 ms. Exchanging and adding the constraint forces costs rank 0 about 1.2 ms at 8x4
and 0.5 ms at 32 processes of one thread, and moves 2.5 MB (8x4) or 3.1 MB (32x1) per increment over
all processes. The former exchange, which shared every constraint's forces with every process (an
``Allgatherv``), moved 19.5 MB and 86.5 MB: at 32 processes one process' large contact was copied 31
times, and an increment took about 127 ms instead of 95 ms (89 instead of 88 ms at 8x4).
Partitioning a contact by its slave points is not done.

Adaptive refinement
-------------------

Adaptive refinement (``hAdaptivity``) runs on a distributed model. What it decides is split by
what it reads:

**Replicated, the same in every process: the topology.** The octree mirror, the marking, the 2:1
balance, the hanging nodes and the conformity check, and the numbers of the new nodes and elements,
are derived from the mesh, the nodes and the node fields -- which every process holds whole -- and
never from an element object (:mod:`~edelweissfe.modelmodifiers.adaptivity.hadaptivity`). The
markers (:mod:`~edelweissfe.adaptivity.marking`) mark element *numbers*:

* a field-output marker thresholds the result of an element field output, row by row in the set
  order of the mesh; it names that field output
  (:meth:`~edelweissfe.adaptivity.marking.MarkerBase.fieldOutputsRead`), which is therefore gathered
  to every process on the output increment a topology check follows (``resultsOfWholeSet``);
* the element-set, node-set and surface markers read the sets and surfaces of the mesh;
* the recovery-error marker reads the node coordinates and a node field at the nodes of the
  refineable elements of the mesh -- global data only, current in every process after the output
  synchronization a topology check follows -- so it, too, marks the same elements everywhere and
  needs no gather of its own.

Every process therefore refines the same mesh, and numbers it the same way. That is verified, not
assumed: after every build of the equation system, a fingerprint
(:func:`~edelweissfe.domaindecomposition.subdomain.layoutFingerprint`) of the mesh (element numbers and
connectivity), its element sets and surfaces with their elements, all in order -- the order the loads
are added in follows from them --, the degree of freedom of every node of every field, the node
coordinates, the size of the system and every slave, master and weight of the multi-point constraints
is compared across all processes, and the run stops if any two differ.

Every process must therefore build its model in the same order. Python orders a ``dict`` by insertion,
the same in every process, but a ``set`` of strings by their hashes, which differ between processes
unless ``PYTHONHASHSEED`` is fixed. Code deciding the model's order by iterating such a set would stop
the run at the fingerprint check; until it is fixed, ``PYTHONHASHSEED=0`` (passed with ``-x
PYTHONHASHSEED`` to every process) makes the processes agree.

**Owner-local: the elements and their states.** The child of a refined element is computed by the
process that computed its parent
(:meth:`~edelweissfe.domaindecomposition.distributedelements.DistributedElements.placeChildElement`).
Only that process creates it from the mesh, assigns it its parent's section and element properties,
and transfers its parent's state to it; the parent is then dropped. Every process updates the mesh,
the nodes, the element and node sets, the surfaces and the node fields identically. After the
modifier, the topology pipeline lets the distribution create and drop the elements of the changed
mesh (:meth:`~edelweissfe.models.elementdistribution.ElementDistribution.updateLocalElements`): a
process drops the elements no longer in the mesh. The equation
system is then built again, as after any topology change, adopting this partition; the next
rebalancing check may move the children like any other element (`Load balancing`_). The hanging-node
constraints of the refinement are multi-point constraints, closed over in the subdomain by degree of
freedom (a process integrating any degree of freedom of a hanging node integrates its masters too),
whichever process computes the elements around them.

**Restart.** A resumed run replays the recorded refinements on the partition its processes start
with: the children are created where the replayed partition computes their parents -- not where the
written run had moved them -- and every process restores, by number, the states of the elements it
created from the one checkpoint rank 0 wrote. The result is that of an uninterrupted run.

A whole-model run (another reason in `The fallback rule`_) refines identically, with every process
creating every child; its refined model is then partitioned afresh.

Load balancing
--------------

The first partition weighs an element by its number of degrees of freedom. A softening material
costs more where it softens -- a return mapping needs more iterations -- so a partition balanced at
the start drifts out of balance as damage localizes. Every element kernel is therefore timed, and on
an output increment, as a step of the increment loop of its own right after the synchronization of
the output, the model is repartitioned with the measured
costs as weights whenever the slowest process has fallen more than ``load-balance-tolerance``
(default 0.1) behind the mean -- and if that is worth it: the time it is expected to save until
the next check (the imbalance beyond the tolerance, times the mean time of a process per increment,
times ``output-frequency``) must exceed what the last repartition cost, measured from deciding it to
the rebuilt equation system, in the slowest process. The first repartition is always made; a model
that cannot be balanced better -- fewer elements than processes, say -- is therefore not
repartitioned again and again. The journal says at level 2 why it did not repartition. The partition
then depends on measured timings; the result does not, since it does not depend on the partition at
all.

The new parts are numbered so that as many elements as possible keep their process
(:func:`~edelweissfe.domaindecomposition.partitioning.keepElementsWhereTheyWere`): METIS numbers
parts arbitrarily, and a renumbered but otherwise similar partition would move almost every element.

Where every process holds the whole model, a repartition changes nothing but which elements each
process computes. A **distributed** model *migrates* the elements whose process changes
(:meth:`~edelweissfe.domaindecomposition.distributedelements.DistributedElements.moveElementsTo`).
An element is completely described by the mesh (its number, type and nodes, which every process
holds), the replicated definitions (its sections, material -- ``materialParameterFromField``
included, evaluated at the element's centre -- and element properties) and its state. So, right after
the increment was accepted, in every process:

1. the process that computed an element until now sends its state (``getStateVars``, the converged
   state) to the element's new process, all of them in one exchange;
2. each process drops the element objects it no longer needs, and creates from the mesh those it
   now needs (:meth:`~edelweissfe.models.femodel.FEModel.createElementOfMesh`) -- its new elements
   -- keeping the elements in mesh order;
3. the element sets and surfaces are resolved to the elements now held, in place; the new elements
   receive their sections and element properties as at setup
   (:meth:`~edelweissfe.models.femodel.FEModel.assignSectionsAndPropertiesToElements`), and every
   element arriving from another process the state that process sent (``setStateVars``).

Everything indexed by the elements a process holds is then built again, as after a change of the
topology: the solver rebuilds the equation system for the elements now held, carrying the solution,
the velocity and the force over -- each process completes them at the degrees of freedom it
integrates from now on, from their previous owners --, and with it the degree-of-freedom indices of the elements, the subdomain and its
interface, the loads of the elements computed here, the lumped inertia and damping (assembled from the
elements now computed here, completed at the interface in model order, so the same bits as before)
and the increment plan with its element timing. The field outputs set up their views of the
element results again for the elements now owned here. All of these are released *before* the
elements move -- the solver keeps only plain copies of the three vectors it carries over, the field
outputs keep element numbers, not elements -- so that a dropped element is freed before the new ones
are created, and a migration does not raise the memory of a process beyond what its elements need:
on a 192 000-element GC3D8 block at 8 processes, two migrations raised the peak resident memory per
process by 80--150 MB (the transient of the exchange and the rebuild), and the memory at the end of
the run equals that of a run that never migrated. A
migrated element is bit for bit the element its previous
process held -- exactly as a resumed restart's element is -- and every sum is still formed in model
order, so a run that migrates elements is bit-identical to a serial run too. The equation system is
built again for the elements now held, which assembles the lumped mass and damping again, where a
serial run keeps them between rebuilds of its equation system: the two agree because an element's lumped inertia and damping do
not depend on its state -- checked at every migration, bit for bit, not assumed; an element violating
it stops the run with a message. The journal reports every migration (``Element migration: ...
element(s) changed process``).

What balancing buys is the waiting it removes, and no more. On that block the costliest process was
only 3--5 % above the mean; a light process (rank 0) waited about 9 % of each increment for it, and
after the migrations a quarter of that, computing more elements in the time it had waited before. The
time per increment, set by the costliest process, could therefore not improve noticeably, and did
not. Balancing pays where damage localizes and the imbalance grows large.

The timed costs make a partition, and so a migration, depend on the machine and its load.
``load-balance-costs=elementNumber`` weighs every element by its number instead: a deterministic,
deliberately uneven cost, with which a test knows that elements move, and when. It balances element
numbers, not work, so a run with it is no measure of performance.
``testfiles/mpi/marmot/NEDRebalanceDistributed`` is such a test: at its output increment 20, after
the material has yielded, elements move between the processes; its result and the checkpoint written
after the migration equal a serial run's, and ``check_element_distribution.py`` checks that elements
moved and that no process still holds an element it dropped.

Output and restart
------------------

Only rank 0 creates output managers. A restart checkpoint is the last output of an output
increment, written after the synchronization of that output, so rank 0's copy of the model holds
every element and every constraint as the process computing it left it. A distributed model
gathers the element states to rank 0 for it instead (see `Where the whole model is read`_); the
file is the same, bit for bit, as a serial run's. That includes the solver's external work: the
work at the prescribed degrees of freedom of each increment is summed exactly (:func:`math.fsum`,
which does not depend on the order of the terms nor on how they are split among processes). Each
process keeps the products at the degrees of freedom it owns, increment by increment, and they are
gathered only where the external work is read -- the energy balance, an output increment, the end
of a step -- and added increment by increment, so every process accumulates that of the whole model,
as a serial run does, without an exchange every increment
(:meth:`~edelweissfe.solvers.nonlinearexplicitdynamicmpi.NEDMPI.gatherExternalWork`). It is an
ordinary checkpoint of the whole model, so a run can be resumed by
``NEDMPI`` on any number of processes, or by ``NED``. The topology check due after that increment
runs at the start of the next one, after the checkpoint, so a resumed run performs it exactly as the
uninterrupted one does (see :doc:`restart`).

On a resume every process restores the whole model from the checkpoint -- every constraint adopting
the checkpointed state of its owner -- and so starts from the same model, and with the external work
of the whole model. A distributed process restores the
elements it created and skips the others
(:meth:`~edelweissfe.models.femodel.FEModel.readRestart`); a checkpoint written by a serial run, by
the whole model on every process, or by a distributed run can be resumed in either way.

What a resumed run does not restore is how the processes shared the elements: it starts from the
partition of the mesh it builds (with a replayed refinement's children where their parents are),
and the count of increments since the last topology change, which weighs a migration after the
next one, restarts at the start of the step. Recording them would make a checkpoint of ``NEDMPI``
differ from one of ``NED``, which a resume in either way relies on not doing. So a resumed run may
place elements differently from the uninterrupted one -- which changes where they are computed,
never the result.

Failures
--------

A failure while evaluating the elements or the constraints -- a material that cannot integrate at
the stable step, above all --, in a constraint's connectivity search, in a topology update (the
marker, the refinement, the mesh refresh), or in finalizing the output (a conditional stop, too) is
raised on all ranks together
(:meth:`~edelweissfe.domaindecomposition.communicator.Communicator.allRanksFailTogether`), and every process
ends the step the same way; a cutback requested anywhere is raised everywhere, with the smallest
size requested. A failure anywhere else aborts all processes, and so does an interrupt (``Ctrl+C``)
of any one of them: the others would otherwise wait forever for the one that stopped
(:mod:`~edelweissfe.domaindecomposition.mpienvironment`). The driver finishes a job normally after a
failed step only if every process raised the failure together
(:class:`~edelweissfe.domaindecomposition.mpienvironment.StepFailedOnAllRanks`): a failure of the
agreement, and the step failures of an increment, which are decided from values every process holds
alike -- a refused cutback, a diverged energy balance
(:meth:`~edelweissfe.solvers.nonlinearexplicitdynamicmpi.NEDMPI.attemptIncrement`). Any other failure
may have happened in one process alone, while the others wait for it in their next exchange; that
process stops all of them as soon as the step ends
(:meth:`~edelweissfe.solvers.nonlinearexplicitdynamicmpi.NEDMPI.endStep`), before the end of the step
would gather results with processes that never arrive.

Nothing inside such an agreement may communicate: a process that failed before a collective operation
would skip it, and leave the others waiting in it. That is enforced: the communicator
(:class:`~edelweissfe.domaindecomposition.communicator.Communicator`) raises at any communication
inside the agreement, which then fails on all ranks like any other failure. What reads more than one
process is split into a part each process does alone and the communication after the agreement. The
output of an increment, for example, is written in three parts
(:meth:`~edelweissfe.solvers.nonlinearexplicitdynamicmpi.NEDMPI.writeIncrementOutput`): each process
reads the results of its own elements for the field outputs, agreeing on failures; the results of
every element set -- and the element states of a checkpoint -- are gathered; then the field outputs
store them and rank 0 writes, agreeing on failures (a conditional stop, too) again.

A model with an element that exposes no state (``getStateVars``) is refused on more than one
process: its state could not be sent to the process writing the output and the checkpoints, nor to
the one computing it after a repartition.

Which way a job held its model does not show in its result. ``testfiles/mpi/check_element_distribution.py``
runs every deck of ``testfiles/mpi`` under the launcher and checks, from the elements each process
created and computed, that it ran distributed or with the whole model as expected -- and, for a deck
expected to migrate, that elements moved and that no dropped element is alive in any process; the MPI
workflow runs it over 2 and 3 processes.

Verifying bit-identity yourself
-------------------------------

The decks in ``testfiles/mpi`` run the same models as their serial counterparts, with
``solver=NEDMPI``. Run each set once serially with ``NED`` and once on several processes, let the
test runner write the final solution of every deck (``--create`` writes ``U.ref`` with 18
significant digits, which reproduces a double exactly, and only rank 0 writes), and compare the
files byte for byte:

.. code-block:: console

    cp -r testfiles/mpi serial && cp -r testfiles/mpi decomposed
    sed -i -e 's/solver=NEDMPI/solver=NED/' -e '/^load-balance-/d' serial/*/*/test.inp
    export PYTHON_GIL=0 OMP_NUM_THREADS=1
    run_tests_edelweissfe serial/edelweiss-only --create
    mpirun -n 3 --bind-to none -x PYTHON_GIL -x OMP_NUM_THREADS \
        run_tests_edelweissfe decomposed/edelweiss-only --create
    for reference in serial/*/*/U.ref; do
        cmp "$reference" "decomposed/${reference#serial/}" || echo "DIFFERS: $reference"
    done

The ``sed`` line turns every deck into a serial one: ``solver=NED``, without the options only
``NEDMPI`` has (``load-balance-tolerance``, ``load-balance-costs``), which ``NED`` would refuse.
Nothing printed by the loop means every deck is bit-identical; ``marmot`` in place of
``edelweiss-only`` runs the decks that need Marmot (some need its private materials). Compare against
a serial run on the same machine rather than against the committed ``U.ref`` files: those were
written on another one, and a serial run differs from them in the last digits wherever the two
machines' math libraries do. ``run_tests_edelweissfe testfiles/mpi/edelweiss-only`` without
``--create`` compares against the committed files within an absolute tolerance of 1e-6.

``U.ref`` holds the solution only. The states of the elements and constraints, the velocity, the net
force and the external work are all in a restart checkpoint, so comparing the checkpoints of a deck
that writes one compares everything a run carries on with -- for example
``testfiles/mpi/marmot/NEDRestartDistributed1Write``, which runs distributed:

.. code-block:: console

    cd serial/marmot/NEDRestartDistributed1Write && edelweissfe test.inp && cd -
    cd decomposed/marmot/NEDRestartDistributed1Write
    mpirun -n 3 --bind-to none -x PYTHON_GIL -x OMP_NUM_THREADS edelweissfe test.inp && cd -
    python - <<'EOF'
    import h5py, numpy as np
    serial = h5py.File("serial/marmot/NEDRestartDistributed1Write/restart_0.h5")
    decomposed = h5py.File("decomposed/marmot/NEDRestartDistributed1Write/restart_0.h5")
    def compare(name, dataset):
        if isinstance(dataset, h5py.Dataset):
            a, b = np.asarray(dataset[()]), np.asarray(decomposed[name][()])
            same = a.shape == b.shape and a.tobytes() == b.tobytes()
            print("" if same else "DIFFERS: " + name, end="")
    serial.visititems(compare)
    EOF

Nothing printed means every dataset -- every element's state variables, every constraint's state,
``U``, ``V`` and ``P`` of every node field, and the solver's ``_externalWork`` -- is the same, byte
for byte, signed zeros included.

Limitations
-----------

* Memory: a distributed process holds its own elements, but every process holds the whole mesh,
  every node with its fields, global-length vectors, and every contact facet and rigid body. Jobs
  under the fallback rule hold the whole model on every process (c1_150 at 8 processes: 7.6 GB per
  process with the whole model, 2.55 GB distributed).
* A migration rebuilds the equation system of every process, which costs about what building it
  at the start does (2.2 s on a 192 000-element block at 8 processes, plus 0.3--0.6 s to move the
  elements); the gain check above weighs that cost.
* METIS repartitions from scratch: even a small imbalance can move a fifth of the elements.
* The topology of an adaptive refinement -- its octree mirror, the 2:1 balance, the hanging nodes --
  is computed by every process on the whole mesh, and the mirror is held by every process: about
  4.7 KB of Python objects per root element (110 MB for 24 000 GC3D20R elements, 334 MB for the
  c1_150 model).
* Before every topology check the whole solution, and the field outputs the markers read, are
  gathered to every process, since every process refines the whole mesh.
* A constraint is evaluated whole, by one process, while the others wait; a single large contact
  constraint is not split (see `Contact, ties and rigid bodies`_). The implicit-only constraint types
  still hold the whole model on every process.
* Only the explicit dynamic solver is decomposed.

Package reference
-----------------

.. automodule:: edelweissfe.domaindecomposition
   :members:

.. automodule:: edelweissfe.domaindecomposition.mpienvironment
   :members:

.. automodule:: edelweissfe.domaindecomposition.metis
   :members:

.. automodule:: edelweissfe.domaindecomposition.communicator
   :members:

.. automodule:: edelweissfe.domaindecomposition.subdomain
   :members:

.. automodule:: edelweissfe.domaindecomposition.loadsonsubdomain
   :members:

.. automodule:: edelweissfe.domaindecomposition.distributedelements
   :members:

.. automodule:: edelweissfe.domaindecomposition.partitioning
   :members:

.. automodule:: edelweissfe.domaindecomposition.subdomaininterface
   :members:

.. automodule:: edelweissfe.domaindecomposition.statesynchronization
   :members:

.. automodule:: edelweissfe.constraints.base.wholemodel
   :members:
