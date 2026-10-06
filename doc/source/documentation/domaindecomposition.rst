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

The model in every process
--------------------------

Every process reads the same input file and builds the complete model: every node, element, set,
constraint and field. What differs between processes is only which part of it each one *computes*.
The elements are ordinary elements -- they know nothing of the decomposition.

This is what makes adaptive refinement tractable (see below), and it is the main limitation: the
memory each process needs does not shrink as processes are added, and a refinement costs every
process what it costs a serial run. Holding only the elements a process computes, with the mesh
itself replicated as lightweight data, is planned.

Subdomains, interface, ownership
--------------------------------

The elements are partitioned by `METIS <https://github.com/KarypisLab/METIS>`_
(:mod:`~edelweissfe.domaindecomposition.metis`, :func:`METIS_PartMeshDual`): the dual graph of the
mesh -- elements adjacent when they share a face, or an edge in 2D -- is split into parts of balanced
weight, minimising the total communication volume. The partition is computed on rank 0 and
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

A process keeps current only what it computes. Field outputs, output managers, the marker of an
adaptive refinement and the refinement itself, and a contact search all read more than one
subdomain, and before each of them every process receives the current solution at every degree of
freedom from its owner, and -- except before a contact search, which reads positions only -- the
current state of every element and stateful constraint from the process that computed it
(:mod:`~edelweissfe.domaindecomposition.statesynchronization`). The states travel through the same
interface restart checkpoints use: whatever a checkpoint must carry to resume a run is what another
process must receive to continue it.

This synchronization happens every ``output-frequency`` increments, at every
``contact-update-frequency`` search, and at the end of a step.

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

Because every process holds the complete, synchronized model at a topology check, every process runs
the same refinement -- the same marker on the same field output, the same state transfer from the
same parent states -- and arrives at the same refined model. That is verified, not assumed: after
every build of the equation system, a fingerprint of the element numbers, the degrees of freedom of
every element (its connectivity in the numbering of the layout), the node order of every field, the
node coordinates and the size of the system is compared across all processes, and the run stops if
any two differ. The refined model is then partitioned afresh. Nothing migrates, because every process already
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

Output and restart
------------------

Only rank 0 creates output managers. A restart checkpoint is the last output of an output
increment, written after the synchronization of that output, so rank 0's copy of the model holds
every element and every constraint as the process computing it left it, and the external work of
the whole model. It is an ordinary checkpoint of the whole model, so a run can be resumed by
``NEDMPI`` on any number of processes, or by ``NED``. The topology check due after that increment
runs at the start of the next one, after the checkpoint, so a resumed run performs it exactly as the
uninterrupted one does (see :doc:`restart`).

On a resume every process restores the whole model from the checkpoint -- every constraint adopting
the checkpointed state of its owner -- and so starts from the same model; rank 0 continues with the
external work of the whole model, the others with none of it.

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

* Memory: every process holds the complete model.
* Adaptive refinement is computed by every process, in full.
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

.. automodule:: edelweissfe.domaindecomposition.partitioning
   :members:

.. automodule:: edelweissfe.domaindecomposition.subdomaininterface
   :members:

.. automodule:: edelweissfe.domaindecomposition.statesynchronization
   :members:
