#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
#
#  _____    _      _              _         _____ _____
# | ____|__| | ___| |_      _____(_)___ ___|  ___| ____|
# |  _| / _` |/ _ \ \ \ /\ / / _ \ / __/ __| |_  |  _|
# | |__| (_| |  __/ |\ V  V /  __/ \__ \__ \  _| | |___
# |_____\__,_|\___|_| \_/\_/ \___|_|___/___/_|   |_____|
#
#
#  Unit of Strength of Materials and Structural Analysis
#  University of Innsbruck,
#  2017 - today
#
#  Matthias Neuner matthias.neuner@uibk.ac.at
#  Alexander Dummer alexander.dummer@uibk.ac.at
#
#  This file is part of EdelweissFE.
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissFE.
#  ---------------------------------------------------------------------
"""The nonlinear explicit dynamic solver, domain-decomposed over MPI processes.

Started by an MPI launcher, every process reads the same input file and builds the same, complete
model; this solver then has each of them compute one *subdomain* of it -- the elements METIS
assigns to it and the constraints dealt to it -- and integrate the degrees of freedom those touch.
An increment is the increment of :class:`~edelweissfe.solvers.nonlinearexplicitdynamic.NED`, with
two additions. At the interface between subdomains, each process holds only its own elements'
contributions to the nodal force; the neighbours exchange the individual contributions there, and
each process sums all of them in the order of the elements in the model
(:mod:`edelweissfe.domaindecomposition.subdomaininterface`). And the forces of the constraints,
each evaluated by one process, are shared with all. That is the communication most increments
need, and its volume is the interface and the constraints, not the model.

The lumped mass and the damping are assembled once per mesh, by each process from its own elements,
completed at the interface like the forces and shared from the owners, so that they are the same
bits as without decomposition; the critical time step is the minimum over the subdomains. The
lumped operators of the whole model are therefore known everywhere, which is what lets a contact
search move a constraint onto nodes another process integrated until then.

**Where the whole model is read.** Field outputs, output managers, a marker deciding a refinement,
the refinement itself, and a contact search all read more than one subdomain. Before each of them,
every process receives the current solution of every degree of freedom, and -- except before a
contact search, which reads positions only -- the current state of every element and stateful
constraint (:mod:`edelweissfe.domaindecomposition.statesynchronization`). That happens on the
``output-frequency`` cadence, at every ``contact-update-frequency`` search and at the end of a step.
A contact search itself -- at a contact update and at a topology check -- runs on the process that
evaluates the constraint only, since nothing but that evaluation reads its outcome.

**Adaptive refinement.** Because every process holds the complete, synchronized model at a
topology check, every process runs the same refinement on the same data and arrives at the same
refined model -- checked, not assumed: the degree-of-freedom layout is compared across all processes
after every build. The refined model is then partitioned afresh. Nothing migrates, because every
process already holds every element.

**Output.** Only rank 0 creates output managers and writes files; the others are silent. A restart
checkpoint is written after the output synchronization, so the copy of the model rank 0 writes it
from holds every element and constraint as its owner left it: it is an ordinary checkpoint of the
whole model, and can be resumed by this solver on any number of processes, or by the serial one.
Every process resumes from it, and so starts from the same model. A conditional stop decided by an
output manager on rank 0 stops every process.

**Results do not depend on the decomposition.** Every sum that decides the solution is formed in the
order it is formed without decomposition: the element contributions at a node in model order, the
loads and the constraint forces after them in deck order. A run on any number of processes is
therefore bit-identical to :class:`NED` (and :class:`NEDParallel`) on the same input -- through
contact searches, refinements and repartitions -- and the load balancing below, whose partition
depends on measured timings, changes the speed of a run and never its result. Only the sums of the
energy table -- external work, kinetic and internal energy -- are formed per subdomain and then
added, and may differ from a serial run's in their last digits; they enter nothing but the table,
and the external work a checkpoint records.

**Load balancing.** The first partition weighs an element by its number of degrees of freedom. A
softening material costs more where it softens, so every element kernel is timed, and on an output
increment the model is repartitioned with the measured costs whenever the slowest process falls
more than ``load-balance-tolerance`` behind the mean.

**What this solver changes.** Nothing in the increment of ``NED``: the subdomain of its process is
the :class:`~edelweissfe.solvers.base.modelpartition.ModelPartition` the increment computes, a
:class:`~edelweissfe.domaindecomposition.subdomain.Subdomain`, and this solver only creates it and
adds the option of the load balancing.

**Limits of this prototype.** Every process holds the complete model -- every element, with its
material and state, the sets, the degree-of-freedom layout -- so the memory per process does not
shrink with the number of processes, and a refinement costs every process what it costs a serial
run. Holding only the elements a process computes is planned. Constraints are evaluated whole, each by one process. An exception outside the element
and constraint evaluation -- where it is agreed on by all processes -- aborts all of them.

Run with, for example::

    mpirun -n 8 edelweissfe input.inp

with ``*solver, solver=NEDMPI, name=...`` in the deck. ``OMP_NUM_THREADS`` sets the threads of each
process' element loop, as for ``NEDParallel``.
"""

from dataclasses import dataclass

from mpi4py import MPI

from edelweissfe.domaindecomposition.mpienvironment import worldCommunicator
from edelweissfe.domaindecomposition.subdomain import Subdomain
from edelweissfe.models.femodel import FEModel
from edelweissfe.numerics.parallelizationutilities import getNumberOfThreads
from edelweissfe.outputmanagers.base.outputmanagerbase import OutputManagerBase
from edelweissfe.solvers.nonlinearexplicitdynamic import NED, NEDSchema
from edelweissfe.solvers.nonlinearexplicitdynamicparallel import NEDParallel
from edelweissfe.utils.fieldoutput import FieldOutputController
from edelweissfe.utils.schema import schemaField


@dataclass(frozen=True)
class NEDMPISchema(NEDSchema):
    """The options of :class:`NEDMPI`: those of :class:`NEDSchema`, and the load balancing."""

    loadBalanceTolerance: float | None = schemaField(
        description=(
            "How far the slowest process may fall behind the mean, as a fraction, before the model is "
            "repartitioned with the measured element costs as weights. Checked on every output "
            "increment. The first partition can only estimate an element's cost -- by its number of "
            "degrees of freedom -- and a softening material costs more where it softens, so a partition "
            "balanced at the start drifts out of balance as damage localizes. 0 disables it."
        ),
        dtype=float,
        default=0.1,
        optionName="load-balance-tolerance",
    )


class NEDMPI(NEDParallel):
    """The nonlinear explicit dynamic solver, domain-decomposed over MPI processes.

    Parameters
    ----------
    jobInfo
        A dictionary containing the job information.
    journal
        The journal instance for logging.
    """

    identification = "NEDMPISolver"

    supportsDomainDecomposition = True

    schema = NEDMPISchema

    SolverSpecificOptions = NED.SolverSpecificOptions | {"load-balance-tolerance": 0.1}

    def createPartition(self) -> Subdomain:
        """The subdomain this process computes, among all processes of the launcher; see
        :meth:`NED.createPartition`.

        Started without a launcher, this solver computes a subdomain of one: the same code path, with
        every exchange a copy.

        Returns
        -------
        Subdomain
            The subdomain.
        """

        communicator = worldCommunicator()
        return Subdomain(
            MPI.COMM_SELF if communicator is None else communicator,
            self.journal,
            self.identification,
            self.options["load-balance-tolerance"],
        )

    def beginStep(
        self,
        step,
        model: FEModel,
        fieldOutputController: FieldOutputController,
        outputmanagers: list[OutputManagerBase],
    ):
        """Report the decomposition, then start the step; see :meth:`NED.beginStep`.

        Parameters
        ----------
        step
            The step to solve.
        model
            The model tree.
        fieldOutputController
            The field output controller.
        outputmanagers
            The output managers; on processes other than rank 0, none.
        """

        self.journal.message(
            "Domain decomposition over {:} MPI process(es), {:} thread(s) each".format(
                self.partition.nProcesses, getNumberOfThreads()
            ),
            self.identification,
            0,
        )
        return super().beginStep(step, model, fieldOutputController, outputmanagers)
