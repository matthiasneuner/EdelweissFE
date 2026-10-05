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
"""The part of a model one process computes.

An explicit solver may compute a model in one process or split it over several, each computing
one *part* of it -- a subdomain -- and exchanging with the others what they share. The solver's
increment is the same in both cases; what differs is captured by one object, a
:class:`ModelPartition`, which answers two kinds of questions:

* **What is computed here**: the elements, the constraints and the degrees of freedom this process
  integrates (:attr:`~ModelPartition.elements`, :attr:`~ModelPartition.constraints`,
  :attr:`~ModelPartition.dofs`).
* **How the results of this part become results of the whole model**: completing a nodal force at
  the degrees of freedom this part shares with others, adding up a quantity such as an energy over
  all parts, and making the whole model current in this process before something reads all of it --
  an output, a contact search, a refinement.

:class:`WholeModel`, the partition of a solver running in a single process, computes everything
here: its sums are the values themselves, its forces are complete as assembled, and it has nothing
to synchronize. The domain-decomposed implementation is
:class:`~edelweissfe.domaindecomposition.subdomain.Subdomain`.
"""

from abc import ABC, abstractmethod
from contextlib import nullcontext

import numpy as np

from edelweissfe.models.femodel import FEModel
from edelweissfe.numerics.dofmanager import DofManager, DofVector
from edelweissfe.numerics.mpctransformation import MultiPointConstraintTransformation
from edelweissfe.solvers.base.parallelelementcomputation import ElementPlan


def addNodalForces(vector: np.ndarray, dofs: np.ndarray, forces: np.ndarray, namesDofMoreThanOnce: bool):
    """Add the nodal forces of one entity -- a constraint -- into a plain vector, in place.

    Parameters
    ----------
    vector
        The vector, as a plain array.
    dofs
        The degrees of freedom the forces act on.
    forces
        The forces.
    namesDofMoreThanOnce
        Whether a degree of freedom appears in ``dofs`` more than once -- a slave node that also
        appears in its own master facet's node list. ``+=`` would then keep only the last write
        instead of summing, so the (much slower) ``np.add.at`` is used then, and only then.
    """

    if namesDofMoreThanOnce:
        np.add.at(vector, dofs, forces)
    else:
        vector[dofs] += forces


class _ForcesAreComplete:
    """The interface assembly of a part that shares no degree of freedom: nothing to complete."""

    def assemble(self, contributions: np.ndarray, vector: DofVector):
        """Leave ``vector`` as it is: its assembled entries are complete.

        Parameters
        ----------
        contributions
            The contribution buffer of an element plan.
        vector
            The vector the plan assembled into.
        """


class ModelPartition(ABC):
    """Which part of the model this process computes, and how its partial results become results of
    the whole model; see the module documentation.

    Every method marked *collective* must be called by every process computing a part of the model,
    at the same point of the computation.
    """

    @abstractmethod
    def define(
        self,
        model: FEModel,
        dofManager: DofManager,
        mpcTransformation: MultiPointConstraintTransformation | None,
        topologyChanged: bool,
    ):
        """Decide which part of the model this process computes. Collective.

        Called whenever the equation system has been built (``topologyChanged``) or its constraints
        re-located (a contact search moved them); everything else this object answers refers to the
        last definition.

        Parameters
        ----------
        model
            The model tree.
        dofManager
            The degree-of-freedom layout of the current equation system.
        mpcTransformation
            The multi-point-constraint transformation of the current equation system, or None.
        topologyChanged
            True after the equation system was built afresh; False when only the constraints were
            re-located.
        """

    @property
    @abstractmethod
    def elements(self) -> dict:
        """The elements computed here, by number, in model order."""

    @property
    @abstractmethod
    def constraints(self) -> dict:
        """The constraints evaluated here, by name, in model order."""

    @property
    @abstractmethod
    def dofs(self) -> slice | np.ndarray:
        """The degrees of freedom integrated here, as an index into a vector of the whole model: a
        slice, or a sorted index array."""

    @abstractmethod
    def integratedValuesOf(self, vector: np.ndarray) -> np.ndarray:
        """The entries of a plain vector of the whole model at the degrees of freedom integrated
        here, in order: a view into ``vector`` where they are all of them, a copy otherwise.

        What an update of the integrated degrees of freedom computes on; write the result back with
        :meth:`storeIntegratedValues`.

        Parameters
        ----------
        vector
            A plain vector of the whole model, or a per-DOF array of its length.

        Returns
        -------
        np.ndarray
            The entries.
        """

    @abstractmethod
    def storeIntegratedValues(self, vector: np.ndarray, values: np.ndarray):
        """Write values at the degrees of freedom integrated here, as :meth:`integratedValuesOf`
        returned them, into a plain vector of the whole model.

        Parameters
        ----------
        vector
            The plain vector.
        values
            The values, one per degree of freedom integrated here.
        """

    @abstractmethod
    def integratedEntries(self, indices: np.ndarray) -> slice | np.ndarray:
        """Which of the given degrees of freedom are integrated here.

        Parameters
        ----------
        indices
            Degree-of-freedom indices.

        Returns
        -------
        slice | np.ndarray
            An index into ``indices``: ``slice(None)`` if all of them are.
        """

    @abstractmethod
    def ownedEntries(self, indices: np.ndarray) -> slice | np.ndarray:
        """Which of the given degrees of freedom this process owns.

        A degree of freedom shared by several parts is integrated by each of them but owned by one;
        a quantity to be counted once for the model -- the work done at a prescribed degree of
        freedom, a kinetic energy -- is counted by the owner only.

        Parameters
        ----------
        indices
            Degree-of-freedom indices.

        Returns
        -------
        slice | np.ndarray
            An index into ``indices``: ``slice(None)`` if all of them are owned.
        """

    @abstractmethod
    def loadedElements(self, elements) -> list:
        """Those of the given elements -- the elements of a distributed load or a body force --
        whose load is added here: those reaching a degree of freedom integrated here.

        Parameters
        ----------
        elements
            An iterable of elements of the model.

        Returns
        -------
        list
            Those elements, in the order given.
        """

    @abstractmethod
    def constraintsSearchedHere(self, model: FEModel, constraints: dict) -> dict:
        """Those of the given constraints whose connectivity search is run here.

        Parameters
        ----------
        model
            The model tree.
        constraints
            Constraints, by name.

        Returns
        -------
        dict
            Those constraints, by name.
        """

    @abstractmethod
    def interfaceAssemblyFor(self, plan: ElementPlan):
        """How the contributions of the elements of a plan are completed at the degrees of freedom
        this part shares with others. Collective.

        Parameters
        ----------
        plan
            A plan of elements computed here.

        Returns
        -------
        object
            An object whose ``assemble(contributions, vector)`` -- collective -- overwrites the
            shared entries of ``vector`` with the sums of all parts' contributions there, in model
            order.
        """

    @abstractmethod
    def shareFromOwners(self, vector: DofVector):
        """Make a vector, correct at the degrees of freedom owned here, the same complete vector in
        every part, in place. Collective.

        Parameters
        ----------
        vector
            A vector of the whole model.
        """

    @abstractmethod
    def addConstraintForces(self, forces: dict, P: DofVector):
        """Add the nodal forces of every constraint of the model into ``P``, in model order.
        Collective.

        Parameters
        ----------
        forces
            The :class:`~edelweissfe.solvers.nonlinearexplicitdynamic.ConstraintForce` of every
            constraint evaluated here, by name.
        P
            The net nodal force vector.
        """

    @abstractmethod
    def synchronizeStates(self, includeElements: bool):
        """Give every stateful constraint -- and, if asked, every element -- of this process' model
        the state the part computing it last left it in. Collective.

        Parameters
        ----------
        includeElements
            Whether the element states are synchronized as well.
        """

    @abstractmethod
    def sumAcrossParts(self, values: list[float]) -> list[float]:
        """The sums of values over all parts, the same bits in every part. Collective.

        Parameters
        ----------
        values
            This part's contributions.

        Returns
        -------
        list[float]
            The sum of every entry over all parts.
        """

    @abstractmethod
    def minAcrossParts(self, value: float) -> float:
        """The minimum of a value over all parts. Collective.

        Parameters
        ----------
        value
            This part's value.

        Returns
        -------
        float
            The minimum.
        """

    @abstractmethod
    def anyPart(self, flag: bool) -> bool:
        """Whether a flag is set in any part. Collective.

        Parameters
        ----------
        flag
            This part's flag.

        Returns
        -------
        bool
            Whether any part set it.
        """

    @abstractmethod
    def requireSameOnAllParts(self, value, description: str):
        """Refuse to continue unless every part holds the same value. Collective.

        Parameters
        ----------
        value
            This part's value; anything comparable for equality.
        description
            What the value says, for the message.

        Raises
        ------
        RuntimeError
            In every part, if two parts hold different values.
        """

    @abstractmethod
    def agreedOnByAllParts(self, operation: str):
        """A context in which an exception raised in one part is raised in every part. Collective.

        A part raising alone would leave the others waiting for it in the next exchange forever. A
        :class:`~edelweissfe.utils.exceptions.ConditionalStop` or a
        :class:`~edelweissfe.utils.exceptions.CutbackRequest` anywhere is raised as such
        everywhere, so that every part takes the same path out of the step.

        Parameters
        ----------
        operation
            What is done in the context, for the message.

        Returns
        -------
        contextlib.AbstractContextManager
            The context.
        """

    @abstractmethod
    def shareOfModelTotal(self, value: float) -> float:
        """This part's share of a total of the whole model restored from a checkpoint -- the external
        work: the whole value in one part, nothing in the others, so that the sum over all parts is
        the total.

        Parameters
        ----------
        value
            The total of the whole model.

        Returns
        -------
        float
            This part's share.
        """

    @abstractmethod
    def measuresElementCosts(self) -> bool:
        """Whether the element kernels are to be timed, for :meth:`rebalance`.

        Returns
        -------
        bool
            Whether to time them.
        """

    @abstractmethod
    def rebalance(self, plan: ElementPlan, costs: np.ndarray | None, nIncrements: int) -> bool:
        """Re-partition the model with the measured element costs, if the parts have drifted out of
        balance. Collective; only when every element state has just been synchronized, since an
        element changing part must arrive with its current state.

        Parameters
        ----------
        plan
            The plan of the elements whose kernels are computed here.
        costs
            The measured kernel time of every element of the plan, in plan order, or None.
        nIncrements
            The increments the costs were measured over.

        Returns
        -------
        bool
            Whether the partition changed; then everything derived from it must be derived again.
        """


class WholeModel(ModelPartition):
    """The partition of a single process: everything is computed here.

    Every element, constraint and degree of freedom is this part's, a force is complete as
    assembled, a sum over the parts is the value itself, and there is nothing to synchronize.
    """

    def __init__(self):
        self._model = None

    def define(self, model, dofManager, mpcTransformation, topologyChanged):
        """Remember the model; everything in it is computed here. See :meth:`ModelPartition.define`."""

        self._model = model

    @property
    def elements(self) -> dict:
        """All elements of the model."""

        return self._model.elements

    @property
    def constraints(self) -> dict:
        """All constraints of the model."""

        return self._model.constraints

    @property
    def dofs(self) -> slice:
        """All degrees of freedom."""

        return slice(None)

    def integratedValuesOf(self, vector: np.ndarray) -> np.ndarray:
        """The vector itself."""

        return vector

    def storeIntegratedValues(self, vector: np.ndarray, values: np.ndarray):
        """Nothing to store: the values are the vector."""

    def integratedEntries(self, indices: np.ndarray) -> slice:
        """All of them."""

        return slice(None)

    def ownedEntries(self, indices: np.ndarray) -> slice:
        """All of them."""

        return slice(None)

    def loadedElements(self, elements) -> list:
        """All of them."""

        return elements

    def constraintsSearchedHere(self, model: FEModel, constraints: dict) -> dict:
        """All of them."""

        return constraints

    def interfaceAssemblyFor(self, plan: ElementPlan) -> _ForcesAreComplete:
        """Nothing to complete: no degree of freedom is shared."""

        return _ForcesAreComplete()

    def shareFromOwners(self, vector: DofVector):
        """Nothing to share: the vector is complete."""

    def addConstraintForces(self, forces: dict, P: DofVector):
        """Add the forces of every constraint, in model order."""

        PPlain = P.asPlainArray()
        for constraintForce in forces.values():
            constraintForce.addInto(PPlain)

    def synchronizeStates(self, includeElements: bool):
        """Nothing to synchronize: every state was computed here."""

    def sumAcrossParts(self, values: list[float]) -> list[float]:
        """The values themselves."""

        return list(values)

    def minAcrossParts(self, value: float) -> float:
        """The value itself."""

        return value

    def anyPart(self, flag: bool) -> bool:
        """The flag itself."""

        return bool(flag)

    def requireSameOnAllParts(self, value, description: str):
        """Nothing to compare."""

    def agreedOnByAllParts(self, operation: str) -> nullcontext:
        """No other part to agree with: an exception is raised as it is."""

        return nullcontext()

    def shareOfModelTotal(self, value: float) -> float:
        """The whole value."""

        return value

    def measuresElementCosts(self) -> bool:
        """No: there is nothing to balance."""

        return False

    def rebalance(self, plan: ElementPlan, costs: np.ndarray | None, nIncrements: int) -> bool:
        """Nothing to balance."""

        return False
