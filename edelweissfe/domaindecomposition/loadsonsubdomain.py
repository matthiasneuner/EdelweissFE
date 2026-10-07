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
"""The loads of a model restricted to the elements one subdomain computes.

A distributed load or a body force acting on an element is evaluated by the process computing the
element (:meth:`~edelweissfe.domaindecomposition.subdomain.Subdomain.loadsOnSubdomain`). The classes
here describe a load of the model restricted to those elements -- what the shared load assembly of
the solvers reads of a load, with a smaller surface or element set -- and the loads of a subdomain
together, for as long as the loads of the model are the same.
"""

import numpy as np

from edelweissfe.stepactions.base.bodyloadbase import BodyLoadBase
from edelweissfe.stepactions.base.distributedloadbase import DistributedLoadBase
from edelweissfe.timesteppers.timestep import TimeStep


class DistributedLoadOnSubdomain:
    """A distributed load, restricted to the faces of the elements computed in a subdomain.

    What the shared load assembly
    (:meth:`~edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase.computeDistributedLoads`)
    reads of a :class:`~edelweissfe.stepactions.base.distributedloadbase.DistributedLoadBase`, with
    a smaller surface.

    Parameters
    ----------
    load
        The distributed load.
    surface
        The faces it acts on in the subdomain: element lists by face, in the load's order.
    """

    def __init__(self, load: DistributedLoadBase, surface: dict):
        self.load = load
        self.surface = surface

    @property
    def loadType(self) -> str:
        """The load's type."""

        return self.load.loadType

    def getCurrentLoad(self, timeStep: TimeStep) -> np.ndarray:
        """The load's current magnitude.

        Parameters
        ----------
        timeStep
            The time step.

        Returns
        -------
        np.ndarray
            The magnitude.
        """

        return self.load.getCurrentLoad(timeStep)


class BodyLoadOnSubdomain:
    """A body load, restricted to the elements computed in a subdomain.

    What the shared load assembly
    (:meth:`~edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase.computeBodyForces`)
    reads of a :class:`~edelweissfe.stepactions.base.bodyloadbase.BodyLoadBase`, with a smaller
    element set.

    Parameters
    ----------
    load
        The body load.
    elementSet
        The elements it acts on in the subdomain, in the load's order.
    """

    def __init__(self, load: BodyLoadBase, elementSet: list):
        self.load = load
        self.elementSet = elementSet

    def getCurrentLoad(self, timeStep: TimeStep) -> np.ndarray:
        """The load's current magnitude.

        Parameters
        ----------
        timeStep
            The time step.

        Returns
        -------
        np.ndarray
            The magnitude.
        """

        return self.load.getCurrentLoad(timeStep)


class LoadsOnSubdomain:
    """The distributed and body loads of a step, restricted to the elements computed in a subdomain,
    and the tags of their nodal forces for the assembly completing them at the interface; see
    :meth:`Subdomain.loadsOnSubdomain` and :meth:`Subdomain.loadAssemblyFor`.

    Parameters
    ----------
    distributedLoads
        The distributed loads of the step, as given.
    bodyLoads
        The body loads of the step, as given.
    restrictedDistributedLoads
        The distributed loads, restricted to the elements computed here.
    restrictedBodyLoads
        The body loads, restricted to the elements computed here.
    entryDofs
        The degree of freedom of every entry of their nodal forces, in the order the restricted loads
        are evaluated.
    entryLoadOrder
        The place of every entry in the order of all load contributions of the model.
    """

    def __init__(
        self,
        distributedLoads: list,
        bodyLoads: list,
        restrictedDistributedLoads: list[DistributedLoadOnSubdomain],
        restrictedBodyLoads: list[BodyLoadOnSubdomain],
        entryDofs: np.ndarray,
        entryLoadOrder: np.ndarray,
    ):
        self._loads = (distributedLoads, bodyLoads)
        #: The distributed loads, restricted to the elements computed here.
        self.distributedLoads = restrictedDistributedLoads
        #: The body loads, restricted to the elements computed here.
        self.bodyLoads = restrictedBodyLoads
        #: The degree of freedom of every entry of their nodal forces.
        self.entryDofs = entryDofs
        #: The place of every entry in the order of all load contributions of the model.
        self.entryLoadOrder = entryLoadOrder
        #: The assembly of their nodal forces, once built (:meth:`Subdomain.loadAssemblyFor`).
        self.assembly = None

    def isFor(self, distributedLoads: list, bodyLoads: list) -> bool:
        """Whether these are the restrictions of the given loads.

        Parameters
        ----------
        distributedLoads
            The distributed loads.
        bodyLoads
            The body loads.

        Returns
        -------
        bool
            Whether they are the loads given here, the same objects in the same order.
        """

        given = (distributedLoads, bodyLoads)
        return all(
            len(mine) == len(theirs) and all(a is b for a, b in zip(mine, theirs))
            for mine, theirs in zip(self._loads, given)
        )
