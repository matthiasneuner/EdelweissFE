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
"""Which elements of the mesh a process creates, and whose results it reports.

A model is built in two stages: the mesh is described as data, then the element objects are made
from it (:mod:`~edelweissfe.models.mesh`). Between the two stages an :class:`ElementDistribution`
decides which elements *this* process creates. A serial run creates every element, and that is what
this class does: every element is created here, every element's results and state are reported
from here, and a result over an element set is simply the result of its elements.

A domain-decomposed run that computes each subdomain in its own process may create only the
elements of its subdomain; its distribution
(:class:`~edelweissfe.domaindecomposition.distributedelements.DistributedElements`) then also says
how a result of a whole element set, or the state of the whole model, is gathered from the
processes that computed it. Code that reads results of a whole element set or of the whole model
goes through the methods below, and so reads the same thing in both cases.
"""

from collections.abc import Iterator

import numpy as np

from edelweissfe.models.mesh import Mesh


class ElementDistribution:
    """Every element of the mesh is created here, and reported from here; see the module
    documentation."""

    #: Whether every process creates every element of the mesh.
    createsEveryElement = True

    #: Changed whenever elements move between processes, so that whatever was derived from the
    #: elements this process holds or reports (e.g. the result views of a field output) is derived
    #: again. Here: never.
    ownershipVersion = 0

    def decideWhichElementsAreCreatedHere(self, mesh: Mesh, domainSize: int):
        """Decide, once the mesh is described and before any element exists, which of its elements
        this process creates. Here: all of them, so there is nothing to decide.

        Parameters
        ----------
        mesh
            The mesh, as described by the input file and the generators.
        domainSize
            The spatial dimension.
        """

    def isCreatedHere(self, number: int) -> bool:
        """Whether this process creates the element with the given number; asked by
        :meth:`~edelweissfe.models.femodel.FEModel.createElementsOfMesh`.

        Parameters
        ----------
        number
            The element number.

        Returns
        -------
        bool
            Always True.
        """

        return True

    def elementsReportedHere(self, elements) -> list:
        """Those of the given elements whose results and state this process reports: the elements it
        computes. Here: all of them.

        Parameters
        ----------
        elements
            Elements created here, e.g. an :class:`~edelweissfe.sets.elementset.ElementSet`.

        Returns
        -------
        list
            The elements reported here, in the order given.
        """

        return list(elements)

    def resultsOfWholeSet(self, elementSet, numbersReportedHere: list, results: np.ndarray | None) -> np.ndarray:
        """The results of every element of a set, in set order, from the results of the elements
        reported here. Here: the results given, since every element is reported here.

        Parameters
        ----------
        elementSet
            The element set.
        numbersReportedHere
            The numbers of the elements of the set reported here (:meth:`elementsReportedHere`), in set
            order.
        results
            Their results, one row per element; None if there are none.

        Returns
        -------
        np.ndarray
            The results of every element of the set, one row per element, in set order.
        """

        return results

    def gatherStatesForCheckpoint(self, elements: dict):
        """Gather the states of the elements computed in other processes, for a restart checkpoint.
        Here: nothing to gather, every element is here.

        Parameters
        ----------
        elements
            The elements created here, by number.
        """

    def forgetGatheredStates(self):
        """Release what :meth:`gatherStatesForCheckpoint` gathered. Here: nothing."""

    def statesOfElements(self, elements: dict) -> Iterator[tuple[int, np.ndarray]]:
        """The converged state of every element of the model, for a restart checkpoint, by element
        number. Here: the state of every element, all created here.

        Parameters
        ----------
        elements
            The elements created here, by number.

        Returns
        -------
        Iterator
            ``(number, state)`` pairs, one per element of the mesh that has an object somewhere.
        """

        return ((number, element.getStateVars()) for number, element in elements.items())
