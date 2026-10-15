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
"""The subdomain: the part of a model an explicit increment runs over.

An explicit increment runs over three things, and they are one small object, a :class:`Subdomain`:

* the **elements** whose forces it computes, and the **constraints** whose forces it evaluates;
* the **degrees of freedom** it integrates -- every one those elements and constraints touch.

A solver computing the whole model in one process -- ``NED``, ``NEDParallel`` -- runs over
:meth:`Subdomain.wholeModel`: every element, constraint and degree of freedom. The degrees of freedom
are then the slice ``slice(None)``, so that indexing a vector with them, ``V[subdomain.dofs]``, is a
view of the whole vector and costs nothing.

In a domain-decomposed run (:class:`~edelweissfe.solvers.nonlinearexplicitdynamicmpi.NEDMPI`), each
process computes one subdomain of the model: the elements a graph partitioner assigned to it, the
constraints dealt to it, and the degrees of freedom those touch: ``slice(None)`` again where its vectors
hold the subdomain only, else a sorted index array into vectors of the whole model. How the
subdomains exchange what they share -- the forces at their common degrees of freedom, the constraint
forces, the states before an output -- and which process counts a degree of freedom two of them
integrate is not the subdomain's business but the decomposition's; see
:class:`~edelweissfe.domaindecomposition.domaindecomposition.DomainDecomposition`.
"""

from dataclasses import dataclass

import numpy as np

from edelweissfe.models.femodel import FEModel


@dataclass(frozen=True)
class Subdomain:
    """The elements, constraints and degrees of freedom an explicit increment runs over; see the
    module documentation.

    Parameters
    ----------
    elements
        The elements of the subdomain, by number, in model order.
    constraints
        The constraints of the subdomain, by name, in model order.
    dofs
        The subdomain degrees of freedom -- those its elements and constraints touch -- as an index
        into the vectors of the process: ``slice(None)`` for all of them, or a sorted index array.
    """

    elements: dict
    constraints: dict
    dofs: slice | np.ndarray

    @classmethod
    def wholeModel(cls, model: FEModel) -> "Subdomain":
        """The subdomain of a solver computing the whole model: every element, constraint and degree of
        freedom.

        Parameters
        ----------
        model
            The model tree.

        Returns
        -------
        Subdomain
            The subdomain.
        """

        return cls(model.elements, model.constraints, slice(None))
