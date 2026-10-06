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

An explicit solver may compute a model in one process, or split it into *parts* -- subdomains --
computed by several processes at once. Its increment is the same in both cases; what differs is
which elements, constraints and degrees of freedom it runs over, and that is one small object, a
:class:`ModelPartition`:

* the **elements** computed here, and the **constraints** evaluated here;
* the **degrees of freedom** integrated here -- every one touched by those elements and
  constraints;
* which of those degrees of freedom are **owned** here. A degree of freedom at the boundary between
  two parts is integrated by both, but a quantity to be counted once for the model -- the work done
  at a prescribed degree of freedom, a kinetic energy -- is counted by its owner only.

A solver computing the whole model in one process uses :meth:`ModelPartition.wholeModel`: every
element, constraint and degree of freedom, all of them owned. The degrees of freedom are then the
slice ``slice(None)``, so that indexing a vector with them, ``V[partition.dofs]``, is a view of the
whole vector and costs nothing.

How the parts exchange what they share -- the forces at their common degrees of freedom, the
constraint forces, the states before an output -- is not the partition's business but the
domain-decomposed solver's; see :class:`~edelweissfe.solvers.nonlinearexplicitdynamicmpi.NEDMPI`.
"""

from dataclasses import dataclass

import numpy as np

from edelweissfe.models.femodel import FEModel


@dataclass(frozen=True)
class ModelPartition:
    """The elements, constraints and degrees of freedom one process computes; see the module
    documentation.

    Parameters
    ----------
    elements
        The elements computed here, by number, in model order.
    constraints
        The constraints evaluated here, by name, in model order.
    dofs
        The degrees of freedom integrated here, as an index into a vector of the whole model:
        ``slice(None)`` for all of them, or a sorted index array.
    ownedDofMask
        Whether each degree of freedom of the model is owned here: True at most at those integrated
        here, and at each degree of freedom of the model in exactly one part.
    """

    elements: dict
    constraints: dict
    dofs: slice | np.ndarray
    ownedDofMask: np.ndarray

    @classmethod
    def wholeModel(cls, model: FEModel, nDof: int) -> "ModelPartition":
        """The partition of a process computing the whole model: everything is computed and owned
        here.

        Parameters
        ----------
        model
            The model tree.
        nDof
            The size of the equation system.

        Returns
        -------
        ModelPartition
            The partition.
        """

        return cls(model.elements, model.constraints, slice(None), np.ones(nDof, dtype=bool))
