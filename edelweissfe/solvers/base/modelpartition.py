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
  constraints.

A solver computing the whole model in one process uses :meth:`ModelPartition.wholeModel`: every
element, constraint and degree of freedom. The degrees of freedom are then the slice
``slice(None)``, so that indexing a vector with them, ``V[partition.dofs]``, is a view of the whole
vector and costs nothing.

How the parts exchange what they share -- the forces at their common degrees of freedom, the
constraint forces, the states before an output -- and which part counts a degree of freedom two of
them integrate is not the partition's business but the domain-decomposed solver's; see
:class:`~edelweissfe.solvers.nonlinearexplicitdynamicmpi.NEDMPI`.
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
    """

    elements: dict
    constraints: dict
    dofs: slice | np.ndarray

    @classmethod
    def wholeModel(cls, model: FEModel) -> "ModelPartition":
        """The partition of a process computing the whole model: everything is computed here.

        Parameters
        ----------
        model
            The model tree.

        Returns
        -------
        ModelPartition
            The partition.
        """

        return cls(model.elements, model.constraints, slice(None))
