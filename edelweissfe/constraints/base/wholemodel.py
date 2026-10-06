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
"""What a constraint needs of the model in a domain-decomposed run.

A constraint is evaluated whole, by one process. It can run on a distributed model -- where each
process creates only its own elements -- if it reads only what every process holds whole: the mesh,
the nodes and their fields, contact facets and rigid bodies. A constraint says whether it does
through ``wholeModelReason`` (see
:func:`~edelweissfe.domaindecomposition.distributedelements.reasonsForTheWholeModel`): None if it
does, otherwise one of the reasons below, or its own.
"""

from edelweissfe.models.femodel import FEModel
from edelweissfe.sets.elementset import ElementSet

#: The reason of a constraint not yet verified, by a distributed test case, to read only what every
#: process holds.
NOT_YET_VERIFIED = (
    "is not yet known to read only what every process holds (the mesh, the nodes, contact facets, rigid bodies)"
)

#: The reason of a constraint with scalar variables of its own (Lagrange multipliers): only the
#: implicit solvers solve for them, and the domain-decomposed solver is explicit.
IMPLICIT_ONLY = "introduces scalar variables (Lagrange multipliers), which only the implicit solvers solve for"


def wholeFacetSet(model: FEModel, facetSetName: str, constraintName: str) -> ElementSet:
    """The contact facets of a surface, which a constraint reads whole -- every facet of the surface,
    in every process, see :meth:`~edelweissfe.models.femodel.FEModel.wholeElementSet`.

    Parameters
    ----------
    model
        The model tree.
    facetSetName
        The name of the element set of the facets.
    constraintName
        The name of the constraint reading them, for the error message.

    Returns
    -------
    ElementSet
        The facets.
    """

    return model.wholeElementSet(facetSetName, "constraint {:}".format(constraintName))
