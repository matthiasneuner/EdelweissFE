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
"""Adding the contributions of entities into the vectors of the equation system."""

import numpy as np


def hasRepeatedDofs(dofs: np.ndarray) -> bool:
    """Whether a degree of freedom appears in ``dofs`` more than once: a slave node that also appears
    in its own master facet's node list, a degenerate element with repeated nodes.

    Parameters
    ----------
    dofs
        The degrees of freedom of one entity.

    Returns
    -------
    bool
        True if any appears more than once.
    """

    return len(np.unique(dofs)) != len(dofs)


def addNodalForces(vector: np.ndarray, dofs: np.ndarray, forces: np.ndarray, namesDofMoreThanOnce: bool):
    """Add the nodal forces of one entity -- an element, a constraint -- into a plain vector, in place.

    Parameters
    ----------
    vector
        The vector, as a plain array.
    dofs
        The degrees of freedom the forces act on.
    forces
        The forces.
    namesDofMoreThanOnce
        Whether a degree of freedom appears in ``dofs`` more than once (:func:`hasRepeatedDofs`).
        ``+=`` would then keep only the last write
        instead of summing, so the (much slower) ``np.add.at`` is used then, and only then.
    """

    if namesDofMoreThanOnce:
        np.add.at(vector, dofs, forces)
    else:
        vector[dofs] += forces
