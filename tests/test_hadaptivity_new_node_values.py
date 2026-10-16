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
"""What lets a process holding only the nodes of its subdomain interpolate the new nodes of a refinement
it shares with another process, with the same bits (:class:`~edelweissfe.modelmodifiers.adaptivity.hadaptivity.NewNodeValues`):
at a new node on a face or an edge of the parent, the shape functions of the parent vanish exactly at
every node off that face or edge, so those nodes enter the sum as exact zeros, and the nodes of the face
or edge -- which the process sharing it holds -- are all the sum reads.
"""

import numpy as np
import pytest

from edelweissfe.adaptivity.hex20shapefunctions import (
    EDGES,
    FACES,
    hex20_shape,
    subdivision_children_param,
)


def _onBoundaryOfParent(parameters: np.ndarray) -> list:
    """The nodes of the parent's faces and edges a point of the parent's reference cube lies on."""

    reference = np.array(subdivision_children_param(1)[0])
    nodes = []
    for entity in list(FACES) + list(EDGES):
        coordinates = reference[list(entity)]
        # a face or an edge is where the coordinates its nodes share are those of the point
        shared = [axis for axis in range(3) if np.all(coordinates[:, axis] == coordinates[0, axis])]
        if shared and all(parameters[axis] == coordinates[0, axis] for axis in shared):
            nodes.append(set(entity))
    return nodes


@pytest.mark.parametrize("splitFactor", [2, 3])
def test_a_new_node_on_a_face_or_edge_reads_only_the_nodes_of_that_face_or_edge(splitFactor):
    rng = np.random.default_rng(7)
    values = rng.standard_normal((20, 3))
    checked = 0
    for child in subdivision_children_param(splitFactor):
        for parameters in np.asarray(child):
            entities = _onBoundaryOfParent(parameters)
            if not entities:
                continue
            N = hex20_shape(*parameters)
            weighted = set(np.flatnonzero(N != 0.0).tolist())
            # the smallest face or edge the node lies on holds every node with a weight
            assert weighted <= min(entities, key=len)
            # summed with exact zeros for the nodes without weight, the bits of the full sum
            substituted = np.where((N != 0.0)[:, None], values, 0.0)
            assert (N @ substituted).tobytes() == (N @ values).tobytes()
            checked += 1
    assert checked > 0
