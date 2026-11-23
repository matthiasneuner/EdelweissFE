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

import numpy as np


class DiscreteSurfaceQuery:
    def __init__(self, mesh):
        """
        The outer surface of a rigid body's mesh and the outward normals of its cells, from which the rigid body takes
        its reference triangles
        (:meth:`~edelweissfe.rigidbodies.discreterigidbody.DiscreteRigidBody.referenceTriangles`).

        Parameters
        ----------
        mesh : pyvista.PolyData
            The PyVista mesh.
        """

        # Extract the outer surface to ensure we have a PolyData object
        self.mesh = mesh.extract_surface(algorithm="dataset_surface")

        # Outward cell normals, oriented from the (closed, manifold) mesh topology rather than the file's winding.
        self.mesh.compute_normals(cell_normals=True, point_normals=False, auto_orient_normals=True, inplace=True)
        self.mesh_cell_normals = np.array(self.mesh.cell_normals)
