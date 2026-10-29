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
"""
Shape functions, their parametric derivatives and the inverse isoparametric mapping of the standard element shapes,
identified by their Ensight names (the ``ensightType`` of an element).

The node orderings are the ones of Marmot (and Abaqus):

* ``bar2``: :math:`\\xi = -1, +1`; ``bar3``: :math:`\\xi = -1, +1, 0` (end, end, mid).
* ``quad4``: corners counter-clockwise from :math:`(-1, -1)`; ``quad8``: corners, then the midsides, the midside
  ``4 + k`` between the corners ``k`` and ``k + 1``.
* ``hexa8``: the four corners of the face :math:`\\zeta = -1`, then those of :math:`\\zeta = +1`, each
  counter-clockwise from :math:`(-1, -1)`; ``hexa20``: corners, the midsides of the bottom face, of the top face, and
  of the vertical edges.
* ``tetra4``: :math:`N = (1 - \\xi - \\eta - \\zeta, \\xi, \\eta, \\zeta)`.

Used to embed points (e.g., the nodes or integration points of reinforcement bars) in host elements, see
:mod:`edelweissfe.utils.embedding`.
"""

import numpy as np

_QUAD_CORNERS = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
_QUAD_MIDSIDES = np.array([[0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
_HEXA_CORNERS = np.array(
    [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ]
)
_HEXA_MIDSIDES = np.array(
    [
        [0.0, -1.0, -1.0],
        [1.0, 0.0, -1.0],
        [0.0, 1.0, -1.0],
        [-1.0, 0.0, -1.0],
        [0.0, -1.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [-1.0, 0.0, 1.0],
        [-1.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
    ]
)

#: Parametric coordinates of the nodes of each supported shape.
NODE_PARAMETRIC_COORDINATES = {
    "bar2": np.array([[-1.0], [1.0]]),
    "bar3": np.array([[-1.0], [1.0], [0.0]]),
    "quad4": _QUAD_CORNERS,
    "quad8": np.vstack([_QUAD_CORNERS, _QUAD_MIDSIDES]),
    "hexa8": _HEXA_CORNERS,
    "hexa20": np.vstack([_HEXA_CORNERS, _HEXA_MIDSIDES]),
    "tetra4": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
}

#: The parametric dimension of each supported shape.
PARAMETRIC_DIMENSION = {shape: coords.shape[1] for shape, coords in NODE_PARAMETRIC_COORDINATES.items()}


def _productsOfOtherFactors(factors: np.ndarray) -> np.ndarray:
    """For factors ``(nNodes, dim)``, the products over all factors of a node except the ``d``-th one, ``(nNodes,
    dim)``: the ``d``-th factor is replaced by 1, so the remaining factors are multiplied in their order."""
    dim = factors.shape[1]
    others = np.repeat(factors[:, None, :], dim, axis=1)
    others[:, np.arange(dim), np.arange(dim)] = 1.0
    return np.prod(others, axis=2)


def _lagrangeTensorProductLinear(xi: np.ndarray, nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Multilinear shape functions of the corner nodes and their derivatives."""
    factors = 0.5 * (1.0 + nodes * xi)  # (nNodes, dim)
    N = np.prod(factors, axis=1)
    dN = 0.5 * nodes * _productsOfOtherFactors(factors)
    return N, dN


def _serendipity(xi: np.ndarray, corners: np.ndarray, midsides: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quadratic serendipity shape functions (quad8, hexa20) and their derivatives."""
    dim = corners.shape[1]

    # corners: prod(1 + xi_i xi) / 2^dim * (sum(xi_i xi) - (dim - 1))
    f = 1.0 + corners * xi
    p = np.prod(f, axis=1) / 2**dim
    s = corners @ xi - (dim - 1)
    dp = corners * _productsOfOtherFactors(f) / 2**dim
    NCorners = p * s
    dNCorners = dp * s[:, None] + p[:, None] * corners

    # midsides: the coordinate with xi_i = 0 enters as (1 - xi^2), the others as (1 + xi_i xi) / 2
    isZero = midsides == 0.0
    g = np.where(isZero, 1.0 - xi**2, 0.5 * (1.0 + midsides * xi))
    dg = np.where(isZero, -2.0 * xi, 0.5 * midsides)
    NMidsides = np.prod(g, axis=1)
    dNMidsides = dg * _productsOfOtherFactors(g)

    return np.concatenate([NCorners, NMidsides]), np.vstack([dNCorners, dNMidsides])


def shapeFunctionsAndDerivatives(shape: str, xi) -> tuple[np.ndarray, np.ndarray]:
    """The shape functions and their derivatives w.r.t. the parametric coordinates.

    Parameters
    ----------
    shape
        The Ensight name of the shape, e.g., ``quad8``.
    xi
        The parametric coordinates.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The shape functions ``(nNodes,)`` and their derivatives ``(nNodes, parametric dimension)``.
    """
    xi = np.atleast_1d(np.asarray(xi, dtype=float))
    if shape in ("quad4", "hexa8"):
        return _lagrangeTensorProductLinear(xi, NODE_PARAMETRIC_COORDINATES[shape])
    if shape == "quad8":
        return _serendipity(xi, _QUAD_CORNERS, _QUAD_MIDSIDES)
    if shape == "hexa20":
        return _serendipity(xi, _HEXA_CORNERS, _HEXA_MIDSIDES)
    if shape == "bar2":
        return np.array([0.5 * (1 - xi[0]), 0.5 * (1 + xi[0])]), np.array([[-0.5], [0.5]])
    if shape == "bar3":
        x = xi[0]
        return np.array([0.5 * (x - 1) * x, 0.5 * (x + 1) * x, 1 - x * x]), np.array([[x - 0.5], [x + 0.5], [-2 * x]])
    if shape == "tetra4":
        N = np.array([1.0 - xi.sum(), xi[0], xi[1], xi[2]])
        dN = np.vstack([-np.ones(3), np.eye(3)])
        return N, dN
    raise NotImplementedError(f"Shape functions of '{shape}' elements are not implemented.")


def shapeFunctions(shape: str, xi) -> np.ndarray:
    """The shape functions at the parametric coordinates ``xi``, see :func:`shapeFunctionsAndDerivatives`."""
    return shapeFunctionsAndDerivatives(shape, xi)[0]


def isInside(shape: str, xi: np.ndarray, tolerance: float = 1e-8) -> bool:
    """Whether the parametric coordinates ``xi`` lie inside of the parent domain (with a tolerance)."""
    if shape == "tetra4":
        return bool(np.min(xi) >= -tolerance and np.sum(xi) <= 1.0 + tolerance)
    return bool(np.max(np.abs(xi)) <= 1.0 + tolerance)


def parentDomainCenter(shape: str) -> np.ndarray:
    """The parametric coordinates of the center of the parent domain."""
    return NODE_PARAMETRIC_COORDINATES[shape].mean(axis=0)


def inverseMapping(shape: str, nodeCoordinates: np.ndarray, x: np.ndarray, maxIterations: int = 50) -> np.ndarray:
    """The parametric coordinates of a point ``x`` in an element (inverse isoparametric mapping, Newton).

    Parameters
    ----------
    shape
        The Ensight name of the shape; its parametric dimension must equal the spatial dimension.
    nodeCoordinates
        The nodal coordinates ``(nNodes, nDim)``.
    x
        The point.
    maxIterations
        The maximum number of Newton iterations.

    Returns
    -------
    np.ndarray
        The parametric coordinates, or ``None`` if the iteration did not converge (e.g., for a point far outside of a
        distorted element).
    """
    xi = parentDomainCenter(shape)
    scale = np.max(np.ptp(nodeCoordinates, axis=0))
    # Coordinates relative to the element's centroid: with absolute coordinates far from the origin (e.g. ~5e3 mm)
    # the round-off of N @ X is larger than any absolute tolerance on dXi, and the iteration never converges.
    x0 = nodeCoordinates.mean(axis=0)
    X = nodeCoordinates - x0
    xRel = np.asarray(x) - x0
    for _ in range(maxIterations):
        N, dN = shapeFunctionsAndDerivatives(shape, xi)
        residual = N @ X - xRel
        J = X.T @ dN
        try:
            dXi = -np.linalg.solve(J, residual)
        except np.linalg.LinAlgError:
            return None
        xi = xi + dXi
        if np.linalg.norm(dXi) < 1e-11 and np.linalg.norm(residual) < 1e-10 * scale:
            return xi
        if np.max(np.abs(xi)) > 1e3:
            return None
    return None
