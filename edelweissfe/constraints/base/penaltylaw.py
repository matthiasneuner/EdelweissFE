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
The normal penalty laws of the penalty contact constraints: how the normal force at a contact point follows from its gap.

A contact constraint is made of three pieces: its *contact points* (where the slave surface is, see
:mod:`~edelweissfe.constraints.base.contactpoints`), the *geometry* that pairs each of them with the master surface and
measures its gap, and the *contact law* that turns the gap into a force. The law is a separate object, held by the
constraint and asked for the force point by point, so that another law -- friction on top of the normal law, or a tie
that also pulls (:class:`TiedPenaltyLaw`) -- takes its place without touching the contact points or the geometry.

Sign convention: the gap :math:`g` is negative when the surfaces penetrate, and the normal force
:math:`f_n` carries the sign of :math:`g`, i.e. :math:`f_n \\le 0` in contact. The stiffness
:math:`\\mathrm{d}f_n/\\mathrm{d}g` is then positive for :math:`g < 0`.

``penaltyTimesArea`` is the penalty modulus times the area the contact point represents: a nodal
tributary area in node-based formulations, a quadrature weight in integrated ones.
"""

from abc import ABC, abstractmethod

import numpy as np


class PenaltyLaw(ABC):
    """A unilateral normal penalty law: a contact point transmits a force only while its gap is closed (negative).

    Every method works element-wise on arrays of points as well as on single points.
    """

    #: The name of the law, as the input file spells it (the ``type`` option of the contact constraints).
    name: str

    #: Whether the law ties the contact points to the master surface (:class:`TiedPenaltyLaw`) instead of letting
    #: them touch it: then a constraint pairs each point once, holds it in every direction, and never releases it.
    tiesThePoints: bool = False

    def isClosed(self, gap: float | np.ndarray) -> bool | np.ndarray:
        """Whether the contact points with these gaps transmit a force: while they penetrate.

        Parameters
        ----------
        gap
            The gap(s), negative in contact.

        Returns
        -------
        bool or numpy.ndarray
            True where the gap is closed.
        """

        return gap < 0.0

    @abstractmethod
    def normalForce(
        self, penaltyTimesArea: float | np.ndarray, gap: float | np.ndarray
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
        """The normal force at closed contact points and its derivative with respect to the gap.

        Parameters
        ----------
        penaltyTimesArea
            The penalty modulus times the area of the contact point(s).
        gap
            The gap(s), negative in contact.

        Returns
        -------
        tuple
            The normal force(s) :math:`f_n` and the stiffness :math:`\\mathrm{d}f_n/\\mathrm{d}g`.
        """

    @abstractmethod
    def storedEnergy(self, penaltyTimesArea: float | np.ndarray, gap: float | np.ndarray) -> float | np.ndarray:
        """The energy stored in the penalty springs of closed contact points, :math:`\\Pi(g)` with
        :math:`\\mathrm{d}\\Pi/\\mathrm{d}g = f_n`: the potential whose gradient is the contact force.

        Parameters
        ----------
        penaltyTimesArea
            The penalty modulus times the area of the contact point(s).
        gap
            The gap(s), negative in contact.

        Returns
        -------
        float or numpy.ndarray
            The stored energy, non-negative.
        """


class LinearPenaltyLaw(PenaltyLaw):
    """:math:`f_n = k A \\, g`: a linear spring with the constant stiffness :math:`k A`, which jumps from zero when the
    gap closes."""

    name = "linear"

    def normalForce(self, penaltyTimesArea, gap):
        return penaltyTimesArea * gap, penaltyTimesArea

    def storedEnergy(self, penaltyTimesArea, gap):
        return 0.5 * penaltyTimesArea * gap**2


class QuadraticPenaltyLaw(PenaltyLaw):
    """:math:`f_n = -\\tfrac{1}{2} k A \\, g^2`: a spring that stiffens with the penetration, its stiffness
    :math:`-k A \\, g` rising continuously from zero when the gap closes."""

    name = "quadratic"

    def normalForce(self, penaltyTimesArea, gap):
        return -0.5 * penaltyTimesArea * gap**2, -penaltyTimesArea * gap

    def storedEnergy(self, penaltyTimesArea, gap):
        return -penaltyTimesArea * gap**3 / 6.0


class TiedPenaltyLaw(PenaltyLaw):
    """A tie: :math:`f = k A \\, g` in tension and in compression, a linear spring that never opens.

    A constraint with this law does not let its contact points touch the master surface, it ties them to it: it pairs
    each point once with the point of the master it is closest to, and holds it there in *every* direction, by one such
    spring along each axis, whose "gap" is the displacement of the point from where it was paired, in that direction.
    Together the springs pull the point back along its whole displacement with the stiffness :math:`k A`, whichever
    way it moves: a linear spring of the same stiffness in every direction. A gap or an overlap the two surfaces have at
    the pairing is where the springs rest.

    It is a contact that never opens and never slides, the stick of a friction law that never slips.
    """

    name = "tied"

    tiesThePoints = True

    def isClosed(self, gap):
        return np.full(np.shape(gap), True)

    def normalForce(self, penaltyTimesArea, gap):
        return penaltyTimesArea * gap, penaltyTimesArea

    def storedEnergy(self, penaltyTimesArea, gap):
        return 0.5 * penaltyTimesArea * gap**2


#: The penalty laws of contact, by the name the input file gives them.
penaltyLaws = {law.name: law for law in (LinearPenaltyLaw, QuadraticPenaltyLaw)}

#: The penalty laws of a constraint that can also tie its points to the master surface
#: (:mod:`~edelweissfe.constraints.surfacetodeformablesurfacepenalty`): those of contact, and the tie.
penaltyLawsAndTheTie = penaltyLaws | {TiedPenaltyLaw.name: TiedPenaltyLaw}


def penaltyLaw(contactType: str, laws: dict = penaltyLaws) -> PenaltyLaw:
    """The penalty law an input file names, case-insensitively.

    Parameters
    ----------
    contactType
        The ``type`` option of a contact constraint: 'linear' or 'quadratic', or 'tied' where a constraint can tie.
    laws
        The laws the constraint accepts, by name: :data:`penaltyLaws`, or :data:`penaltyLawsAndTheTie`.

    Returns
    -------
    PenaltyLaw
        The law.
    """

    law = laws.get(contactType.lower())
    if law is None:
        raise ValueError(
            f"Constraint type '{contactType.lower()}' is not supported. Use "
            + " or ".join(f"'{name}'" for name in laws)
            + "."
        )
    return law()
