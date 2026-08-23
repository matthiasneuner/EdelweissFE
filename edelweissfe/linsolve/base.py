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

"""The common base class every registered ``linsolver`` inherits, so the nonlinear solver can treat
all of them uniformly instead of special-casing per capability.

Every ``linsolve/<name>/__init__.py``'s ``createSolver(opts)`` factory returns an instance of a
:class:`LinearSolver` subclass, callable as ``(A, b) -> x``. :meth:`LinearSolver.setJournal` and
:meth:`LinearSolver.setModel` are part of that base contract with safe defaults, so a caller can call
either on *any* registered solver unconditionally, whether or not that particular solver actually uses
the information -- a solver that needs more than the default overrides it; everything else inherits a
default that does nothing harmful.

§27 collapsed what used to be a growing pile of individual, per-capability setters
(``setFieldStructure``, then ``requestedP1FieldNames``/``setP1Maps`` for §22's p-multigrid, then
``requestedNodeCoordinateFieldNames``/``setNodeCoordinates`` for §26's rigid-body near null-space) into
one: :meth:`setModel`, which simply hands a solver the live :class:`~edelweissfe.models.femodel.FEModel`
and :class:`~edelweissfe.numerics.dofmanager.DofManager` it is solving for. Every one of those
capabilities turned out to be derivable from those two objects alone (field layout from the
``DofManager``, node coordinates and element topology from the ``FEModel``), so a solver that wants
more than the base class's default field-structure bookkeeping (e.g. ``blockamg`` building a P1
topology map or reading node coordinates) can simply keep the references and compute whatever it needs
lazily, on its own schedule -- the driver no longer has to know in advance what any given solver might
want, query it, and push the answer back before every solve. :meth:`setFieldStructure` remains as a
lower-level escape hatch for callers that know the field-block layout directly but have no full model
to hand over (e.g. an offline probe script replaying a captured system).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class FieldBlock:
    """One physical field's contiguous block in the DOF vector.

    Attributes
    ----------
    name
        The field name, e.g. ``"displacement"`` or ``"nonlocal damage"``.
    start, stop
        The half-open DOF range ``[start, stop)`` of the field (fields are contiguous and field-major).
    dimension
        The nodal dimension of the field: the number of components per node (e.g. ``3`` for a 3D
        displacement, ``1`` for a scalar damage field). Determines the field's near null-space -- a
        vector field's rigid-body translations, a scalar field's constant.
    """

    name: str
    start: int
    stop: int
    dimension: int


class LinearSolver:
    """Common base for every ``linsolver`` registry entry. Callable as ``(A, b) -> x``.

    Subclasses implement :meth:`__call__`. :meth:`setJournal`, :meth:`setModel` and
    :meth:`setFieldStructure` have safe defaults here so the nonlinear solver can call any of them on
    any solver without asking first which ones care.
    """

    _journal = None
    _fieldStructure: "list[FieldBlock] | None" = None
    _model = None
    _dofManager = None

    def setJournal(self, journal) -> None:
        """Receive the shared :class:`~edelweissfe.journal.journal.Journal` instance.

        Default just stores it on ``self._journal`` for solvers that want to log through it; solvers
        with no logging needs simply never read the attribute.
        """
        self._journal = journal

    def setModel(self, model, dofManager) -> None:
        """Receive the live model and DOF manager this solver is being asked to solve for (§27).

        Called by the nonlinear solver whenever the equation system is (re)built -- i.e. on the first
        solve and again after any AMR/connectivity change, exactly the points where
        :meth:`setFieldStructure` used to be called directly. The default implementation derives and
        stores the per-field DOF-block structure (name, DOF range, nodal dimension) any field-split
        solver needs -- equivalent to calling :meth:`setFieldStructure` with those blocks -- and keeps
        ``model``/``dofManager`` themselves on ``self._model``/``self._dofManager`` for a solver that
        wants to go further (e.g. ``blockamg`` reading node coordinates for a rigid-body near
        null-space, or a P1 topology map for p-multigrid) without the driver having to know about it.

        A solver overriding this should normally call ``super().setModel(model, dofManager)`` first to
        get the field-structure bookkeeping for free, then do whatever else it needs.

        Parameters
        ----------
        model
            The live :class:`~edelweissfe.models.femodel.FEModel`.
        dofManager
            The live :class:`~edelweissfe.numerics.dofmanager.DofManager` describing the current
            equation system's layout.
        """
        self._model = model
        self._dofManager = dofManager
        self._fieldStructure = [
            FieldBlock(fieldName, fieldIndices.start, fieldIndices.stop, model.nodeFields[fieldName].dimension)
            for fieldName, fieldIndices in dofManager.idcsOfFieldsInDofVector.items()
        ]

    def setFieldStructure(self, fields: "list[FieldBlock]") -> None:
        """Receive the ordered field blocks of the DOF vector directly (in DOF order), bypassing
        :meth:`setModel`.

        An escape hatch for a caller that knows the field-block layout but has no full
        ``FEModel``/``DofManager`` to hand over -- e.g. an offline probe script driving a solver
        directly on a captured ``(A, b)`` system. A live run goes through :meth:`setModel` instead,
        whose default implementation computes exactly this and stores it the same way; a solver that
        only ever needs the field-block structure (not the model/dofManager references themselves)
        does not need to care which path supplied it.

        Default no-op -- only field-split solvers (e.g. ``blockamg``) need this; every other solver
        simply ignores the call.
        """
        self._fieldStructure = list(fields)

    def __call__(self, A, b):
        raise NotImplementedError
