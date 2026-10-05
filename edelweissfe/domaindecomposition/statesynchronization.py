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
"""Bringing the states of the elements and constraints a process does not compute up to date.

A process computes, and so keeps current, only the states of its own elements and constraints; the
states of all others in its copy of the model are those of the last synchronization. Whatever reads
the whole model -- the field outputs, the output managers, a marker deciding a refinement, the
state transfer carrying the history of a refined element onto its children -- therefore needs a
synchronization first: every process sends the states it owns to every other.

It uses the interface the restart checkpoints use (``getStateVars``/``setStateVars`` on elements,
``getRestartData``/``setRestartData`` on constraints): whatever a checkpoint must carry to resume a
run is exactly what another process must receive to continue it.

Restoring a constraint's restart data restores the constraint completely: a contact constraint
adopts the owner's frozen contact search, and with it the degrees of freedom it couples (see
:mod:`~edelweissfe.constraints.base.frozencontactsearch`). After a synchronization every copy of a
constraint is therefore its owner's, so a checkpoint written from any process' copy of the model is
the owners' state, and every process resumes from the same model. A copy's DOF footprint can change
here without its process' degree-of-freedom indices following until the next rebuild; nothing reads
them there, since a constraint is evaluated, and its forces exchanged, only with its owner's
indices (:class:`~edelweissfe.domaindecomposition.subdomaininterface.ConstraintForceExchange`).
"""

import numpy as np
from mpi4py import MPI


class ModelStateSynchronization:
    """The layout of one exchange of element and constraint states, for one partition of a model.

    Constructed collectively, and valid until the model's elements or their partition change.

    Parameters
    ----------
    communicator
        The communicator of the processes sharing the model.
    elements
        The elements of the model, by number.
    elementOwners
        The rank of every element, by element number.
    constraints
        The constraints of the model, by name.
    constraintOwners
        The rank of every constraint, by name.
    """

    def __init__(self, communicator, elements: dict, elementOwners: dict, constraints: dict, constraintOwners: dict):
        self.communicator = communicator
        rank = communicator.Get_rank()
        size = communicator.Get_size()
        self._rank = rank

        #: Elements that expose no state through getStateVars; their state is not synchronized.
        self.elementsWithoutState = []

        elementsOfRank = [[] for _ in range(size)]
        stateSizesOfRank = [[] for _ in range(size)]
        for number, element in elements.items():
            try:
                stateSize = element.getStateVars().shape[0]
            except NotImplementedError:
                self.elementsWithoutState.append(number)
                continue
            if stateSize:
                owner = elementOwners[number]
                elementsOfRank[owner].append(element)
                stateSizesOfRank[owner].append(stateSize)

        self._ownedElements = elementsOfRank[rank]
        self._elementsOfRank = elementsOfRank
        self._stateOffsetsOfRank = [
            np.concatenate([[0], np.cumsum(sizes, dtype=np.int64)]) for sizes in stateSizesOfRank
        ]
        self._counts = np.array([offsets[-1] for offsets in self._stateOffsetsOfRank], dtype=np.int64)
        self._displacements = np.concatenate([[0], np.cumsum(self._counts[:-1])]).astype(np.int64)
        self._receiveBuffer = np.empty(self._counts.sum())

        self._constraints = constraints
        self._ownedConstraints = [name for name in constraints if constraintOwners[name] == rank]

    @property
    def nStateVariables(self) -> int:
        """The number of state variables one synchronization exchanges, over all processes."""

        return int(self._counts.sum())

    def synchronizeElementStates(self):
        """Give every element the state of its owner. Collective."""

        if self._ownedElements:
            send = np.concatenate([element.getStateVars() for element in self._ownedElements])
        else:
            send = np.empty(0)

        self.communicator.Allgatherv(send, [self._receiveBuffer, self._counts, self._displacements, MPI.DOUBLE])

        for owner, ownersElements in enumerate(self._elementsOfRank):
            if owner == self._rank or not ownersElements:
                continue
            received = self._receiveBuffer[
                self._displacements[owner] : self._displacements[owner] + self._counts[owner]
            ]
            offsets = self._stateOffsetsOfRank[owner]
            for position, element in enumerate(ownersElements):
                element.setStateVars(received[offsets[position] : offsets[position + 1]].copy())

    def synchronizeConstraintStates(self):
        """Give every stateful constraint the state of its owner. Collective."""

        owned = {}
        for name in self._ownedConstraints:
            # A constraint carrying nothing between increments has nothing to send.
            data = self._constraints[name].getRestartData()
            if data:
                owned[name] = {entry: np.array(values) for entry, values in data.items()}

        for owner, received in enumerate(self.communicator.allgather(owned)):
            if owner == self._rank:
                continue
            for name, data in received.items():
                self._constraints[name].setRestartData(data)
