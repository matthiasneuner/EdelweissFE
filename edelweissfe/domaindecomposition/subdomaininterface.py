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
"""The degrees of freedom a subdomain integrates, the interface it shares with the others, and the
two exchanges across that interface.

Every process integrates the degrees of freedom its elements and constraints touch -- its
*subdomain degrees of freedom*. A degree of freedom touched by several processes lies on their
*interface*: each of them holds only its own elements' contributions to the nodal force there.
:class:`InterfaceForceAssembly` completes it, once per increment, by exchanging with each
neighbouring process the individual element contributions at the degrees of freedom they share,
and summing all contributions at each in the order of the elements in the model -- the order in
which a single process computing the whole model sums them. The force at an interface degree of
freedom is therefore the same bits on every process integrating it, and the same bits as without
decomposition: the result does not depend on the number of processes, nor on the partition.

Each degree of freedom is also *owned* by exactly one process, the lowest-ranked one integrating
it; a degree of freedom no process touches is owned (and integrated) by rank 0. Ownership decides
which process counts a quantity that must be counted once -- a concentrated load, a kinetic energy
-- and which process' value is authoritative when the whole vector is gathered everywhere, by
:meth:`SubdomainInterface.gatherFromOwners`.
"""

import numpy as np
from mpi4py import MPI

from edelweissfe.numerics.assembly import addNodalForces

_INTERFACE_TAG = 7


class SubdomainInterface:
    """The subdomain degrees of freedom of this process, their ownership, and the exchange plans.

    Constructed collectively: every process of the communicator must construct its instance at the
    same time.

    Parameters
    ----------
    communicator
        The communicator of the processes sharing the model.
    touchedDofs
        The degrees of freedom this process' elements and constraints touch, in any order and with
        repetitions.
    nDof
        The size of the equation system.
    """

    def __init__(self, communicator, touchedDofs: np.ndarray, nDof: int):
        self.communicator = communicator
        rank = communicator.Get_rank()
        size = communicator.Get_size()
        self.nDof = nDof

        localDofs = np.unique(np.asarray(touchedDofs, dtype=np.int64))

        counts = np.array(communicator.allgather(localDofs.shape[0]), dtype=np.int64)
        allDofs = np.empty(counts.sum(), dtype=np.int64)
        communicator.Allgatherv(localDofs, [allDofs, counts, _displacements(counts), MPI.INT64_T])
        allRanks = np.repeat(np.arange(size, dtype=np.int64), counts)

        # A degree of freedom no process touches has no force but its own and is integrated by rank
        # 0, so that the equation of every degree of freedom is integrated somewhere.
        touched = np.zeros(nDof, dtype=bool)
        touched[allDofs] = True
        untouched = np.flatnonzero(~touched)
        if untouched.size:
            allDofs = np.concatenate([allDofs, untouched])
            allRanks = np.concatenate([allRanks, np.zeros(untouched.shape[0], dtype=np.int64)])
            if rank == 0:
                localDofs = np.union1d(localDofs, untouched)

        #: The sorted degrees of freedom this process integrates.
        self.subdomainDofs = localDofs

        order = np.lexsort((allRanks, allDofs))
        allDofs = allDofs[order]
        allRanks = allRanks[order]

        firstOfDof = np.ones(allDofs.shape[0], dtype=bool)
        firstOfDof[1:] = allDofs[1:] != allDofs[:-1]
        owner = np.empty(nDof, dtype=np.int64)
        owner[allDofs[firstOfDof]] = allRanks[firstOfDof]

        #: Whether this process owns each degree of freedom, over the whole equation system.
        self.ownedDofMask = owner == rank
        #: The owner of every degree of freedom.
        self.dofOwners = owner

        # The whole-vector gather: each process sends its owned entries, in ascending DOF order.
        self._ownedDofs = np.flatnonzero(self.ownedDofMask)
        self._ownedCounts = np.bincount(owner, minlength=size).astype(np.int64)
        self._gatheredOrder = np.argsort(owner, kind="stable")
        self._gatherBuffer = np.empty(nDof)

        # The interface: every (degree of freedom, other process) pair among this process' own.
        isLocal = np.zeros(nDof, dtype=bool)
        isLocal[localDofs] = True
        shared = isLocal[allDofs] & (allRanks != rank)
        sharedDofs = allDofs[shared]
        sharedRanks = allRanks[shared]

        #: The degrees of freedom this process shares with at least one other, sorted.
        self.interfaceDofs = np.unique(sharedDofs)
        #: The neighbouring processes, ascending.
        self.neighbours = np.unique(sharedRanks).tolist()

        #: The degrees of freedom shared with each neighbour, sorted (allDofs is sorted by DOF first).
        self._sharedWith = {neighbour: sharedDofs[sharedRanks == neighbour] for neighbour in self.neighbours}

    @property
    def nInterfaceDofs(self) -> int:
        """The number of degrees of freedom this process shares with another."""

        return self.interfaceDofs.shape[0]

    def sharedDofsWith(self, neighbour: int) -> np.ndarray:
        """The degrees of freedom this process shares with a neighbour, sorted.

        Parameters
        ----------
        neighbour
            The rank of a neighbouring process.

        Returns
        -------
        np.ndarray
            The shared degrees of freedom.
        """

        return self._sharedWith[neighbour]

    def gatherFromOwners(self, vector: np.ndarray):
        """Overwrite every entry of a vector with its owner's value, in place, so that it is the same,
        complete vector on every process.

        Collective.

        Parameters
        ----------
        vector
            A vector over the whole equation system, correct at least at the owned entries.
        """

        plain = vector.view(np.ndarray)
        self.communicator.Allgatherv(
            np.ascontiguousarray(plain[self._ownedDofs]),
            [self._gatherBuffer, self._ownedCounts, _displacements(self._ownedCounts), MPI.DOUBLE],
        )
        plain[self._gatheredOrder] = self._gatherBuffer


class _TaggedContributions:
    """The exchange of individual nodal contributions with the neighbouring processes, each tagged
    with its degree of freedom and its place in the order in which a single process computing the
    whole model adds it; what the assemblies at the interface have in common.

    Each process holds its contributions in one buffer, an entry per contribution. It sends each
    neighbour the entries at the degrees of freedom the two share -- the tags once, when this
    exchange is built, the values every increment -- and receives the neighbour's in turn. The
    *merged* buffer holds the selected own entries first, then those received from each neighbour,
    in neighbour order; its entries are summed per degree of freedom in the order of their tags.
    A pair of neighbours with nothing to exchange in one direction sends no message that way.

    Constructed collectively among neighbours.

    Parameters
    ----------
    interface
        The subdomain interface.
    entryDofs
        The degree of freedom of every entry of this process' contribution buffer.
    entryOrder
        The place of every entry in the order a single process adds the contributions in.
    ownEntries
        The entries of the buffer that are summed here, as indices into it.
    """

    def __init__(self, interface: SubdomainInterface, entryDofs, entryOrder, ownEntries: np.ndarray):
        self.interface = interface
        communicator = interface.communicator
        entryDofs = np.asarray(entryDofs, dtype=np.int64)
        entryOrder = np.asarray(entryOrder, dtype=np.int64)
        interfaceEntries = self.entriesAtInterface(interface, entryDofs)

        #: The entries of the contribution buffer summed here, in buffer order.
        self.ownEntries = ownEntries
        #: Whether this process takes part in an exchange at all.
        self.hasNeighbours = bool(interface.neighbours)

        sendEntries = {
            neighbour: interfaceEntries[np.isin(entryDofs[interfaceEntries], interface.sharedDofsWith(neighbour))]
            for neighbour in interface.neighbours
        }

        # The tags of what each neighbour will send: exchanged once, here.
        sendCounts = {
            neighbour: np.array([entries.shape[0]], dtype=np.int64) for neighbour, entries in sendEntries.items()
        }
        receiveCounts = {neighbour: np.empty(1, dtype=np.int64) for neighbour in interface.neighbours}
        _exchange(communicator, sendCounts, receiveCounts)

        self._sendEntries = {neighbour: entries for neighbour, entries in sendEntries.items() if entries.shape[0]}
        receiveCounts = {neighbour: int(count[0]) for neighbour, count in receiveCounts.items() if count[0]}

        sendTags = {
            neighbour: np.ascontiguousarray(np.stack([entryDofs[entries], entryOrder[entries]], axis=1).ravel())
            for neighbour, entries in self._sendEntries.items()
        }
        receiveTags = {neighbour: np.empty(2 * count, dtype=np.int64) for neighbour, count in receiveCounts.items()}
        _exchange(communicator, sendTags, receiveTags)

        tagDofs = [entryDofs[ownEntries]]
        tagOrder = [entryOrder[ownEntries]]
        begin = ownEntries.shape[0]
        self._receiveSlices = {}
        for neighbour in receiveCounts:
            tags = receiveTags[neighbour].reshape(-1, 2)
            tagDofs.append(tags[:, 0])
            tagOrder.append(tags[:, 1])
            self._receiveSlices[neighbour] = slice(begin, begin + tags.shape[0])
            begin += tags.shape[0]

        #: The degree of freedom and the order tag of every entry of the merged buffer.
        self.tagDofs = np.concatenate(tagDofs)
        self.tagOrder = np.concatenate(tagOrder)
        #: The merged buffer: the own entries, then what each neighbour sends.
        self.merged = np.empty(begin)
        #: The merged entries sorted by degree of freedom, and at each in the order they are added.
        self.order = np.lexsort((self.tagOrder, self.tagDofs))
        self._sendBuffers = {neighbour: np.empty(entries.shape[0]) for neighbour, entries in self._sendEntries.items()}

        sortedDofs, sortedOrder = self.tagDofs[self.order], self.tagOrder[self.order]
        if np.any((sortedDofs[1:] == sortedDofs[:-1]) & (sortedOrder[1:] == sortedOrder[:-1])):
            raise RuntimeError("A nodal contribution at the subdomain interface was received twice.")

    @staticmethod
    def entriesAtInterface(interface: SubdomainInterface, entryDofs: np.ndarray) -> np.ndarray:
        """The entries of a contribution buffer at a degree of freedom another process integrates too.

        Parameters
        ----------
        interface
            The subdomain interface.
        entryDofs
            The degree of freedom of every entry.

        Returns
        -------
        np.ndarray
            The entries, as indices into the buffer, ascending.
        """

        isInterface = np.zeros(interface.nDof, dtype=bool)
        isInterface[interface.interfaceDofs] = True
        return np.flatnonzero(isInterface[np.asarray(entryDofs, dtype=np.int64)])

    def exchange(self, contributions: np.ndarray):
        """Fill the merged buffer: this process' own entries, and the neighbours' at the degrees of
        freedom shared with them. Collective among neighbours.

        Parameters
        ----------
        contributions
            This process' contribution buffer.
        """

        communicator = self.interface.communicator
        requests = [
            communicator.Irecv(self.merged[receiveSlice], source=neighbour, tag=_INTERFACE_TAG)
            for neighbour, receiveSlice in self._receiveSlices.items()
        ]
        for neighbour, entries in self._sendEntries.items():
            np.take(contributions, entries, out=self._sendBuffers[neighbour])
            requests.append(communicator.Isend(self._sendBuffers[neighbour], dest=neighbour, tag=_INTERFACE_TAG))

        np.take(contributions, self.ownEntries, out=self.merged[: self.ownEntries.shape[0]])
        MPI.Request.Waitall(requests)


class InterfaceForceAssembly:
    """The element contributions to the interface degrees of freedom of one subdomain, and how to
    sum them in the order of the elements in the model.

    Each process evaluates its own elements into one buffer of element contributions -- an entry
    per element and degree of freedom, element after element. At a degree of freedom interior to
    its subdomain those are all the contributions there are, and summing them in buffer order is
    summing them in model order. At an interface degree of freedom the neighbours' contributions
    are missing: each neighbour sends the entries of its elements at the degrees of freedom the two
    share, tagged once, when this assembly is built, with the degree of freedom and the position of
    the element in the model. Every increment the received entries and the own ones are merged in
    that order and summed, from zero, as the element forces are summed without decomposition.

    Constructed collectively among neighbours, whenever the elements of a subdomain or the degree-of-
    freedom layout change.

    Parameters
    ----------
    interface
        The subdomain interface.
    entryDofs
        The degree of freedom of every entry of this process' contribution buffer.
    entryElementPositions
        The position in the model of the element of every entry.
    """

    def __init__(self, interface: SubdomainInterface, entryDofs: np.ndarray, entryElementPositions: np.ndarray):
        self._contributions = _TaggedContributions(
            interface,
            entryDofs,
            entryElementPositions,
            _TaggedContributions.entriesAtInterface(interface, entryDofs),
        )
        self._targets = np.searchsorted(interface.interfaceDofs, self._contributions.tagDofs[self._contributions.order])

    def assemble(self, contributions: np.ndarray, vector: np.ndarray):
        """Overwrite the interface entries of ``vector`` with the complete sums of all element
        contributions, in the order of the elements in the model. Collective among neighbours.

        Parameters
        ----------
        contributions
            This process' contribution buffer, one entry per element and degree of freedom.
        vector
            The vector to complete; only the interface entries are written.
        """

        exchange = self._contributions
        if not exchange.hasNeighbours:
            return

        exchange.exchange(contributions)
        interfaceDofs = exchange.interface.interfaceDofs
        vector.view(np.ndarray)[interfaceDofs] = np.bincount(
            self._targets, weights=exchange.merged[exchange.order], minlength=interfaceDofs.shape[0]
        )


class ConstraintForceExchange:
    """How the processes share the forces of the constraints they evaluate, every increment.

    A constraint is evaluated whole, by the process that owns it, and its forces are added by every
    process, in model order -- the order a single process adds them in. Which degrees of freedom each
    constraint acts on changes only when the equation system is rebuilt, so the layout -- every
    constraint's owner, its place in its owner's buffer, its degrees of freedom -- is exchanged once
    per rebuild, here; an increment exchanges the forces alone, in one ``Allgatherv``.

    Constructed collectively.

    Parameters
    ----------
    communicator
        The communicator of the processes sharing the model.
    constraints
        The constraints of the model, by name, in model order.
    ownedConstraints
        Those this process evaluates, by name.
    constraintDofs
        The degrees of freedom of every owned constraint, by constraint.
    """

    def __init__(self, communicator, constraints: dict, ownedConstraints: dict, constraintDofs: dict):
        self.communicator = communicator
        rank = communicator.Get_rank()

        owned = [
            (name, np.asarray(constraintDofs[constraint], dtype=np.int64))
            for name, constraint in constraints.items()
            if name in ownedConstraints
        ]
        self._ownedNames = [name for name, _ in owned]

        self._layout = {}
        counts = []
        for owner, entries in enumerate(communicator.allgather(owned)):
            offset = 0
            for name, dofs in entries:
                # whether a DOF is named more than once; see addNodalForces
                self._layout[name] = (owner, offset, dofs, np.unique(dofs).shape[0] != dofs.shape[0])
                offset += dofs.shape[0]
            counts.append(offset)

        self._order = list(constraints)
        self._counts = np.array(counts, dtype=np.int64)
        self._displacements = _displacements(self._counts)
        self._sendBuffer = np.empty(self._counts[rank])
        self._receiveBuffer = np.empty(self._counts.sum())

    def addAllConstraintForces(self, ownedForces: dict, vector: np.ndarray):
        """Share this process' constraint forces and add those of every constraint, in model order.

        Collective.

        Parameters
        ----------
        ownedForces
            The forces of every owned constraint, by name.
        vector
            The net nodal force vector to add into.
        """

        begin = 0
        for name in self._ownedNames:
            forces = ownedForces[name]
            self._sendBuffer[begin : begin + forces.shape[0]] = forces
            begin += forces.shape[0]

        self.communicator.Allgatherv(
            self._sendBuffer, [self._receiveBuffer, self._counts, self._displacements, MPI.DOUBLE]
        )

        plain = vector.view(np.ndarray)
        for name in self._order:
            owner, offset, dofs, namesDofMoreThanOnce = self._layout[name]
            begin = self._displacements[owner] + offset
            addNodalForces(plain, dofs, self._receiveBuffer[begin : begin + dofs.shape[0]], namesDofMoreThanOnce)


def _exchange(communicator, sendBuffers: dict, receiveBuffers: dict):
    """Exchange one buffer with each neighbour, both ways. Collective among neighbours."""

    requests = [
        communicator.Irecv(receiveBuffers[neighbour], source=neighbour, tag=_INTERFACE_TAG)
        for neighbour in receiveBuffers
    ]
    requests += [
        communicator.Isend(sendBuffers[neighbour], dest=neighbour, tag=_INTERFACE_TAG) for neighbour in sendBuffers
    ]
    MPI.Request.Waitall(requests)


def _displacements(counts: np.ndarray) -> np.ndarray:
    """The receive displacements of a variable-length gather with these counts."""

    displacements = np.zeros_like(counts)
    np.cumsum(counts[:-1], out=displacements[1:])
    return displacements
