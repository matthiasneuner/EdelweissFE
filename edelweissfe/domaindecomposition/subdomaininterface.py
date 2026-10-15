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
exchanges across that interface.

Every process integrates the degrees of freedom its elements and constraints touch -- its
*subdomain degrees of freedom*. A degree of freedom touched by several processes lies on their
*interface*: each of them holds only its own elements' contributions to the nodal force there.
:class:`InterfaceForceAssembly` completes it, once per increment, by exchanging with each
neighbouring process the individual element contributions at the degrees of freedom they share,
and summing all contributions at each in the order of the elements in the model -- the order in
which a single process computing the whole model sums them. The force at an interface degree of
freedom is therefore the same bits on every process integrating it, and the same bits as without
decomposition: the result does not depend on the number of processes, nor on the partition.
The loads acting on elements (:class:`InterfaceLoadAssembly`) and the forces of the constraints
(:class:`ConstraintForceExchange`) travel the same way, to the processes integrating their degrees of
freedom only, and are added there in the order a single process adds them.

Each degree of freedom is also *owned* by exactly one process, the lowest-ranked one integrating
it; a degree of freedom no process touches is owned (and integrated) by rank 0. Ownership decides
which process counts a quantity that must be counted once -- a concentrated load, a kinetic energy
-- and which process' value is authoritative when the whole vector is gathered everywhere, by
:meth:`SubdomainInterface.allgatherOwnedValues`.
"""

from enum import IntEnum, unique

import numpy as np
from mpi4py import MPI
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from edelweissfe.numerics.assembly import addNodalForces
from edelweissfe.numerics.scatteradd import sumAtPositions


@unique
class MessageTag(IntEnum):
    """The message tag of every kind of point-to-point exchange, named in one place. Two kinds never
    share a tag, so that a message of one kind cannot be received as one of another between the same
    two processes -- as it could the day two exchanges overlapped; today each waits for all its
    messages before the next begins. An exchange sends its layout, when it is built, under its own
    tag as well. Two kinds of the same value are refused when this module is imported
    (:func:`enum.unique`)."""

    #: The element contributions at the interface (:class:`InterfaceForceAssembly`).
    ELEMENT_FORCES = 7
    #: The load contributions of the elements at the interface (:class:`InterfaceLoadAssembly`).
    LOADS = 8
    #: The forces of the constraints (:class:`ConstraintForceExchange`).
    CONSTRAINT_FORCES = 9
    #: Vector entries received from their owners (:class:`ValuesFromOwners`).
    VALUES_FROM_OWNERS = 10


#: The bound on the entries of one constraint: an entry of a constraint force is ordered by the
#: position of its constraint in the model times this, plus its index within the constraint.
_ENTRIES_PER_CONSTRAINT = 2**32


class ElementProcessesAtDofs:
    """Which processes' elements touch each degree of freedom of the model, derived in every process from
    data every process holds -- the element partition, the connectivity of the mesh and the global
    numbering of the degrees of freedom (:meth:`ofMesh`) -- instead of gathered from the processes.

    Most degrees of freedom are touched by the elements of one process: for each it holds the
    lowest-ranked process whose elements touch it. The few touched by the elements of several
    processes -- the element interface -- it lists with every one of them.

    Parameters
    ----------
    lowestProcess
        Per degree of freedom of the model, the lowest rank whose elements touch it; ``nProcesses``
        where no element does.
    sharedDofs
        The degrees of freedom touched by the elements of more than one process, one entry per such
        process, sorted by degree of freedom and then by process.
    sharedProcesses
        The process of each of those entries.
    nProcesses
        The number of processes.
    """

    def __init__(self, lowestProcess: np.ndarray, sharedDofs: np.ndarray, sharedProcesses: np.ndarray, nProcesses: int):
        self.lowestProcess = lowestProcess
        self.sharedDofs = sharedDofs
        self.sharedProcesses = sharedProcesses
        self.nProcesses = nProcesses

    @property
    def nDof(self) -> int:
        """The number of degrees of freedom of the model."""

        return self.lowestProcess.shape[0]

    @classmethod
    def ofElementDofs(cls, dofs: np.ndarray, processes: np.ndarray, nDof: int, nProcesses: int):
        """From the degrees of freedom every element touches, each entry with its element's process.

        Parameters
        ----------
        dofs
            A degree of freedom an element touches, per entry, with repetitions.
        processes
            The process of the element of each entry.
        nDof
            The number of degrees of freedom of the model.
        nProcesses
            The number of processes.

        Returns
        -------
        ElementProcessesAtDofs
            The processes at the degrees of freedom.
        """

        lowest, sharedDofs, sharedProcesses = _processesAtPlaces(dofs, processes, nDof, nProcesses)
        return cls(lowest, sharedDofs, sharedProcesses, nProcesses)

    @classmethod
    def ofMesh(cls, connectivity, numbering, processOfElements: np.ndarray, nProcesses: int):
        """From the connectivity of the mesh, the global numbering of its degrees of freedom and the process
        of every element. Node by node and field by field -- an element touches every component of
        each field at each of its nodes --, so that no array per element and degree of freedom is made.

        Parameters
        ----------
        connectivity
            The connectivity of the mesh (:meth:`~edelweissfe.models.mesh.Mesh.connectivity`).
        numbering
            The global numbering of the degrees of freedom
            (:class:`~edelweissfe.numerics.globaldofnumbering.GlobalDofNumbering`).
        processOfElements
            The process of every element of the mesh, in mesh order.
        nProcesses
            The number of processes.

        Returns
        -------
        ElementProcessesAtDofs
            The processes at the degrees of freedom.
        """

        lowest = np.full(numbering.nDof, nProcesses, dtype=np.int32)
        sharedDofs, sharedProcesses = [], []
        for field in numbering.fields:
            rows, processes = [], []
            for group in connectivity.groups:
                groupProcesses = processOfElements[group.positions]
                for i, fields in enumerate(group.typeInfo.fields):
                    if field in fields:
                        rows.append(group.nodeRows[:, i])
                        processes.append(groupProcesses)
            if not rows:
                continue
            nodeLowest, sharedNodes, sharedNodeProcesses = _processesAtPlaces(
                np.concatenate(rows), np.concatenate(processes), numbering.nodeLabels.shape[0], nProcesses
            )
            carrying = numbering.nodesCarryingField[field]
            first, dimension = numbering.firstDofOfField[field], numbering.dimensionOfField[field]
            if np.any(nodeLowest[~carrying] < nProcesses):
                raise RuntimeError("an element touches field {:} at a node not carrying it".format(field))
            lowest[first : first + np.count_nonzero(carrying) * dimension] = np.repeat(nodeLowest[carrying], dimension)
            components = np.arange(dimension, dtype=np.int64)
            dofs = first + numbering.rankInField[field][sharedNodes].astype(np.int64)[:, None] * dimension + components
            sharedDofs.append(dofs.ravel())
            sharedProcesses.append(np.repeat(sharedNodeProcesses, dimension))

        sharedDofs, sharedProcesses = _uniquePairs(
            np.concatenate(sharedDofs) if sharedDofs else np.empty(0, dtype=np.int64),
            np.concatenate(sharedProcesses) if sharedProcesses else np.empty(0, dtype=np.int64),
            nProcesses,
        )
        return cls(lowest, sharedDofs, sharedProcesses, nProcesses)

    def dofsOfProcess(self, process: int) -> np.ndarray:
        """The degrees of freedom the elements of a process touch.

        Parameters
        ----------
        process
            The rank.

        Returns
        -------
        np.ndarray
            The degrees of freedom, sorted.
        """

        return _sortedUnique(
            np.concatenate(
                [np.flatnonzero(self.lowestProcess == process), self.sharedDofs[self.sharedProcesses == process]]
            )
        )

    def processesAt(self, dofs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Every process whose elements touch one of the given degrees of freedom.

        Parameters
        ----------
        dofs
            Degrees of freedom, sorted and without repetitions.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Pairs of a degree of freedom and a process, as two arrays.
        """

        lowest = self.lowestProcess[dofs]
        touched = lowest < self.nProcesses
        positions = np.searchsorted(dofs, self.sharedDofs)
        positions[positions == dofs.shape[0]] = 0
        atDofs = dofs.shape[0] > 0
        shared = (dofs[positions] == self.sharedDofs) if atDofs else np.zeros(self.sharedDofs.shape[0], dtype=bool)
        return _uniquePairs(
            np.concatenate([dofs[touched], self.sharedDofs[shared]]),
            np.concatenate([lowest[touched].astype(np.int64), self.sharedProcesses[shared]]),
            self.nProcesses,
        )


def _processesAtPlaces(places: np.ndarray, processes: np.ndarray, nPlaces: int, nProcesses: int) -> tuple:
    """The lowest process at each place, and every process at the places with more than one.

    Parameters
    ----------
    places
        A place per entry, with repetitions.
    processes
        The process of each entry.
    nPlaces
        The number of places.
    nProcesses
        The number of processes, the lowest process of a place without entries.

    Returns
    -------
    tuple
        The lowest process at each place; the places with more than one process, one entry per process,
        sorted; the process of each of those entries.
    """

    places = np.asarray(places, dtype=np.int64)
    processes = np.asarray(processes, dtype=np.int64)
    lowest = np.full(nPlaces, nProcesses, dtype=np.int32)
    np.minimum.at(lowest, places, processes.astype(np.int32))
    highest = np.full(nPlaces, -1, dtype=np.int32)
    np.maximum.at(highest, places, processes.astype(np.int32))
    several = (highest > lowest)[places]
    sharedPlaces, sharedProcesses = _uniquePairs(places[several], processes[several], nProcesses)
    return lowest, sharedPlaces, sharedProcesses


def _sortedUnique(values: np.ndarray) -> np.ndarray:
    """The distinct values, sorted: :func:`numpy.unique` by a sort, many times faster for the large
    integer arrays of a rebuild than the hash table :func:`numpy.unique` uses (numpy 2.5).

    Parameters
    ----------
    values
        A one-dimensional array.

    Returns
    -------
    np.ndarray
        The distinct values, ascending.
    """

    values = np.sort(values)
    if values.shape[0] < 2:
        return values
    return values[np.concatenate(([True], values[1:] != values[:-1]))]


def _uniquePairs(places: np.ndarray, processes: np.ndarray, nProcesses: int) -> tuple[np.ndarray, np.ndarray]:
    """Pairs of a place and a process without repetitions, sorted by place and then by process."""

    keys = _sortedUnique(np.asarray(places, dtype=np.int64) * nProcesses + np.asarray(processes, dtype=np.int64))
    return keys // nProcesses, keys % nProcesses


class SubdomainInterface:
    """The subdomain degrees of freedom of this process, their ownership, and the exchange plans.

    Which processes integrate a degree of freedom is derived in every process from what every process
    holds: which processes' elements touch it (:class:`ElementProcessesAtDofs`, from the element
    partition and the mesh) and the multi-point constraints linking it to others. Only the degrees of
    freedom a process touches beyond its elements' -- those of the constraints it evaluates, which
    only it holds current -- are gathered to every process. No process gathers the subdomain degrees
    of freedom of the others.

    Constructed collectively: every process of the communicator must construct its instance at the
    same time.

    Parameters
    ----------
    communicator
        The communicator of the processes sharing the model.
    elementProcesses
        Which processes' elements touch each degree of freedom; the same in every process.
    touchedDofs
        The degrees of freedom this process' elements and constraints touch, in any order and with
        repetitions.
    multiPointConstraintLinks
        The slave and the master of every dependency of the multi-point constraints, the same in every
        process, or None.
    """

    def __init__(
        self,
        communicator,
        elementProcesses: "ElementProcessesAtDofs",
        touchedDofs: np.ndarray,
        multiPointConstraintLinks: tuple[np.ndarray, np.ndarray] | None = None,
    ):
        self.communicator = communicator
        rank = communicator.Get_rank()
        size = communicator.Get_size()
        nDof = elementProcesses.nDof
        self.nDof = nDof

        touched = _sortedUnique(np.asarray(touchedDofs, dtype=np.int64))

        # Which processes' elements touch a degree of freedom, every process derives itself. What it
        # cannot derive are the degrees of freedom a process touches beyond its elements' -- those of
        # its constraints --: each process sends those, and they are gathered to every process.
        byOwnElements = elementProcesses.dofsOfProcess(rank)
        derivedWrongly = not np.isin(byOwnElements, touched, kind="table").all()
        if communicator.allreduce(int(derivedWrongly), op=MPI.MAX):
            raise RuntimeError(
                "The degrees of freedom the elements of a process touch, as derived from the element partition "
                "and the mesh, include some its elements do not touch: the processes would disagree on which "
                "process integrates which degree of freedom."
            )
        claimed = touched[~np.isin(touched, byOwnElements, kind="table")]
        counts = np.array(communicator.allgather(claimed.shape[0]), dtype=np.int64)
        claimedDofs = np.empty(counts.sum(), dtype=np.int64)
        communicator.Allgatherv(claimed, [claimedDofs, counts, _displacements(counts), MPI.INT64_T])
        claimedProcesses = np.repeat(np.arange(size, dtype=np.int64), counts)
        #: The number of degrees of freedom the processes touch beyond their elements', gathered to every
        #: process.
        self.nClaimedDofs = claimedDofs.shape[0]

        def touchingProcessesAt(dofs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """Every process touching one of the given sorted degrees of freedom, with its elements or its
            constraints, as pairs of a degree of freedom and a process."""

            elementDofs, elementRanks = elementProcesses.processesAt(dofs)
            atDofs = np.isin(claimedDofs, dofs, kind="table")
            return _uniquePairs(
                np.concatenate([elementDofs, claimedDofs[atDofs]]),
                np.concatenate([elementRanks, claimedProcesses[atDofs]]),
                size,
            )

        # A process touching a degree of freedom of a group linked by multi-point constraints integrates
        # all of the group (a slave's force is folded onto its masters, its velocity interpolated from
        # theirs): every degree of freedom of a group, with every process touching the group.
        linkedDofs, groupOfLinked = multiPointConstraintGroups(multiPointConstraintLinks)
        pairDofs, pairRanks = touchingProcessesAt(linkedDofs)
        groupKeys = _sortedUnique(groupOfLinked[np.searchsorted(linkedDofs, pairDofs)] * size + pairRanks)
        closureDofs, closureRanks = _membersWithProcesses(
            linkedDofs, groupOfLinked, groupKeys // size, groupKeys % size
        )

        # The owner: the lowest-ranked process integrating a degree of freedom; rank 0 for one nothing
        # touches, which rank 0 integrates, so that every equation is integrated somewhere.
        owner = elementProcesses.lowestProcess.astype(np.int64)
        np.minimum.at(owner, claimedDofs, claimedProcesses)
        np.minimum.at(owner, closureDofs, closureRanks)
        untouched = owner == size
        owner[untouched] = 0

        subdomainDofs = [touched, closureDofs[closureRanks == rank]]
        if rank == 0:
            subdomainDofs.append(np.flatnonzero(untouched))
        subdomainDofs = _sortedUnique(np.concatenate(subdomainDofs))
        #: The sorted degrees of freedom this process integrates.
        self.subdomainDofs = subdomainDofs

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
        pairDofs, pairRanks = touchingProcessesAt(subdomainDofs)
        atSubdomain = np.isin(closureDofs, subdomainDofs, kind="table")
        sharedDofs, sharedRanks = _uniquePairs(
            np.concatenate([pairDofs, closureDofs[atSubdomain]]),
            np.concatenate([pairRanks, closureRanks[atSubdomain]]),
            size,
        )
        other = sharedRanks != rank
        sharedDofs, sharedRanks = sharedDofs[other], sharedRanks[other]

        #: The degrees of freedom this process shares with at least one other, sorted.
        self.interfaceDofs = _sortedUnique(sharedDofs)
        #: The neighbouring processes, ascending.
        self.neighbours = np.unique(sharedRanks).tolist()

        #: The degrees of freedom shared with each neighbour, sorted (the pairs are sorted by DOF first).
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

    def allgatherOwnedValues(self, vector: np.ndarray):
        """Collective. Overwrite every entry of a vector with its owner's value, in place, so that it is the
        same, complete vector on every process.

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

    def gatherOwnedValuesToRoot(self, vector: np.ndarray):
        """Collective. Overwrite every entry of a vector on rank 0 with its owner's value, in place, so that
        it is the complete vector there -- as :meth:`allgatherOwnedValues` does on every process. The
        other processes send their owned entries and keep their vector as it was: complete at the degrees
        of freedom they integrate only.

        Parameters
        ----------
        vector
            A vector over the whole equation system, correct at least at the owned entries.
        """

        plain = vector.view(np.ndarray)
        send = np.ascontiguousarray(plain[self._ownedDofs])
        if self.communicator.Get_rank() == 0:
            self.communicator.Gatherv(
                send, [self._gatherBuffer, self._ownedCounts, _displacements(self._ownedCounts), MPI.DOUBLE], root=0
            )
            plain[self._gatheredOrder] = self._gatherBuffer
        else:
            self.communicator.Gatherv(send, None, root=0)


def multiPointConstraintGroups(links: tuple[np.ndarray, np.ndarray] | None) -> tuple[np.ndarray, np.ndarray]:
    """The groups of degrees of freedom linked by multi-point constraints, transitively: a slave with its
    masters, and every group sharing a degree of freedom with another merged with it. Computed among the
    linked degrees of freedom only, numbered compactly: a graph of the size of the multi-point
    constraints, not of the equation system.

    Parameters
    ----------
    links
        The slave and the master of every dependency
        (:meth:`~edelweissfe.numerics.mpctransformation.MultiPointConstraintTransformation.slaveMasterDofPairs`),
        or None.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The linked degrees of freedom, sorted, and the group of each.
    """

    if links is None or not links[0].size:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    slaves, masters = links
    linked, positions = np.unique(np.concatenate([slaves, masters]).astype(np.int64), return_inverse=True)
    nLinks = slaves.shape[0]
    graph = coo_matrix((np.ones(nLinks), (positions[:nLinks], positions[nLinks:])), shape=(linked.shape[0],) * 2)
    _, group = connected_components(graph, directed=False)
    return linked, group.astype(np.int64)


def _membersWithProcesses(linkedDofs, groupOfLinked, groups, processes) -> tuple[np.ndarray, np.ndarray]:
    """Every degree of freedom of each given group, paired with the process given with the group.

    Parameters
    ----------
    linkedDofs
        The linked degrees of freedom.
    groupOfLinked
        The group of each.
    groups
        A group per pair.
    processes
        The process of each pair.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Pairs of a degree of freedom and a process.
    """

    byGroup = np.argsort(groupOfLinked, kind="stable")
    nGroups = int(groupOfLinked.max()) + 1 if groupOfLinked.size else 0
    first = np.searchsorted(groupOfLinked[byGroup], np.arange(nGroups + 1))
    nMembers = np.diff(first)[groups]
    starts = np.repeat(first[groups], nMembers)
    within = np.arange(int(nMembers.sum())) - np.repeat(np.cumsum(nMembers) - nMembers, nMembers)
    return linkedDofs[byGroup[starts + within]], np.repeat(processes, nMembers)


class _TaggedContributions:
    """The exchange of individual nodal contributions with the neighbouring processes, each tagged
    with its degree of freedom and its place in the order in which a single process computing the
    whole model adds it; what :class:`InterfaceForceAssembly`, :class:`InterfaceLoadAssembly` and
    :class:`ConstraintForceExchange` have in common.

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
    tag
        The message tag of this kind of exchange.
    """

    def __init__(self, interface: SubdomainInterface, entryDofs, entryOrder, ownEntries: np.ndarray, tag: MessageTag):
        self.interface = interface
        #: The message tag of every message of this exchange.
        self.tag = tag
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
        _exchange(communicator, sendCounts, receiveCounts, tag)

        self._sendEntries = {neighbour: entries for neighbour, entries in sendEntries.items() if entries.shape[0]}
        receiveCounts = {neighbour: int(count[0]) for neighbour, count in receiveCounts.items() if count[0]}

        sendTags = {
            neighbour: np.ascontiguousarray(np.stack([entryDofs[entries], entryOrder[entries]], axis=1).ravel())
            for neighbour, entries in self._sendEntries.items()
        }
        receiveTags = {neighbour: np.empty(2 * count, dtype=np.int64) for neighbour, count in receiveCounts.items()}
        _exchange(communicator, sendTags, receiveTags, tag)

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

    @property
    def nEntriesSent(self) -> int:
        """The number of entries this process sends to its neighbours per exchange."""

        return sum(entries.shape[0] for entries in self._sendEntries.values())

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
        """Collective among neighbours. Fill the merged buffer: this process' own entries, and the
        neighbours' at the degrees of freedom shared with them.

        Parameters
        ----------
        contributions
            This process' contribution buffer.
        """

        communicator = self.interface.communicator
        requests = [
            communicator.Irecv(self.merged[receiveSlice], source=neighbour, tag=self.tag)
            for neighbour, receiveSlice in self._receiveSlices.items()
        ]
        for neighbour, entries in self._sendEntries.items():
            np.take(contributions, entries, out=self._sendBuffers[neighbour])
            requests.append(communicator.Isend(self._sendBuffers[neighbour], dest=neighbour, tag=self.tag))

        np.take(contributions, self.ownEntries, out=self.merged[: self.ownEntries.shape[0]])
        # Waits on the requests, not the communicator: the guard against communicating inside an
        # agreement (Communicator.withoutCommunication) already refused the Isend/Irecv above.
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
            MessageTag.ELEMENT_FORCES,
        )
        self._targets = np.searchsorted(interface.interfaceDofs, self._contributions.tagDofs[self._contributions.order])

    def assemble(self, contributions: np.ndarray, vector: np.ndarray):
        """Collective among neighbours. Overwrite the interface entries of ``vector`` with the complete sums
        of all element contributions, in the order of the elements in the model.

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
        # A left fold per degree of freedom, in model order, as in the element loop.
        vector.view(np.ndarray)[interfaceDofs] = sumAtPositions(
            self._targets, exchange.merged[exchange.order], interfaceDofs.shape[0]
        )


class InterfaceLoadAssembly:
    """The nodal forces of the loads acting on the elements of one subdomain -- distributed loads and
    body forces -- and how to add them, at every degree of freedom, in the order a single process
    adds them.

    A load acting on an element is a contribution of that element, so it is evaluated where the
    element is: by the process computing it, with its current solution. Each process evaluates the
    loads of its elements into one buffer, an entry per element and degree of freedom, in the order
    of the loads -- the load in deck order, the face of its surface, the element in the set. At a
    degree of freedom another process integrates too, that process receives the entries, tagged with
    the place of the contribution in the order of all loads of the model; so every process
    integrating a degree of freedom holds all the load contributions there.

    They are *added* onto the vector, one after another, in that order: a single process adds every
    load contribution onto the net force the elements and the concentrated loads already left there
    (:meth:`~edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase.assembleLoads`), and
    floating-point addition is not associative, so the sums are formed the same way here -- the
    first contribution at every degree of freedom, then the second, and so on. Within one such
    *layer* no degree of freedom appears twice, so each is one vectorized addition.

    Constructed collectively among neighbours, whenever the subdomain or the loads change.

    Parameters
    ----------
    interface
        The subdomain interface.
    entryDofs
        The degree of freedom of every entry of this process' load buffer.
    entryLoadOrder
        The place of every entry in the order of all load contributions of the model.
    """

    def __init__(self, interface: SubdomainInterface, entryDofs: np.ndarray, entryLoadOrder: np.ndarray):
        entryDofs = np.asarray(entryDofs, dtype=np.int64)
        self._contributions = _TaggedContributions(
            interface, entryDofs, entryLoadOrder, np.arange(entryDofs.shape[0], dtype=np.int64), MessageTag.LOADS
        )
        order = self._contributions.order
        sortedDofs = self._contributions.tagDofs[order]

        # The rank of every sorted entry among those at its degree of freedom: 0 for the first added.
        firstOfDof = np.ones(sortedDofs.shape[0], dtype=bool)
        firstOfDof[1:] = sortedDofs[1:] != sortedDofs[:-1]
        starts = np.flatnonzero(firstOfDof)
        rankAtDof = np.arange(sortedDofs.shape[0]) - np.repeat(starts, np.diff(np.append(starts, sortedDofs.shape[0])))

        #: Per layer, the merged entries added and the degrees of freedom they are added at.
        self._layers = [
            (order[rankAtDof == rank], sortedDofs[rankAtDof == rank])
            for rank in range(int(rankAtDof.max()) + 1 if rankAtDof.size else 0)
        ]

    def assemble(self, contributions: np.ndarray, vector: np.ndarray):
        """Collective among neighbours. Add the load contributions of all processes at the subdomain degrees
        of freedom onto ``vector``, in the order of the loads of the model.

        Parameters
        ----------
        contributions
            This process' load buffer, one entry per loaded element and degree of freedom.
        vector
            The vector to add into.
        """

        exchange = self._contributions
        exchange.exchange(contributions)
        plain = vector.view(np.ndarray)
        for entries, dofs in self._layers:
            plain[dofs] += exchange.merged[entries]


class ConstraintForceExchange:
    """How the processes exchange the forces of the constraints they evaluate, every increment.

    A constraint is evaluated whole, by the process that owns it, which integrates every degree of
    freedom it acts on. Another process integrating one of them shares that degree of freedom with the
    owner, and needs the constraint's forces there -- only there. So the owner sends each neighbour,
    point to point, the forces of its constraints at the degrees of freedom the two share, and nothing
    else (:class:`_TaggedContributions`); no process receives a constraint it does not integrate.

    Every process then adds the forces it holds constraint by constraint, in model order, each as a
    single process adds it (:func:`~edelweissfe.numerics.assembly.addNodalForces`). At a subdomain
    degree of freedom those are the forces of every constraint acting there, so they are added in
    the order, and onto the same bits, as without decomposition: the sum is the same bits. Each entry is
    tagged with its place in that order: the position of its constraint in the model and its index
    within the constraint.

    Which degrees of freedom each constraint acts on changes only when the equation system is rebuilt,
    so the layout -- which entries go to which neighbour, and their tags -- is exchanged once per
    rebuild, here; an increment exchanges the forces alone.

    The net force vector so holds the constraint forces at the subdomain degrees of freedom only.
    Elsewhere it holds none, and is not read before it is next gathered from the owners
    (:meth:`SubdomainInterface.allgatherOwnedValues`).

    Constructed collectively among neighbours.

    Parameters
    ----------
    interface
        The subdomain interface.
    constraints
        The constraints of the model, by name, in model order.
    ownedConstraints
        Those this process evaluates, by name.
    constraintDofs
        The degrees of freedom of every owned constraint, by constraint.
    """

    def __init__(self, interface: SubdomainInterface, constraints: dict, ownedConstraints: dict, constraintDofs: dict):
        owned = [
            (position, name, np.asarray(constraintDofs[constraint], dtype=np.int64))
            for position, (name, constraint) in enumerate(constraints.items())
            if name in ownedConstraints
        ]
        self._ownedNames = [name for _, name, _ in owned]
        self._ownedOffsets = np.cumsum([0] + [dofs.shape[0] for _, _, dofs in owned]).astype(np.int64)
        #: The forces of the owned constraints, one after another, in model order.
        self._ownedForces = np.empty(int(self._ownedOffsets[-1]))

        if any(dofs.shape[0] >= _ENTRIES_PER_CONSTRAINT for _, _, dofs in owned):
            raise ValueError("A constraint acts on more degrees of freedom than its forces can be ordered by.")
        entryDofs = [np.empty(0, dtype=np.int64)] + [dofs for _, _, dofs in owned]
        entryOrder = [np.empty(0, dtype=np.int64)] + [
            position * _ENTRIES_PER_CONSTRAINT + np.arange(dofs.shape[0], dtype=np.int64) for position, _, dofs in owned
        ]
        self._contributions = _TaggedContributions(
            interface,
            np.concatenate(entryDofs),
            np.concatenate(entryOrder),
            np.arange(self._ownedForces.shape[0], dtype=np.int64),
            MessageTag.CONSTRAINT_FORCES,
        )

        # The merged entries, constraint by constraint in model order, each in its own order.
        tagOrder = self._contributions.tagOrder
        byOrder = np.argsort(tagOrder)
        constraintPositions = tagOrder[byOrder] // _ENTRIES_PER_CONSTRAINT
        #: Per local constraint: its merged entries, their degrees of freedom, and whether one of
        #: them is named more than once (see addNodalForces).
        self._additions = []
        for entries in np.split(byOrder, np.flatnonzero(np.diff(constraintPositions)) + 1):
            if entries.shape[0]:
                dofs = self._contributions.tagDofs[entries]
                self._additions.append((entries, dofs, np.unique(dofs).shape[0] != dofs.shape[0]))

    @property
    def nEntriesSent(self) -> int:
        """The number of constraint force entries this process sends to its neighbours per increment."""

        return self._contributions.nEntriesSent

    def addAllConstraintForces(self, ownedForces: dict, vector: np.ndarray):
        """Collective among neighbours. Send this process' constraint forces to the neighbours integrating
        their degrees of freedom, and add the forces of every constraint at the subdomain degrees of
        freedom, in model order.

        Parameters
        ----------
        ownedForces
            The forces of every owned constraint, by name.
        vector
            The net nodal force vector to add into; only its entries at the subdomain degrees
            of freedom are written.
        """

        for i, name in enumerate(self._ownedNames):
            self._ownedForces[self._ownedOffsets[i] : self._ownedOffsets[i + 1]] = ownedForces[name]

        exchange = self._contributions
        exchange.exchange(self._ownedForces)
        plain = vector.view(np.ndarray)
        for entries, dofs, namesDofMoreThanOnce in self._additions:
            addNodalForces(plain, dofs, exchange.merged[entries], namesDofMoreThanOnce)


class ValuesFromOwners:
    """Copies of vector entries at chosen degrees of freedom, received point to point from the
    processes owning them -- instead of the whole vector, where a process needs a few entries it does
    not integrate: the positions a contact search reads, or the entries at degrees of freedom it
    integrates from now on.

    Which entries each process sends to which is agreed once, here; :meth:`receive` then moves the
    values alone. Only the owners of the entries asked for and the processes asking take part in
    that; a process asking for nothing and owning nothing asked for sends and receives no message.
    A copy is a copy: the received entries are the owner's bits.

    Constructed collectively: every process of the communicator must construct its instance at the
    same time.

    Parameters
    ----------
    communicator
        The communicator of the processes sharing the model.
    neededDofs
        The degrees of freedom whose entries this process needs, in any order and with repetitions.
        Those it owns itself are not asked for.
    dofOwners
        The owner of every degree of freedom of the model, the same in every process.
    """

    def __init__(self, communicator, neededDofs: np.ndarray, dofOwners: np.ndarray):
        self.communicator = communicator
        rank = communicator.Get_rank()
        size = communicator.Get_size()

        neededDofs = _sortedUnique(np.asarray(neededDofs, dtype=np.int64))
        owners = dofOwners[neededDofs]
        asks = [neededDofs[owners == other] if other != rank else np.empty(0, dtype=np.int64) for other in range(size)]
        askedOfThisProcess = communicator.alltoall(asks)

        #: The degrees of freedom received from each owner, by rank.
        self._receiveDofs = {other: dofs for other, dofs in enumerate(asks) if dofs.shape[0]}
        #: The degrees of freedom sent to each process asking, by rank.
        self._sendDofs = {other: dofs for other, dofs in enumerate(askedOfThisProcess) if dofs.shape[0]}

    @property
    def nReceived(self) -> int:
        """The number of entries of each vector this process receives."""

        return sum(dofs.shape[0] for dofs in self._receiveDofs.values())

    def receive(self, vectors: list[np.ndarray]):
        """Collective among the processes taking part. Overwrite the needed entries of the vectors with
        their owners' values, in place, and send the owned entries other processes need.

        Parameters
        ----------
        vectors
            Vectors over the whole equation system, the same vectors in every process, correct at
            least at the owned entries.
        """

        plains = [vector.view(np.ndarray) for vector in vectors]
        receiveBuffers = {other: np.empty((len(plains), dofs.shape[0])) for other, dofs in self._receiveDofs.items()}
        sendBuffers = {other: np.stack([plain[dofs] for plain in plains]) for other, dofs in self._sendDofs.items()}
        _exchange(self.communicator, sendBuffers, receiveBuffers, MessageTag.VALUES_FROM_OWNERS)
        for other, dofs in self._receiveDofs.items():
            for plain, received in zip(plains, receiveBuffers[other]):
                plain[dofs] = received


def _exchange(communicator, sendBuffers: dict, receiveBuffers: dict, tag: MessageTag):
    """Collective among those processes. Send a buffer to each process of ``sendBuffers`` and receive one
    from each of ``receiveBuffers``, with the message tag of the exchange (:class:`MessageTag`)."""

    requests = [
        communicator.Irecv(receiveBuffers[neighbour], source=neighbour, tag=tag) for neighbour in receiveBuffers
    ]
    requests += [communicator.Isend(sendBuffers[neighbour], dest=neighbour, tag=tag) for neighbour in sendBuffers]
    # Waits on the requests, not the communicator: the guard against communicating inside an
    # agreement (Communicator.withoutCommunication) already refused the Isend/Irecv above.
    MPI.Request.Waitall(requests)


def _displacements(counts: np.ndarray) -> np.ndarray:
    """The receive displacements of a variable-length gather with these counts."""

    displacements = np.zeros_like(counts)
    np.cumsum(counts[:-1], out=displacements[1:])
    return displacements
