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

Every exchange is keyed by the global number of a degree of freedom
(:class:`~edelweissfe.numerics.globaldofnumbering.GlobalDofNumbering`), the same in every process.
Where it reads or writes a vector of this process, it does so at the *position* of the degree of
freedom in that vector (:meth:`SubdomainInterface.positionsOf`): its global number in a vector of the
whole model; its place among the sorted subdomain degrees of freedom in a vector of the subdomain
only -- the order-preserving compression of the global numbering, which the DofManager of the
subdomain's node fields numbers alike.
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


class ElementProcessesAtNodes:
    """Which processes' elements touch each field at each node of the mesh, derived in every process from
    the element partition and the connectivity of the mesh, which every process holds: made once per
    partition (:meth:`ofMesh`). Per field, the lowest-ranked process at every node, and every process at
    the nodes the elements of several processes touch.

    Parameters
    ----------
    fields
        Per field, the lowest process at every node (``nProcesses`` where no element touches the
        field there), the nodes touched by several processes (one entry per process, sorted), and the
        process of each of those entries.
    nNodes
        The number of nodes of the mesh.
    nProcesses
        The number of processes.
    """

    def __init__(self, fields: dict, nNodes: int, nProcesses: int):
        self.fields = fields
        self.nNodes = nNodes
        self.nProcesses = nProcesses

    @classmethod
    def ofMesh(cls, connectivity, processOfElements: np.ndarray, nNodes: int, nProcesses: int):
        """From the connectivity of the mesh and the process of every element.

        Parameters
        ----------
        connectivity
            The connectivity of the mesh (:meth:`~edelweissfe.models.mesh.Mesh.connectivity`).
        processOfElements
            The process of every element of the mesh, in mesh order.
        nNodes
            The number of nodes of the mesh.
        nProcesses
            The number of processes.

        Returns
        -------
        ElementProcessesAtNodes
            The processes at the nodes.
        """

        rowsOfField, processesOfField = {}, {}
        for group in connectivity.groups:
            groupProcesses = processOfElements[group.positions]
            for i, fieldsAtNode in enumerate(group.typeInfo.fields):
                for field in fieldsAtNode:
                    rowsOfField.setdefault(field, []).append(group.nodeRows[:, i])
                    processesOfField.setdefault(field, []).append(groupProcesses)
        return cls(
            {
                field: _processesAtPlaces(
                    np.concatenate(rows), np.concatenate(processesOfField[field]), nNodes, nProcesses
                )
                for field, rows in rowsOfField.items()
            },
            nNodes,
            nProcesses,
        )

    def lowestProcessAtNodes(self) -> np.ndarray:
        """The lowest-ranked process whose elements touch each node, whichever field.

        Returns
        -------
        np.ndarray
            The process per node, in mesh order; ``nProcesses`` for a node no element touches.
        """

        lowest = np.full(self.nNodes, self.nProcesses, dtype=np.int32)
        for lowestOfField, _, _ in self.fields.values():
            np.minimum(lowest, lowestOfField, out=lowest)
        return lowest


class ElementProcessesAtDofs:
    """Which processes' elements touch each degree of freedom of the model, derived in every process from
    data every process holds -- the element partition and the connectivity of the mesh
    (:class:`ElementProcessesAtNodes`), and the global numbering of the degrees of freedom
    (:meth:`ofNodes`) -- instead of gathered from the processes.

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

        lowest, sharedDofs, sharedProcesses = _processesAtPlaces(
            np.asarray(dofs, dtype=np.int64), processes, nDof, nProcesses
        )
        return cls(lowest, sharedDofs, sharedProcesses, nProcesses)

    @classmethod
    def ofNodes(cls, atNodes: "ElementProcessesAtNodes", numbering):
        """From which processes' elements touch each field at each node, and the global numbering of the
        degrees of freedom: an element touches every component of a field at a node.

        Parameters
        ----------
        atNodes
            The processes at the nodes, field by field.
        numbering
            The global numbering of the degrees of freedom
            (:class:`~edelweissfe.numerics.globaldofnumbering.GlobalDofNumbering`).

        Returns
        -------
        ElementProcessesAtDofs
            The processes at the degrees of freedom.
        """

        nProcesses = atNodes.nProcesses
        lowest = np.full(numbering.nDof, nProcesses, dtype=np.int32)
        sharedDofs, sharedProcesses = [], []
        for field in numbering.fields:
            if field not in atNodes.fields:
                continue
            nodeLowest, sharedNodes, sharedNodeProcesses = atNodes.fields[field]
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

    # 32-bit throughout: an entry per element and node of the mesh, the largest arrays of a rebuild
    processes = np.asarray(processes, dtype=np.int32)
    lowest = np.full(nPlaces, nProcesses, dtype=np.int32)
    np.minimum.at(lowest, places, processes)
    highest = np.full(nPlaces, -1, dtype=np.int32)
    np.maximum.at(highest, places, processes)
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
    vectorsOfSubdomain
        Whether the vectors of this process hold the subdomain degrees of freedom only, in ascending
        global order; else every degree of freedom of the model (see :meth:`positionsOf`).
    """

    def __init__(
        self,
        communicator,
        elementProcesses: "ElementProcessesAtDofs",
        touchedDofs: np.ndarray,
        multiPointConstraintLinks: tuple[np.ndarray, np.ndarray] | None = None,
        vectorsOfSubdomain: bool = False,
    ):
        self.communicator = communicator
        #: The number of degrees of freedom of the model.
        self.nDof = elementProcesses.nDof
        #: Whether the vectors of this process hold the subdomain degrees of freedom only.
        self.vectorsOfSubdomain = vectorsOfSubdomain
        self._elementProcesses = elementProcesses
        touched = _sortedUnique(np.asarray(touchedDofs, dtype=np.int64))

        self._requireDerivedElementDofsTouched(touched)
        self._gatherConstraintDofsBeyondElements(touched)
        closureDofs, closureProcesses = self._closeOverMultiPointConstraints(multiPointConstraintLinks)
        untouched = self._findOwners(closureDofs, closureProcesses)
        self._defineSubdomainDofs(touched, closureDofs, closureProcesses, untouched)
        self._planWholeVectorGather()
        self._findNeighbours(closureDofs, closureProcesses)
        #: The positions of the interface degrees of freedom in the vectors of this process.
        self.interfacePositions = self.positionsOf(self.interfaceDofs)
        #: Whether this process owns the degree of freedom at each position of its vectors.
        self.ownedPositionMask = self.ownedDofMask[self.subdomainDofs] if vectorsOfSubdomain else self.ownedDofMask
        # what the steps above read, not kept beyond them
        self._elementProcesses = None
        self._constraintDofsBeyondElements = self._constraintProcessesBeyondElements = None

    def _requireDerivedElementDofsTouched(self, touched: np.ndarray):
        """Collective. Refuse to continue, in every process, if the degrees of freedom the elements of any
        process touch, as derived from the partition and the mesh, include one its elements do not touch:
        the processes would then disagree on which process integrates which degree of freedom.

        Parameters
        ----------
        touched
            The degrees of freedom this process' elements and constraints touch, sorted.

        Raises
        ------
        RuntimeError
            In every process, if one process' derivation is wrong.
        """

        derived = self._elementProcesses.dofsOfProcess(self.communicator.Get_rank())
        derivedWrongly = not _isInSorted(derived, touched).all()
        if self.communicator.allreduce(int(derivedWrongly), op=MPI.MAX):
            raise RuntimeError(
                "The degrees of freedom the elements of a process touch, as derived from the element partition "
                "and the mesh, include some its elements do not touch: the processes would disagree on which "
                "process integrates which degree of freedom."
            )

    def _gatherConstraintDofsBeyondElements(self, touched: np.ndarray):
        """Collective. Gather to every process the degrees of freedom each process touches beyond its
        elements' -- those of the constraints it evaluates, current there only, which the others cannot
        derive --, each with its process.

        Parameters
        ----------
        touched
            The degrees of freedom this process' elements and constraints touch, sorted.
        """

        communicator = self.communicator
        derived = self._elementProcesses.dofsOfProcess(communicator.Get_rank())
        beyondElements = touched[~_isInSorted(touched, derived)]
        counts = np.array(communicator.allgather(beyondElements.shape[0]), dtype=np.int64)
        gathered = np.empty(counts.sum(), dtype=np.int64)
        communicator.Allgatherv(beyondElements, [gathered, counts, _displacements(counts), MPI.INT64_T])
        #: The degrees of freedom the processes touch beyond their elements', and the process of each.
        self._constraintDofsBeyondElements = gathered
        self._constraintProcessesBeyondElements = np.repeat(np.arange(counts.shape[0], dtype=np.int64), counts)
        #: How many there are, over all processes.
        self.nConstraintDofsBeyondElements = gathered.shape[0]

    def _touchingProcessesAt(self, dofs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Every process touching one of the given degrees of freedom, with its elements or its
        constraints.

        Parameters
        ----------
        dofs
            Degrees of freedom, sorted and without repetitions.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Pairs of a degree of freedom and a process, sorted, as two arrays.
        """

        elementDofs, elementProcesses = self._elementProcesses.processesAt(dofs)
        atDofs = _isInSorted(self._constraintDofsBeyondElements, dofs)
        return _uniquePairs(
            np.concatenate([elementDofs, self._constraintDofsBeyondElements[atDofs]]),
            np.concatenate([elementProcesses, self._constraintProcessesBeyondElements[atDofs]]),
            self.communicator.Get_size(),
        )

    def _closeOverMultiPointConstraints(self, links) -> tuple[np.ndarray, np.ndarray]:
        """The processes integrating the degrees of freedom linked by multi-point constraints: a slave's
        force is folded onto its masters, its velocity interpolated from theirs, so a process touching
        one degree of freedom of a group (:func:`multiPointConstraintGroups`) integrates all of it.

        Parameters
        ----------
        links
            The slave and the master of every dependency, or None.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Every linked degree of freedom paired with every process touching its group, as two arrays.
        """

        size = self.communicator.Get_size()
        linkedDofs, groupOfLinked = multiPointConstraintGroups(links)
        touchingDofs, touchingProcesses = self._touchingProcessesAt(linkedDofs)
        groupAndProcess = _sortedUnique(
            groupOfLinked[np.searchsorted(linkedDofs, touchingDofs)] * size + touchingProcesses
        )
        return _membersWithProcesses(linkedDofs, groupOfLinked, groupAndProcess // size, groupAndProcess % size)

    def _findOwners(self, closureDofs: np.ndarray, closureProcesses: np.ndarray) -> np.ndarray:
        """The owner of every degree of freedom: the lowest-ranked process integrating it; rank 0 for one
        nothing touches, which rank 0 integrates, so that every equation is integrated somewhere.

        Parameters
        ----------
        closureDofs, closureProcesses
            The linked degrees of freedom with the processes integrating them
            (:meth:`_closeOverMultiPointConstraints`).

        Returns
        -------
        np.ndarray
            Whether each degree of freedom is touched by nothing.
        """

        size, rank = self.communicator.Get_size(), self.communicator.Get_rank()
        owner = self._elementProcesses.lowestProcess.astype(np.int64)
        np.minimum.at(owner, self._constraintDofsBeyondElements, self._constraintProcessesBeyondElements)
        np.minimum.at(owner, closureDofs, closureProcesses)
        untouched = owner == size
        owner[untouched] = 0
        #: The owner of every degree of freedom.
        self.dofOwners = owner
        #: Whether this process owns each degree of freedom, over the whole equation system.
        self.ownedDofMask = owner == rank
        return untouched

    def _defineSubdomainDofs(self, touched, closureDofs, closureProcesses, untouched: np.ndarray):
        """The degrees of freedom this process integrates: those it touches, the rest of every
        multi-point-constraint group among them, and, on rank 0, those nothing touches."""

        rank = self.communicator.Get_rank()
        subdomainDofs = [touched, closureDofs[closureProcesses == rank]]
        if rank == 0:
            subdomainDofs.append(np.flatnonzero(untouched))
        #: The sorted degrees of freedom this process integrates.
        self.subdomainDofs = _sortedUnique(np.concatenate(subdomainDofs))

    def _planWholeVectorGather(self):
        """The whole-vector gather: each process sends its owned entries, in ascending order. A buffer of
        the whole model is kept only where the vectors are of the whole model; else a gather allocates
        it for its own duration (:meth:`wholeVector`)."""

        self._ownedDofs = np.flatnonzero(self.ownedDofMask)
        self._ownedPositions = self.positionsOf(self._ownedDofs)
        self._ownedCounts = np.bincount(self.dofOwners, minlength=self.communicator.Get_size()).astype(np.int64)
        self._gatheredOrder = np.argsort(self.dofOwners, kind="stable")
        self._gatherBuffer = None if self.vectorsOfSubdomain else np.empty(self.nDof)

    @property
    def nPositions(self) -> int:
        """The length of the vectors of this process: the number of subdomain degrees of freedom, or of
        degrees of freedom of the model."""

        return self.subdomainDofs.shape[0] if self.vectorsOfSubdomain else self.nDof

    def positionsOf(self, dofs: np.ndarray) -> np.ndarray:
        """The positions of degrees of freedom in the vectors of this process: their global numbers in a
        vector of the whole model, their places among the sorted subdomain degrees of freedom in a vector
        of the subdomain.

        Parameters
        ----------
        dofs
            Global degrees of freedom; subdomain degrees of freedom where the vectors are of the subdomain.

        Returns
        -------
        np.ndarray
            Their positions, in the same order.

        Raises
        ------
        RuntimeError
            If the vectors are of the subdomain, and a degree of freedom is not a subdomain degree of
            freedom.
        """

        dofs = np.asarray(dofs, dtype=np.int64)
        if not self.vectorsOfSubdomain:
            return dofs
        positions = np.searchsorted(self.subdomainDofs, dofs)
        if dofs.shape[0] and not np.array_equal(
            self.subdomainDofs[np.minimum(positions, self.subdomainDofs.shape[0] - 1)], dofs
        ):
            raise RuntimeError("A degree of freedom read or written here is not a subdomain degree of freedom.")
        return positions

    def globalDofsOf(self, positions: np.ndarray) -> np.ndarray:
        """The global degrees of freedom at positions of the vectors of this process: the inverse of
        :meth:`positionsOf`.

        Parameters
        ----------
        positions
            Positions in a vector of this process.

        Returns
        -------
        np.ndarray
            The global degrees of freedom.
        """

        positions = np.asarray(positions, dtype=np.int64)
        return self.subdomainDofs[positions] if self.vectorsOfSubdomain else positions

    def _findNeighbours(self, closureDofs: np.ndarray, closureProcesses: np.ndarray):
        """The interface: every other process integrating a subdomain degree of freedom of this one."""

        rank = self.communicator.Get_rank()
        touchingDofs, touchingProcesses = self._touchingProcessesAt(self.subdomainDofs)
        atSubdomain = _isInSorted(closureDofs, self.subdomainDofs)
        sharedDofs, sharedProcesses = _uniquePairs(
            np.concatenate([touchingDofs, closureDofs[atSubdomain]]),
            np.concatenate([touchingProcesses, closureProcesses[atSubdomain]]),
            self.communicator.Get_size(),
        )
        other = sharedProcesses != rank
        sharedDofs, sharedProcesses = sharedDofs[other], sharedProcesses[other]

        #: The degrees of freedom this process shares with at least one other, sorted.
        self.interfaceDofs = _sortedUnique(sharedDofs)
        #: The neighbouring processes, ascending.
        self.neighbours = np.unique(sharedProcesses).tolist()
        #: The degrees of freedom shared with each neighbour, sorted (the pairs are sorted by DOF first).
        self._sharedWith = {neighbour: sharedDofs[sharedProcesses == neighbour] for neighbour in self.neighbours}

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

        self._requireVectorsOfWholeModel()
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

        self._requireVectorsOfWholeModel()
        plain = vector.view(np.ndarray)
        send = np.ascontiguousarray(plain[self._ownedDofs])
        if self.communicator.Get_rank() == 0:
            self.communicator.Gatherv(
                send, [self._gatherBuffer, self._ownedCounts, _displacements(self._ownedCounts), MPI.DOUBLE], root=0
            )
            plain[self._gatheredOrder] = self._gatherBuffer
        else:
            self.communicator.Gatherv(send, None, root=0)

    def wholeVector(self, vector: np.ndarray, toEveryProcess: bool) -> np.ndarray | None:
        """Collective. The vector of the whole model, every entry from its owner, from vectors of this
        process' kind -- of the subdomain or of the whole model --, made for this call: in every process,
        or on rank 0 only.

        Parameters
        ----------
        vector
            A vector of this process, correct at least at the owned entries.
        toEveryProcess
            Whether every process receives the whole vector, or rank 0 only.

        Returns
        -------
        np.ndarray | None
            The vector of the whole model, in global order; None in a process that receives none.
        """

        send = np.ascontiguousarray(vector.view(np.ndarray)[self._ownedPositions])
        layout = [None, self._ownedCounts, _displacements(self._ownedCounts), MPI.DOUBLE]
        receives = toEveryProcess or self.communicator.Get_rank() == 0
        gathered = np.empty(self.nDof) if receives else None
        layout[0] = gathered
        if toEveryProcess:
            self.communicator.Allgatherv(send, layout)
        elif receives:
            self.communicator.Gatherv(send, layout, root=0)
        else:
            self.communicator.Gatherv(send, None, root=0)
        if not receives:
            return None
        whole = np.empty(self.nDof)
        whole[self._gatheredOrder] = gathered
        return whole

    def _requireVectorsOfWholeModel(self):
        """Refuse an exchange that writes vectors of the whole model in place, where the vectors are of
        the subdomain (:meth:`wholeVector` instead)."""

        if self.vectorsOfSubdomain:
            raise RuntimeError("the vectors of this process hold its subdomain only, not the whole model")


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

    # The members of the groups one group after another, and where the members of each group begin.
    membersInGroupOrder = linkedDofs[np.argsort(groupOfLinked, kind="stable")]
    nMembers = np.bincount(groupOfLinked, minlength=int(groups.max()) + 1 if groups.size else 0)
    firstMember = np.cumsum(nMembers) - nMembers

    # A (group, process) pair stands for one pair per member of its group: repeat it that often, and
    # number the repetitions 0, 1, ... within each pair to step through the group's members.
    repeats = nMembers[groups]
    withinGroup = np.arange(int(repeats.sum())) - np.repeat(np.cumsum(repeats) - repeats, repeats)
    members = membersInGroupOrder[np.repeat(firstMember[groups], repeats) + withinGroup]
    return members, np.repeat(processes, repeats)


def _isInSorted(values: np.ndarray, sortedSet: np.ndarray) -> np.ndarray:
    """Whether each value is in a sorted array: :func:`numpy.isin` by binary search, without the lookup
    table or the sort of both arrays numpy builds.

    Parameters
    ----------
    values
        The values.
    sortedSet
        The array to look them up in, sorted.

    Returns
    -------
    np.ndarray
        Per value, True if it is in ``sortedSet``.
    """

    if not sortedSet.shape[0]:
        return np.zeros(values.shape[0], dtype=bool)
    positions = np.minimum(np.searchsorted(sortedSet, values), sortedSet.shape[0] - 1)
    return sortedSet[positions] == values


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
        The degree of freedom of every entry of this process' contribution buffer, as its position in
        the vectors of this process (:meth:`SubdomainInterface.positionsOf`).
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
        interfaceEntries = self.entriesAtInterface(interface, entryDofs)
        # tagged, sent and summed by the global degree of freedom
        entryDofs = interface.globalDofsOf(entryDofs)
        entryOrder = np.asarray(entryOrder, dtype=np.int64)

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
            The degree of freedom of every entry, as its position in the vectors of this process.

        Returns
        -------
        np.ndarray
            The entries, as indices into the buffer, ascending.
        """

        isInterface = np.zeros(interface.nPositions, dtype=bool)
        isInterface[interface.interfacePositions] = True
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
        The degree of freedom of every entry of this process' contribution buffer, as its position in
        the vectors of this process.
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
        interfacePositions = exchange.interface.interfacePositions
        # A left fold per degree of freedom, in model order, as in the element loop.
        vector.view(np.ndarray)[interfacePositions] = sumAtPositions(
            self._targets, exchange.merged[exchange.order], interfacePositions.shape[0]
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
        The degree of freedom of every entry of this process' load buffer, as its position in the
        vectors of this process.
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

        #: Per layer, the merged entries added and the positions of the degrees of freedom they are
        #: added at.
        self._layers = [
            (order[rankAtDof == rank], interface.positionsOf(sortedDofs[rankAtDof == rank]))
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
        The degrees of freedom of every owned constraint, by constraint, as their positions in the
        vectors of this process.
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
        #: Per local constraint: its merged entries, the positions of their degrees of freedom, and
        #: whether one of them is named more than once (see addNodalForces).
        self._additions = []
        for entries in np.split(byOrder, np.flatnonzero(np.diff(constraintPositions)) + 1):
            if entries.shape[0]:
                dofs = interface.positionsOf(self._contributions.tagDofs[entries])
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
    receivedInto
        The interface whose vectors receive the entries (:meth:`SubdomainInterface.positionsOf`); None
        for vectors of the whole model.
    sentFrom
        The interface whose vectors the entries are sent from; None for vectors of the whole model.
    """

    def __init__(
        self,
        communicator,
        neededDofs: np.ndarray,
        dofOwners: np.ndarray,
        receivedInto: SubdomainInterface | None = None,
        sentFrom: SubdomainInterface | None = None,
    ):
        self.communicator = communicator
        rank = communicator.Get_rank()
        size = communicator.Get_size()

        neededDofs = _sortedUnique(np.asarray(neededDofs, dtype=np.int64))
        owners = dofOwners[neededDofs]
        asks = [neededDofs[owners == other] if other != rank else np.empty(0, dtype=np.int64) for other in range(size)]
        askedOfThisProcess = communicator.alltoall(asks)

        #: The positions of the degrees of freedom received from each owner in the receiving vectors, by
        #: rank.
        self._receiveDofs = {
            other: dofs if receivedInto is None else receivedInto.positionsOf(dofs)
            for other, dofs in enumerate(asks)
            if dofs.shape[0]
        }
        #: The positions of the degrees of freedom sent to each process asking in the sending vectors,
        #: by rank.
        self._sendDofs = {
            other: dofs if sentFrom is None else sentFrom.positionsOf(dofs)
            for other, dofs in enumerate(askedOfThisProcess)
            if dofs.shape[0]
        }

    @property
    def nReceived(self) -> int:
        """The number of entries of each vector this process receives."""

        return sum(dofs.shape[0] for dofs in self._receiveDofs.values())

    def receive(self, vectors: list[np.ndarray], sentFromVectors: list[np.ndarray] | None = None):
        """Collective among the processes taking part. Overwrite the needed entries of the vectors with
        their owners' values, in place, and send the owned entries other processes need.

        Parameters
        ----------
        vectors
            The receiving vectors, the same vectors in every process.
        sentFromVectors
            The vectors the owned entries are sent from, one per receiving vector, correct at least at
            the owned entries; None for ``vectors`` themselves.
        """

        plains = [vector.view(np.ndarray) for vector in vectors]
        sending = plains if sentFromVectors is None else [vector.view(np.ndarray) for vector in sentFromVectors]
        receiveBuffers = {other: np.empty((len(plains), dofs.shape[0])) for other, dofs in self._receiveDofs.items()}
        sendBuffers = {other: np.stack([plain[dofs] for plain in sending]) for other, dofs in self._sendDofs.items()}
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
