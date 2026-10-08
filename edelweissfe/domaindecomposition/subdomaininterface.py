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

import numpy as np
from mpi4py import MPI

_INTERFACE_TAG = 7
#: The bound on the entries of one constraint: an entry of a constraint force is ordered by the
#: position of its constraint in the model times this, plus its index within the constraint.
_ENTRIES_PER_CONSTRAINT = 2**32


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
        isLocalDof = np.zeros(nDof, dtype=bool)
        isLocalDof[localDofs] = True
        shared = isLocalDof[allDofs] & (allRanks != rank)
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

    def allgatherOwnedValues(self, vector: np.ndarray):
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
        # A left fold per degree of freedom, in model order, as in the element loop; see
        # ElementPlan.assembleInto for why np.bincount sums in the order given.
        vector.view(np.ndarray)[interfaceDofs] = np.bincount(
            self._targets, weights=exchange.merged[exchange.order], minlength=interfaceDofs.shape[0]
        )


class _ContributionsAddedInOrder:
    """Nodal contributions, each with its place in the order a single process adds them, added onto a
    vector one after another in that order at every degree of freedom integrated here; what
    :class:`InterfaceLoadAssembly` and :class:`ConstraintForceExchange` have in common.

    Each process holds its contributions in one buffer. At a degree of freedom another process
    integrates too, that process receives the entries (:class:`_TaggedContributions`), so every process
    integrating a degree of freedom holds all the contributions there. They are *added* onto the vector,
    which already holds the same bits on every process integrating the degree of freedom, one after
    another, in the order of their tags: floating-point addition is not associative, so the sums are
    formed as a single process forms them -- the first contribution at every degree of freedom, then the
    second, and so on. Within one such *layer* no degree of freedom appears twice, so each is one
    vectorized addition.

    Only the entries of the vector at the degrees of freedom integrated here are written.

    Constructed collectively among neighbours.

    Parameters
    ----------
    interface
        The subdomain interface.
    entryDofs
        The degree of freedom of every entry of this process' buffer.
    entryOrder
        The place of every entry in the order a single process adds all contributions in.
    """

    def __init__(self, interface: SubdomainInterface, entryDofs: np.ndarray, entryOrder: np.ndarray):
        entryDofs = np.asarray(entryDofs, dtype=np.int64)
        self._contributions = _TaggedContributions(
            interface, entryDofs, entryOrder, np.arange(entryDofs.shape[0], dtype=np.int64)
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
        """Add the contributions of all processes at the degrees of freedom integrated here onto
        ``vector``, in the order a single process adds them. Collective among neighbours.

        Parameters
        ----------
        contributions
            This process' buffer, one entry per contribution.
        vector
            The vector to add into.
        """

        exchange = self._contributions
        exchange.exchange(contributions)
        plain = vector.view(np.ndarray)
        for entries, dofs in self._layers:
            plain[dofs] += exchange.merged[entries]


class InterfaceLoadAssembly(_ContributionsAddedInOrder):
    """The nodal forces of the loads acting on the elements of one subdomain -- distributed loads and
    body forces -- and how to add them, at every degree of freedom, in the order a single process
    adds them.

    A load acting on an element is a contribution of that element, so it is evaluated where the
    element is: by the process computing it, with its current solution. Each process evaluates the
    loads of its elements into one buffer, an entry per element and degree of freedom, in the order
    of the loads -- the load in deck order, the face of its surface, the element in the set. Each
    entry is tagged with the place of the contribution in the order of all loads of the model, and
    added onto the net force the elements and the concentrated loads already left there, as a single
    process adds every load contribution
    (:meth:`~edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase.assembleLoads`); see
    :class:`_ContributionsAddedInOrder`.

    Constructed collectively among neighbours, whenever the subdomain or the loads change.

    Parameters
    ----------
    interface
        The subdomain interface.
    entryDofs
        The degree of freedom of every entry of this process' load buffer.
    entryOrder
        The place of every entry in the order of all load contributions of the model.
    """


class ConstraintForceExchange(_ContributionsAddedInOrder):
    """How the processes exchange the forces of the constraints they evaluate, every increment.

    A constraint is evaluated whole, by the process that owns it, which integrates every degree of
    freedom it acts on. Another process integrating one of them shares that degree of freedom with the
    owner, and needs the constraint's forces there -- only there. So the owner sends each neighbour,
    point to point, the forces of its constraints at the degrees of freedom the two share, and nothing
    else; no process receives a constraint whole unless it integrates all of it.

    Every process then adds, at each degree of freedom it integrates, the forces of all constraints
    acting there in model order -- a constraint after the constraints before it in the model, and the
    entries of one constraint in their own order, as :func:`~edelweissfe.numerics.assembly.addNodalForces`
    adds them -- which is the order a single process adds them in, so the sum is the same bits; see
    :class:`_ContributionsAddedInOrder`. Each entry is tagged with that place: the position of its
    constraint in the model and its index within the constraint.

    Which degrees of freedom each constraint acts on changes only when the equation system is rebuilt,
    so the layout -- which entries go to which neighbour, and their tags -- is exchanged once per
    rebuild, here; an increment exchanges the forces alone.

    The net force vector holds the constraint forces at the degrees of freedom integrated here only;
    elsewhere it holds none, and is not read before it is next gathered from the owners
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
        entryDofs = np.concatenate([dofs for _, _, dofs in owned]) if owned else np.empty(0, dtype=np.int64)
        entryOrder = (
            np.concatenate(
                [
                    position * _ENTRIES_PER_CONSTRAINT + np.arange(dofs.shape[0], dtype=np.int64)
                    for position, _, dofs in owned
                ]
            )
            if owned
            else np.empty(0, dtype=np.int64)
        )
        super().__init__(interface, entryDofs, entryOrder)

    def addAllConstraintForces(self, ownedForces: dict, vector: np.ndarray):
        """Send this process' constraint forces to the neighbours integrating their degrees of freedom,
        and add the forces of every constraint at the degrees of freedom integrated here, in model order.

        Collective among neighbours.

        Parameters
        ----------
        ownedForces
            The forces of every owned constraint, by name.
        vector
            The net nodal force vector to add into; only its entries at the degrees of freedom
            integrated here are written.
        """

        for i, name in enumerate(self._ownedNames):
            self._ownedForces[self._ownedOffsets[i] : self._ownedOffsets[i + 1]] = ownedForces[name]
        self.assemble(self._ownedForces, vector)


def _exchange(communicator, sendBuffers: dict, receiveBuffers: dict):
    """Exchange one buffer with each neighbour, both ways. Collective among neighbours."""

    requests = [
        communicator.Irecv(receiveBuffers[neighbour], source=neighbour, tag=_INTERFACE_TAG)
        for neighbour in receiveBuffers
    ]
    requests += [
        communicator.Isend(sendBuffers[neighbour], dest=neighbour, tag=_INTERFACE_TAG) for neighbour in sendBuffers
    ]
    # Waits on the requests, not the communicator: the guard against communicating inside an
    # agreement (Communicator.withoutCommunication) already refused the Isend/Irecv above.
    MPI.Request.Waitall(requests)


def _displacements(counts: np.ndarray) -> np.ndarray:
    """The receive displacements of a variable-length gather with these counts."""

    displacements = np.zeros_like(counts)
    np.cumsum(counts[:-1], out=displacements[1:])
    return displacements
