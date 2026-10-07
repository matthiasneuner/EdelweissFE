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
#  Alexander Dummer alexander.dummer@uibk.ac.at
#  Paul Hofer Paul.Hofer@uibk.ac.at
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

from edelweissfe.elements.base.baseelement import BaseElement
from edelweissfe.utils.misc import checkSuccessfulExtension

if checkSuccessfulExtension("edelweissfe.elements.marmotelement.element"):
    from edelweissfe.elements.marmotelement.element import MarmotElementWrapper
else:
    MarmotElementWrapper = None

if checkSuccessfulExtension("edelweissfe.materials.marmot.marmothypoelastic") or checkSuccessfulExtension(
    "edelweissfe.materials.marmot.marmotgradientenhancedhypoelastic"
):
    # MarmotMaterialWrappingElement can drive either point-wise material family; each has its
    # own separately compiled extension, and only one of the two building is enough for the
    # element to be partially usable (see materialdrivers.py's createMaterial methods for the
    # per-family lazy import that fails, cleanly, if its own extension isn't built).
    from edelweissfe.elements.marmotsingleqpelement.element import (
        MarmotMaterialWrappingElement,
    )
else:
    MarmotMaterialWrappingElement = None

from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.sets.orderedset import OrderedSet
from edelweissfe.utils.exceptions import TopologyError
from edelweissfe.utils.meshtools import extractNodesFromElementSet


class ElementSet(OrderedSet):
    """A basic element set.
    It has a name, and a list containing unique elements.

    **Local and complete.** In a distributed (domain-decomposed) run each process holds only some
    of the elements of the model -- its *local* elements, in the terms of deal.II and PETSc. An
    element set of a model resolved from its mesh is an :class:`ElementSetOfMesh`, which holds the
    local elements of its set. The set is *complete* if every element of the set is local here. A
    serial run is always complete, and so is a set of element objects like this one, which holds the
    elements it was given.

    A reader that wants the local part asks for it explicitly, with :meth:`localElements`. What is
    known of the whole set in every process is read through :meth:`elementNumbersOfWholeSet` and
    :meth:`extractNodeSet`; a reader that needs the whole set states it with :meth:`requireComplete`.

    Parameters
    ----------
    name
        The unique name of this element set.
    elements
        A list of elements.
    """

    def __init__(
        self,
        label: str,
        elements,
    ):
        self.allowedObjectTypes = [BaseElement]
        self.allowedObjectTypes.append(MarmotElementWrapper) if MarmotElementWrapper is not None else None
        (
            self.allowedObjectTypes.append(MarmotMaterialWrappingElement)
            if MarmotMaterialWrappingElement is not None
            else None
        )

        super().__init__(label, elements)
        self._nodes = None

        self.elements = self.items

    @property
    def isComplete(self) -> bool:
        """Whether every element of the set is local to this process: always, for a set of the
        element objects it was given."""

        return True

    def localElements(self):
        """The elements of the set local to this process, in set order, as a read-only view: every
        element of the set if it is complete, e.g. in a serial run.

        Returns
        -------
        KeysView
            The local elements.
        """

        return self.items

    def elementNumbersOfWholeSet(self) -> list:
        """The numbers of every element of the set, in set order: here, of its elements.

        Returns
        -------
        list
            The element numbers.
        """

        return [element.elNumber for element in self]

    def requireComplete(self, reader: str):
        """State that ``reader`` needs every element of this set, not only its local part.

        A set that is not complete, read as if it were whole, gives silently wrong results -- an
        output, a marker or a checkpoint computed from part of the set. A whole-set reader therefore
        calls this, so that such a reading fails loudly instead.

        Parameters
        ----------
        reader
            Who reads the set, for the error message.

        Raises
        ------
        TopologyError
            If this process created only some of the elements of the set.
        """

        if not self.isComplete:
            raise TopologyError(
                "{:} needs the whole element set {:}, but only part of it is local to this process".format(
                    reader, self.name
                )
            )

    def extractNodeSet(
        self,
    ):
        """The nodes of the set, without duplicates, in the order the elements list them.

        Returns
        -------
        NodeSet
            The nodes, named like the set.
        """
        if not self._nodes:
            self._nodes = self._findNodes()
        return self._nodes

    def _findNodes(self) -> NodeSet:
        """The nodes of the set, for :meth:`extractNodeSet`: here, those of its elements.

        Returns
        -------
        NodeSet
            The nodes, named like the set.
        """
        return extractNodesFromElementSet(self)

    def forgetNodes(self):
        """Forget the nodes :meth:`extractNodeSet` found, because the set changed."""
        self._nodes = None

    def replaceMembers(self, item_s):
        """Replace all members in-place (see :meth:`OrderedSet.replaceMembers`), additionally
        invalidating the cached :meth:`extractNodeSet` result, which is stale once the element
        membership changes."""
        super().replaceMembers(item_s)
        self.forgetNodes()


class ElementSetOfMesh(ElementSet):
    """An element set of a model, resolved from its mesh: it holds the elements of the set that are
    **local** to this process (see :class:`ElementSet`).

    The model describes its mesh as data first and then creates the element objects from it
    (:meth:`~edelweissfe.models.femodel.FEModel.createElementsOfMesh`), which resolves every element
    set of the mesh to such a set (:meth:`~edelweissfe.models.femodel.FEModel.resolveElementSetOfMesh`,
    the only place one is made). In a serial run that is every element of the set. A domain-decomposed
    run may create only part of the mesh in each process; a set then holds only its local part, and says
    so through :attr:`isComplete`. What the mesh and the nodes describe is known for the whole set in
    every process: its element numbers (:meth:`elementNumbersOfWholeSet`) and its nodes
    (:meth:`extractNodeSet`).

    Parameters
    ----------
    label
        The name of the set, in the mesh as here.
    elements
        The elements of the set local to this process, in set order.
    mesh
        The :class:`~edelweissfe.models.mesh.Mesh` describing this set under the same name.
    nodesOfModel
        The nodes of the model, by label; the nodes of the whole set are taken from them.
    """

    def __init__(self, label: str, elements, mesh, nodesOfModel: dict):
        super().__init__(label, elements)
        self.describedBy(mesh, nodesOfModel)

    def describedBy(self, mesh, nodesOfModel: dict):
        """Read the whole set from the given mesh and nodes from now on, and forget the nodes found
        before: the set may have changed in the mesh, also where the part of it created here did not.

        Parameters
        ----------
        mesh
            The mesh describing this set under the same name.
        nodesOfModel
            The nodes of the model, by label.
        """

        #: The mesh describing this set.
        self.mesh = mesh
        #: The nodes of the model, by label.
        self.nodesOfModel = nodesOfModel
        self.forgetNodes()

    @property
    def isComplete(self) -> bool:
        """True if every element of the set in the mesh is local to this process. Derived, not stored,
        so that no change of the members can leave it stale. Counting suffices, since the set holds
        only elements of its set in the mesh."""

        return len(self.data) == len(self._numbersInMesh())

    def _numbersInMesh(self):
        """The numbers of every element of the set in the mesh, in set order, as the mesh holds them
        (not a copy).

        Returns
        -------
        list
            The element numbers.
        """

        return self.mesh.elementSets[self.name]

    def elementNumbersOfWholeSet(self) -> list:
        """The numbers of every element of the set, in set order -- also of those not local here,
        read from the mesh.

        Returns
        -------
        list
            The element numbers.
        """

        return list(self._numbersInMesh())

    def _findNodes(self) -> NodeSet:
        """The nodes of the whole set, also where only part of it was created here: the node labels of
        its elements, read from the mesh, which every process holds whole.

        Returns
        -------
        NodeSet
            The nodes, named like the set.
        """
        records = self.mesh.elements
        labels = dict.fromkeys(label for number in self._numbersInMesh() for label in records[number].nodeLabels)
        return NodeSet(self.name, [self.nodesOfModel[label] for label in labels])


class ElementSetOfSurfaceFace(ElementSetOfMesh):
    """The elements exposing one face of a surface of the mesh, where the mesh lists them by number
    instead of naming an element set (a surface refined by a model modifier; see
    :class:`~edelweissfe.models.mesh.SurfaceFace`). It holds the local elements of the face, like
    an :class:`ElementSetOfMesh` of a named set, and is complete if all of them are local.

    Parameters
    ----------
    surfaceName
        The name of the surface, in the mesh as in the model.
    face
        The face number.
    elements
        The elements of the face local to this process, in order.
    mesh
        The :class:`~edelweissfe.models.mesh.Mesh` describing the surface.
    nodesOfModel
        The nodes of the model, by label.
    """

    def __init__(self, surfaceName: str, face: int, elements, mesh, nodesOfModel: dict):
        #: The name of the surface.
        self.surfaceName = surfaceName
        #: The face number.
        self.face = face
        super().__init__("{:}_S{:}".format(surfaceName, face), elements, mesh, nodesOfModel)

    def _numbersInMesh(self):
        """The numbers of the elements exposing the face, in order, as the mesh lists them.

        Returns
        -------
        tuple
            The element numbers.
        """

        return self.mesh.surfaces[self.surfaceName][self.face].elementNumbers
