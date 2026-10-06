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
"""The mesh of a model, as data.

A model is built in two stages:

1. **Describe the mesh.** The ``*element``/``*elset``/``*surface`` keywords and the mesh generators
   fill the :class:`Mesh` held by the model as ``model.mesh``: per element its number, type, provider
   and node labels; element sets as ordered lists of element numbers; surfaces as element faces. No
   element object exists yet.
2. **Make the elements.** :meth:`~edelweissfe.models.femodel.FEModel.createElementsOfMesh` creates an
   element object for each element of the mesh, and resolves the element sets and surfaces to those
   objects.

Nodes are not part of this description: they are created directly as :class:`~edelweissfe.points.node.Node`
objects in ``model.nodes``, and node sets as :class:`~edelweissfe.sets.nodeset.NodeSet` objects, as before.

Which fields an element has at which of its nodes depends on its type only. The mesh therefore
answers that question per element without any element object of the model, by asking one prototype
element per type (:meth:`Mesh.typeOf`). That is what lets the model activate the fields at its
nodes -- and so lay out its degrees of freedom -- from the mesh alone.
"""

from dataclasses import dataclass

import numpy as np

from edelweissfe.utils.exceptions import TopologyError


@dataclass(frozen=True)
class ElementTypeInfo:
    """What the mesh needs to know about an element type, without an element of the model.

    Parameters
    ----------
    fields
        The fields of the element, per node: ``fields[i]`` lists the fields at its ``i``-th node.
    ensightType
        The element shape, e.g. ``hexa20``; it determines the faces of the element.
    """

    fields: list
    ensightType: str


class MeshElement:
    """One element of the mesh, as data.

    Parameters
    ----------
    number
        The element number.
    elType
        The element type, e.g. ``C3D20R``.
    provider
        The element provider, e.g. ``marmot``; ``None`` for an element its owner made itself (see
        :meth:`Mesh.addElementMadeByOwner`), which cannot be created from this record.
    nodeLabels
        The labels of the element's nodes, in the element's node order.
    ownTypeInfo
        The type information of an element made by its owner; ``None`` for an element whose type
        alone determines it.
    """

    __slots__ = ("number", "elType", "provider", "nodeLabels", "ownTypeInfo")

    def __init__(
        self, number: int, elType: str, provider: str, nodeLabels: np.ndarray, ownTypeInfo: ElementTypeInfo = None
    ):
        self.number = number
        self.elType = elType
        self.provider = provider
        self.nodeLabels = nodeLabels
        self.ownTypeInfo = ownTypeInfo

    @property
    def isMadeByOwner(self) -> bool:
        """True for an element its owner made itself (a contact facet, a point mass); it cannot be
        created from this record."""
        return self.ownTypeInfo is not None


class Mesh:
    """The mesh of a model: its elements, element sets and surfaces, as data.

    Element numbers are never chosen here: the input file gives them, or the model's number
    allocator (:meth:`~edelweissfe.models.topologypipeline.TopologyPipeline.reserveElementNumbers`)
    hands them out, so that the mesh, the contact facets and every model modifier share one counter.
    """

    def __init__(self):
        #: The elements, by number, in the order they were described.
        self.elements = {}
        #: The element sets, by name: ordered lists of element numbers without duplicates.
        self.elementSets = {}
        #: The element-based surfaces, by name: per face number, either the name of an element set
        #: of this mesh or (once a model modifier has refined it) the list of element numbers.
        self.surfaces = {}
        #: Element class and type information per (type, provider), asked once per type.
        self._types = {}

    def addElement(self, number: int, elType: str, provider: str, nodeLabels) -> MeshElement:
        """Describe an element.

        Parameters
        ----------
        number
            The element number, from the input file or the model's number allocator.
        elType
            The element type.
        provider
            The element provider; ``None`` selects the default provider (``marmot``).
        nodeLabels
            The labels of the element's nodes.

        Returns
        -------
        MeshElement
            The record of the element.
        """

        if number in self.elements:
            raise TopologyError(
                "element number {:} is already taken in the mesh -- every element number is described once, "
                "and numbers from TopologyPipeline.reserveElementNumbers() are never recycled".format(number)
            )

        record = MeshElement(number, elType, provider, np.asarray(nodeLabels, dtype=np.int64))
        self.elements[number] = record
        return record

    def addElementMadeByOwner(self, element) -> MeshElement:
        """Describe an element its owner made itself, from the element object.

        Contact facets and the point masses of rigid bodies are made by the code that owns them
        (with data an element type alone does not carry: a parent face, a mass), on every process.
        They are part of the mesh all the same -- they have numbers, nodes and fields -- so they are
        described here from the object, including the fields it reports.

        Parameters
        ----------
        element
            The element object.

        Returns
        -------
        MeshElement
            The record of the element.
        """

        record = self.addElement(element.elNumber, element.elType, None, [node.label for node in element.nodes])
        record.ownTypeInfo = ElementTypeInfo(element.fields, element.ensightType)
        return record

    def removeElement(self, number: int):
        """Remove an element from the mesh. Its number is retired, never reissued.

        The element sets are left as they are: whoever removes elements updates the sets it changes
        (see :meth:`setElementSet`).

        Parameters
        ----------
        number
            The element number.
        """

        del self.elements[number]

    def setElementSet(self, name: str, numbers):
        """Define (or redefine) an element set.

        Parameters
        ----------
        name
            The name of the set.
        numbers
            The element numbers, in order; repeated numbers are kept once, at their first position.
        """

        numbers = list(dict.fromkeys(int(number) for number in numbers))
        missing = [number for number in numbers if number not in self.elements]
        if missing:
            raise KeyError(
                "element set {:}: element(s) {:} are not in the mesh".format(name, ", ".join(map(str, missing[:10])))
            )
        self.elementSets[name] = numbers

    def addSurface(self, name: str, faceToElementSetName: dict):
        """Define an element-based surface.

        Parameters
        ----------
        name
            The name of the surface.
        faceToElementSetName
            Per face number, the name of the element set of this mesh whose elements expose that face.
        """

        for setName in faceToElementSetName.values():
            if setName not in self.elementSets:
                raise KeyError("surface {:}: element set {:} is not in the mesh".format(name, setName))
        self.surfaces[name] = dict(faceToElementSetName)

    def setSurfaceElements(self, name: str, faceToElementNumbers: dict):
        """Redefine a surface by the element numbers of each face, e.g. after a refinement replaced
        the elements of a face by their children.

        Parameters
        ----------
        name
            The name of the surface.
        faceToElementNumbers
            Per face number, the element numbers, in order.
        """

        self.surfaces[name] = {face: list(numbers) for face, numbers in faceToElementNumbers.items()}

    def elementNumbersOfSurface(self, name: str) -> dict:
        """The element numbers of each face of a surface.

        Parameters
        ----------
        name
            The name of the surface.

        Returns
        -------
        dict
            Per face number, the element numbers, in order.
        """

        return {
            face: self.elementSets[entry] if isinstance(entry, str) else entry
            for face, entry in self.surfaces[name].items()
        }

    def elementClassOf(self, record: MeshElement) -> type:
        """The class that creates the element of a record.

        Parameters
        ----------
        record
            The record of the element.

        Returns
        -------
        type
            The element class.
        """

        if record.isMadeByOwner:
            raise TopologyError(
                "element {:} ({:}) was made by its owner and cannot be created from the mesh".format(
                    record.number, record.elType
                )
            )
        return self._typeEntry(record.elType, record.provider)[0]

    def typeOf(self, record: MeshElement) -> ElementTypeInfo:
        """The fields per node and the shape of an element, without an element object of the model.

        Parameters
        ----------
        record
            The record of the element.

        Returns
        -------
        ElementTypeInfo
            The type information.
        """

        if record.ownTypeInfo is not None:
            return record.ownTypeInfo
        return self._typeEntry(record.elType, record.provider)[1]

    def _typeEntry(self, elType: str, provider: str) -> tuple:
        """The element class and the type information of an element type, from one prototype
        element created once per (type, provider)."""

        key = (elType, provider)
        entry = self._types.get(key)
        if entry is None:
            from edelweissfe.config.elementlibrary import getElementClass

            elementClass = getElementClass(elType, provider)
            prototype = elementClass(elType, 0)
            entry = (elementClass, ElementTypeInfo(prototype.fields, prototype.ensightType))
            self._types[key] = entry
        return entry

    def __getstate__(self):
        """Pickle (and deep-copy) the data only; the prototype elements are recreated on demand."""

        state = self.__dict__.copy()
        state["_types"] = {}
        return state
