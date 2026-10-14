Sets
====

Currently, EdelweissFE supports nodesets and elementsets.
Since for the usage of nodesets or elementsets
in the outputmanagers or constraints
the ordering of the respective objects is of importance
they are based on a custom ordered set object.

Element sets and node sets are local ordered sets
(:class:`~edelweissfe.sets.orderedset.LocalOrderedSet`). An element set of a model
(:class:`~edelweissfe.sets.elementset.ElementSetOfMesh`) holds the elements of its set in the mesh
(``model.mesh.elementSets``) that are *local* to this process -- in a serial run, all of them -- and
is *complete* if that is all of them. A node set of the mesh
(:class:`~edelweissfe.sets.nodeset.NodeSetOfMesh`) likewise holds the
:class:`~edelweissfe.points.node.Node` objects of its set in the mesh (``model.mesh.nodeSets``) held in
this process. Reading a set that is not complete as a whole (iterating it, indexing it, ``len()``)
raises; a reader of the local part asks for
:meth:`~edelweissfe.sets.elementset.ElementSet.localElements` or
:meth:`~edelweissfe.sets.nodeset.NodeSet.localNodes`. A set of the objects it was given -- e.g. the
slave nodes a tie tied -- is always complete. See :ref:`mesh_to_elements`.

Relevant modules:

 * ``edelweissfe.sets.orderedset``
 * ``edelweissfe.sets.nodeset``
 * ``edelweissfe.sets.elementset``

.. automodule:: edelweissfe.sets.orderedset
   :members:

.. automodule:: edelweissfe.sets.nodeset
   :members:

.. automodule:: edelweissfe.sets.elementset
   :members:
