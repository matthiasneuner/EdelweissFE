Sets
====

Currently, EdelweissFE supports nodesets and elementsets.
Since for the usage of nodesets or elementsets
in the outputmanagers or constraints
the ordering of the respective objects is of importance
they are based on a custom ordered set object.

Node sets hold :class:`~edelweissfe.points.node.Node` objects. An element set of a model
(:class:`~edelweissfe.sets.elementset.ElementSetOfMesh`) holds the elements of its set in the mesh
(``model.mesh.elementSets``) that are *local* to this process -- in a serial run, all of them -- and
is *complete* if that is all of them. Reading a set that is not complete as a whole (iterating it,
indexing it, ``len()``) raises; a reader of the local part asks for
:meth:`~edelweissfe.sets.elementset.ElementSet.localElements`. See :ref:`mesh_to_elements`.

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
