Sets
====

Currently, EdelweissFE supports nodesets and elementsets.
Since for the usage of nodesets or elementsets
in the outputmanagers or constraints
the ordering of the respective objects is of importance
they are based on a custom ordered set object.

Node sets hold :class:`~edelweissfe.points.node.Node` objects. An element set holds the elements of
its set in the mesh (``model.mesh.elementSets``) that were created in this process -- in a serial run,
all of them -- and knows whether that is all of them; a reader that needs the whole set calls
:meth:`~edelweissfe.sets.elementset.ElementSet.requireComplete`. See :ref:`mesh_to_elements`.

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
