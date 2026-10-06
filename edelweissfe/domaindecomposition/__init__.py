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
"""Domain decomposition: computing one model in several MPI processes.

Every process reads the same input file and builds the same mesh, nodes, sets and definitions, and
*computes* one subdomain of the model: the elements a graph partitioner assigned to it, the
constraints assigned to it, and the degrees of freedom those touch. A degree of freedom at the
interface between two subdomains is integrated by both, and its nodal force is completed by summing
the partial forces of every subdomain touching it (:mod:`.subdomaininterface`). A
:class:`~.subdomain.Subdomain` decides the subdomain of a process -- as the
:class:`~edelweissfe.solvers.base.modelpartition.ModelPartition` the explicit increment runs over,
in place of the whole model -- and carries out every exchange between the processes.

Which elements a process *creates* is decided once per job (:mod:`.distributedelements`): a
**distributed** model is partitioned before its elements exist, and each process creates only its
own elements (and the loaded elements touching them; contact facets and the point masses of rigid
bodies, which are surface-sized, are made by every process), so that the memory of the elements
falls with the number of processes; whole-model readers gather what they read, and adaptive
refinement and contact read the mesh, the nodes and the facets, which every process holds whole. A
model with something that still reads or changes the element objects of the whole model during a
run -- some model modifiers, constraints and generators -- holds the **whole model on every
process** instead: every process then keeps the state of the parts it does not compute current
(:mod:`.statesynchronization`).

Nothing here is imported by a serial run except :mod:`.mpienvironment`, which decides whether the
process was started by an MPI launcher, and :mod:`.distributedelements`, which then asks it; only
under a launcher is ``mpi4py`` imported.
"""
