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

Each process holds the complete model -- every node, element, set and constraint, built from the
same input file -- and *computes* one subdomain of it: the elements a graph partitioner assigned to
it, the constraints assigned to it, and the degrees of freedom those touch. A degree of freedom at
the interface between two subdomains is integrated by both, and its nodal force is completed by
summing the partial forces of every subdomain touching it (:mod:`.subdomaininterface`).
The subdomain of a process, as the explicit solver computes it, is a :mod:`.subdomain` -- the
model partition of :mod:`edelweissfe.solvers.base.modelpartition` that a domain-decomposed solver
creates in place of the whole model.

Holding the whole model everywhere is what makes adaptive refinement tractable: at a topology change
every process first receives the current state of the parts it did not compute
(:mod:`.statesynchronization`), then runs the same, deterministic refinement on the same data, and
the model is partitioned afresh (:mod:`.partitioning`). No element ever migrates, because every
process holds the complete model, every element included. Holding only the elements a process
computes -- memory that falls with the number of processes -- is planned.

Nothing here is imported by a serial run; :mod:`.mpienvironment` decides whether the process was
started by an MPI launcher, and only then imports ``mpi4py``.
"""
