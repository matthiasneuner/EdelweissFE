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

"""Registry-facing factory for the Schur complement preconditioner that condenses embedded bar dofs.

The implementation lives in :mod:`edelweissfe.linsolve.barschur.barschur`; see there for the method.
"""

from collections.abc import Callable, Mapping


def createSolver(opts) -> Callable:
    """Create the bar Schur complement solver from a ``linsolverConfigFile``'s options.

    Parameters
    ----------
    opts
        The options: ``outerTolerance`` (relative residual of the outer flexible GMRES, default 1e-3),
        ``innerCycles`` (applications of the inner preconditioner per outer iteration, default 2; ``null`` for inner
        solves to ``innerTolerance``), ``innerTolerance`` (relative tolerance of each inner ``blockamg`` solve, default
        1e-1), ``restart`` (default 50), ``maxIterations`` (default 100), ``schurApproximation`` (``"local"``, the
        default, ``"nodal"`` or ``"none"``), ``schurDropTolerance`` (default 1e-4), ``schurWindow`` (default 600),
        ``schurHalo`` (default: ``schurWindow``), ``dumpSlowSolvesDir``/``dumpSlowSolveSeconds``/``dumpSlowSolvesMax``
        (default none/300/5), ``reuseSchurComplement`` (default true), ``schurStalenessFactor`` (default 1.5) and
        ``blockamg`` (the options of the inner ``blockamg`` solver). A key that is not one of them raises a
        ``ValueError``. A non-mapping ``opts`` is tolerated, in which case every default applies.

    Returns
    -------
    Callable
        A :class:`~edelweissfe.linsolve.barschur.barschur.BarSchurSolver`, callable as ``(A, b) -> x``.
    """

    from edelweissfe.linsolve.barschur.barschur import BarSchurSolver

    optionMap = dict(opts) if isinstance(opts, Mapping) else {}
    known = {
        "outerTolerance",
        "innerTolerance",
        "innerCycles",
        "restart",
        "maxIterations",
        "schurApproximation",
        "schurDropTolerance",
        "schurWindow",
        "schurHalo",
        "dumpSlowSolvesDir",
        "dumpSlowSolveSeconds",
        "dumpSlowSolvesMax",
        "reuseSchurComplement",
        "schurStalenessFactor",
        "blockamg",
    }
    unknown = set(optionMap) - known
    if unknown:
        raise ValueError(f"barschur: unknown option(s) {sorted(unknown)}; known are {sorted(known)}")
    return BarSchurSolver(
        blockamgOptions=optionMap.get("blockamg", {}),
        outerTolerance=float(optionMap.get("outerTolerance", 1e-3)),
        innerTolerance=float(optionMap.get("innerTolerance", 1e-1)),
        innerCycles=None if optionMap.get("innerCycles", 2) is None else int(optionMap.get("innerCycles", 2)),
        restart=int(optionMap.get("restart", 50)),
        maxIterations=int(optionMap.get("maxIterations", 100)),
        schurApproximation=str(optionMap.get("schurApproximation", "local")),
        schurDropTolerance=float(optionMap.get("schurDropTolerance", 1e-4)),
        schurWindow=int(optionMap.get("schurWindow", 600)),
        schurHalo=None if optionMap.get("schurHalo") is None else int(optionMap["schurHalo"]),
        dumpSlowSolvesDir=optionMap.get("dumpSlowSolvesDir"),
        dumpSlowSolveSeconds=float(optionMap.get("dumpSlowSolveSeconds", 300.0)),
        dumpSlowSolvesMax=int(optionMap.get("dumpSlowSolvesMax", 5)),
        reuseSchurComplement=bool(optionMap.get("reuseSchurComplement", True)),
        schurStalenessFactor=float(optionMap.get("schurStalenessFactor", 1.5)),
    )
