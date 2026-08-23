#!/usr/bin/env python3
"""Offline diagnosis of BlockAMGSolver degradation dumps from the AnchorPryOut perf/linsolve-investigation
run. Reads examples/AnchorPryOut/degradation_dumps/manifest.jsonl and probes a representative subset of
the ten captured degraded systems: per-block structural stats, isolated-field AMG quality (displacement
vs. nonlocal damage), a fresh production replay at fixed eta, a replay with the nonlocal-damage field's
AMGCL preconditioner tuned to match displacement's (power_iters, eps_strong), and a PARDISO-exact
block-Gauss-Seidel ceiling test on the single worst system.
"""

import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, gmres

sys.path.insert(0, "/home/taylor/constitutive_modeling/next_v2611/EdelweissFE")

from edelweissfe.linsolve.amgcl.amgcl import PyAMGCLSolver  # noqa: E402
from edelweissfe.linsolve.base import FieldBlock  # noqa: E402
from edelweissfe.linsolve.blockamg.blockamg import (  # noqa: E402
    _DEFAULT_SCALAR_PRECOND,
    _DEFAULT_VECTOR_PRECOND,
    BlockAMGSolver,
)
from edelweissfe.linsolve.blockamg.nullspace import translationNullspace  # noqa: E402

DUMP_DIR = "/home/taylor/constitutive_modeling/next_v2611/EdelweissFE/examples/AnchorPryOut/degradation_dumps"


def loadManifest():
    with open(os.path.join(DUMP_DIR, "manifest.jsonl")) as f:
        records = [json.loads(line) for line in f if line.strip()]
    records.sort(key=lambda r: r["solveCount"])
    return records


def loadSystem(record):
    A = sp.load_npz(os.path.join(DUMP_DIR, record["matrixFile"])).tocsr()
    b = np.load(os.path.join(DUMP_DIR, record["rhsFile"]))
    return A, b


def equilibrate(A, b):
    dinv = 1.0 / np.sqrt(np.abs(A.diagonal()))
    As = (sp.diags(dinv) @ A @ sp.diags(dinv)).tocsr()
    bs = dinv * b
    return As, bs, dinv


def blockStats(As, blocks):
    slices = {blk["name"]: slice(blk["start"], blk["stop"]) for blk in blocks}
    stats = {}
    for name, sl in slices.items():
        diagBlk = As[sl, :][:, sl].tocsr()
        d = diagBlk.diagonal()
        adiag = np.abs(d)
        stats[name] = dict(
            n=diagBlk.shape[0],
            nnz=diagBlk.nnz,
            nnzPerRow=diagBlk.nnz / diagBlk.shape[0],
            diagMin=float(adiag.min()),
            diagMax=float(adiag.max()),
            diagMedian=float(np.median(adiag)),
            fracBelow1em3Max=float(np.mean(adiag < 1e-3 * adiag.max())),
            fracBelow1em6Max=float(np.mean(adiag < 1e-6 * adiag.max())),
            frob=float(np.sqrt((diagBlk.data**2).sum())),
        )
    names = list(slices.keys())
    for i in range(len(names)):
        for j in range(len(names)):
            if i == j:
                continue
            off = As[slices[names[i]], :][:, slices[names[j]]]
            key = "coupling {:} -> {:}".format(names[i], names[j])
            stats[key] = dict(
                nnz=off.nnz,
                frob=float(np.sqrt((off.data**2).sum())) if off.nnz else 0.0,
            )
    return stats


def isolatedFieldTest(diagBlk, precondParams, label, nullspace=None, eta=1e-6, maxiter=400, seed=0):
    n = diagBlk.shape[0]
    solver = PyAMGCLSolver({"precond": precondParams, "backendPrecision": "double", "backendBlockSize": 1})
    if nullspace is not None:
        solver.set_nullspace(nullspace)
    t0 = time.time()
    solver.build(diagBlk)
    buildTime = time.time() - t0
    M = LinearOperator((n, n), matvec=solver.applyPreconditioner, dtype=diagBlk.dtype)
    rng = np.random.default_rng(seed)
    b = rng.standard_normal(n)
    hist = []
    t0 = time.time()
    x, info = gmres(
        diagBlk,
        b,
        M=M,
        rtol=eta,
        atol=0.0,
        restart=100,
        maxiter=maxiter,
        callback=lambda r: hist.append(r),
        callback_type="pr_norm",
    )
    solveTime = time.time() - t0
    trueRes = float(np.linalg.norm(diagBlk @ x - b) / np.linalg.norm(b))
    print(
        "    [{:38s}] iters={:4d} info={:d} trueRes={:.2e} build={:6.2f}s solve={:6.2f}s".format(
            label, len(hist), info, trueRes, buildTime, solveTime
        ),
        flush=True,
    )
    return dict(label=label, iters=len(hist), info=info, trueRes=trueRes, buildTime=buildTime, solveTime=solveTime)


def productionReplay(A, b, blocks, eta, fieldPreconds=None, sweeps=1, label="baseline"):
    fieldBlocks = [FieldBlock(blk["name"], blk["start"], blk["stop"], blk["dimension"]) for blk in blocks]
    solver = BlockAMGSolver(
        outerTol=eta,
        outerSolver="amgcl_lgmres",
        sweeps=sweeps,
        symmetric=True,
        fieldPreconds=fieldPreconds,
        useRigidBodyNullspace=True,  # will fall back to translations-only: no model/coords in an offline probe
        verbosity="silent",
    )
    solver.setFieldStructure(fieldBlocks)
    t0 = time.time()
    x = solver(A.copy(), b.copy())
    elapsed = time.time() - t0
    trueRes = float(np.linalg.norm(A @ x - b) / np.linalg.norm(b))
    print(
        "    [{:38s}] outerIters={:4d} trueRes={:.2e} time={:7.2f}s".format(
            label, solver._lastOuterIters, trueRes, elapsed
        ),
        flush=True,
    )
    return dict(label=label, outerIters=solver._lastOuterIters, trueRes=trueRes, time=elapsed)


def pardisoBlockGSCeiling(A, b, blocks, eta, maxOuter=60):
    """Exact per-field solves (PARDISO) inside the same symmetric block-Gauss-Seidel sweep blockamg.py
    uses, as the outer preconditioner for scipy GMRES -- establishes the best case the block-splitting
    scheme itself can reach, independent of AMG quality within a field."""
    from edelweissfe.linsolve.pardiso.pardiso import PardisoSolver

    n = A.shape[0]
    As, bs, dinv = equilibrate(A, b)
    slices = [slice(blk["start"], blk["stop"]) for blk in blocks]
    nFields = len(slices)
    sizes = [blk["stop"] - blk["start"] for blk in blocks]

    diagBlocks = [As[sl, :][:, sl].tocsr() for sl in slices]
    offBlocks = {}
    for i in range(nFields):
        rowBlock = As[slices[i], :]
        for j in range(nFields):
            if i != j:
                offBlocks[(i, j)] = rowBlock[:, slices[j]].tocsr()

    solvers = []
    for blk in diagBlocks:
        s = PardisoSolver(reuseSymbolicFactorization=False)
        t0 = time.time()
        s.factorize(blk)
        print(
            "      factorized {:} x {:} block in {:.2f}s".format(blk.shape[0], blk.shape[1], time.time() - t0),
            flush=True,
        )
        solvers.append(s)

    def sweepOnce(order, residual, x):
        for i in order:
            localResidual = residual[slices[i]].copy()
            for j in range(nFields):
                if j != i:
                    localResidual -= offBlocks[(i, j)] @ x[j]
            x[i] = solvers[i].solveFactorized(localResidual)

    def blockGS(residual):
        x = [np.zeros(sizes[i]) for i in range(nFields)]
        sweepOnce(range(nFields), residual, x)
        sweepOnce(range(nFields - 1, -1, -1), residual, x)
        return np.concatenate(x)

    M = LinearOperator((n, n), matvec=blockGS, dtype=As.dtype)
    hist = []
    t0 = time.time()
    z, info = gmres(
        As,
        bs,
        M=M,
        rtol=eta,
        atol=0.0,
        restart=100,
        maxiter=maxOuter,
        callback=lambda r: hist.append(r),
        callback_type="pr_norm",
    )
    elapsed = time.time() - t0
    x = dinv * z
    trueRes = float(np.linalg.norm(A @ x - b) / np.linalg.norm(b))
    print(
        "    [{:38s}] outerIters={:4d} info={:d} trueRes={:.2e} time={:7.2f}s".format(
            "PARDISO-exact block-GS ceiling", len(hist), info, trueRes, elapsed
        ),
        flush=True,
    )
    return dict(outerIters=len(hist), info=info, trueRes=trueRes, time=elapsed)


def main():
    records = loadManifest()
    print("=== manifest summary ({:} dumps) ===".format(len(records)))
    print(
        "{:>10} {:>8} {:>10} {:>10} {:>10} {:>18} {:>10}".format(
            "solveCount", "rows", "outerIters", "prevIters", "eta", "newIncrement", "growth"
        )
    )
    for r in records:
        growth = (
            "-"
            if r["previousOuterIters"] is None
            else "{:+.1f}x".format(r["outerIters"] / max(r["previousOuterIters"], 1))
        )
        print(
            "{:>10} {:>8} {:>10} {:>10} {:>10.1e} {:>18} {:>10}".format(
                r["solveCount"],
                r["rows"],
                r["outerIters"],
                r["previousOuterIters"] or -1,
                r["eta"],
                r["newIncrement"],
                growth,
            )
        )
    print()

    # Representative subset: the worst loose-tolerance case, a mid-run pattern-change case, and a
    # late-run, larger-mesh tight-tolerance case.
    targets = [658, 694, 1773]
    selected = [r for r in records if r["solveCount"] in targets]

    allProductionResults = {}
    for record in selected:
        print("=" * 100)
        print(
            "solve #{:} rows={:} outerIters(live)={:} eta={:.2e} newIncrement={:}".format(
                record["solveCount"], record["rows"], record["outerIters"], record["eta"], record["newIncrement"]
            )
        )
        A, b = loadSystem(record)
        blocks = record["blocks"]
        As, bs, dinv = equilibrate(A, b)

        print("  -- block structure --")
        stats = blockStats(As, blocks)
        for name, s in stats.items():
            print("    {:38s} {:}".format(name, s))

        dispBlk = next(blk for blk in blocks if blk["name"] == "displacement")
        dmgBlk = next(blk for blk in blocks if blk["name"] == "nonlocal damage")
        dispSlice = slice(dispBlk["start"], dispBlk["stop"])
        dmgSlice = slice(dmgBlk["start"], dmgBlk["stop"])
        dispDiag = As[dispSlice, :][:, dispSlice].tocsr()
        dmgDiag = As[dmgSlice, :][:, dmgSlice].tocsr()

        print("  -- isolated-field AMG quality (random rhs, eta=1e-8, no coupling) --")
        nsDisp = translationNullspace(
            FieldBlock("displacement", dispBlk["start"], dispBlk["stop"], dispBlk["dimension"]),
            dinv[dispSlice],
        )
        isolatedFieldTest(
            dispDiag, dict(_DEFAULT_VECTOR_PRECOND), "displacement, production params", nullspace=nsDisp, eta=1e-8
        )
        isolatedFieldTest(dmgDiag, dict(_DEFAULT_SCALAR_PRECOND), "damage, production (untuned) params", eta=1e-8)
        tunedScalar = dict(_DEFAULT_SCALAR_PRECOND)
        tunedScalar["relax"] = {"type": "chebyshev", "degree": 5, "power_iters": 300, "lower": 0.01}
        isolatedFieldTest(dmgDiag, tunedScalar, "damage, +power_iters=300,lower=0.01", eta=1e-8)
        tunedScalar2 = dict(tunedScalar)
        tunedScalar2["coarsening"] = {"type": "smoothed_aggregation", "aggr": {"eps_strong": 0.01}}
        isolatedFieldTest(dmgDiag, tunedScalar2, "damage, +eps_strong=0.01 too", eta=1e-8)

        print("  -- full-system production replay at fixed eta (translations-only nullspace: no coords offline) --")
        base = productionReplay(A, b, blocks, record["eta"], fieldPreconds=None, label="baseline (current defaults)")
        tunedFieldPreconds = {"nonlocal damage": tunedScalar2}
        tuned = productionReplay(
            A, b, blocks, record["eta"], fieldPreconds=tunedFieldPreconds, label="damage field tuned like displacement"
        )
        moreSweeps = productionReplay(
            A, b, blocks, record["eta"], fieldPreconds=None, sweeps=2, label="sweeps=2 (baseline precond)"
        )
        allProductionResults[record["solveCount"]] = dict(baseline=base, tuned=tuned, moreSweeps=moreSweeps)
        print()

    # PARDISO-exact block-GS ceiling test on the single worst system only (expensive).
    worst = next(r for r in records if r["solveCount"] == 658)
    print("=" * 100)
    print("PARDISO-exact block-Gauss-Seidel ceiling test on solve #{:}".format(worst["solveCount"]))
    A, b = loadSystem(worst)
    pardisoBlockGSCeiling(A, b, worst["blocks"], worst["eta"], maxOuter=60)

    print()
    print("=== summary: production replay outerIters, baseline vs. damage-field-tuned vs. sweeps=2 ===")
    for solveCount, res in allProductionResults.items():
        print(
            "  solve #{:}: baseline={:}  tuned={:}  sweeps=2={:}".format(
                solveCount, res["baseline"]["outerIters"], res["tuned"]["outerIters"], res["moreSweeps"]["outerIters"]
            )
        )


if __name__ == "__main__":
    main()
