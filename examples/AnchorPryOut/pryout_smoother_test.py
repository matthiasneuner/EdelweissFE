#!/usr/bin/env python3
"""Follow-up to pryout_diagnose.py: test whether a smoother that tolerates non-symmetry (ILU0,
Gauss-Seidel, SPAI0) beats Chebyshev on the displacement block specifically -- the block the earlier
probe isolated as the sole bottleneck (PARDISO-exact block-GS ceiling: 3 outer iters vs 229 with AMG).
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
    _DEFAULT_VECTOR_PRECOND,
    BlockAMGSolver,
)

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


def isolatedFieldTest(diagBlk, precondParams, label, nullspace=None, eta=1e-8, maxiter=400, seed=0):
    n = diagBlk.shape[0]
    solver = PyAMGCLSolver({"precond": precondParams, "backendPrecision": "double", "backendBlockSize": 1})
    if nullspace is not None:
        solver.set_nullspace(nullspace)
    t0 = time.time()
    try:
        solver.build(diagBlk)
    except Exception as exc:
        print("    [{:38s}] BUILD FAILED: {:}".format(label, exc), flush=True)
        return None
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


def productionReplay(A, b, blocks, eta, fieldPreconds, sweeps, label):
    fieldBlocks = [FieldBlock(blk["name"], blk["start"], blk["stop"], blk["dimension"]) for blk in blocks]
    solver = BlockAMGSolver(
        outerTol=eta,
        outerSolver="amgcl_lgmres",
        sweeps=sweeps,
        symmetric=True,
        fieldPreconds=fieldPreconds,
        useRigidBodyNullspace=True,
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


def main():
    records = loadManifest()
    targets = [658, 694, 1773]
    selected = [r for r in records if r["solveCount"] in targets]

    # Filling in the gap: earlier rounds showed 5->15 is a big win, 15->40 keeps cutting iterations but
    # wall-clock turns *worse* past ~15-21 (per-iteration cost outgrows the iteration savings). This
    # round runs full-system production replay (not just the isolated block) at every degree in the
    # 5..21 range to find where wall-clock actually bottoms out.
    degrees = [5, 7, 9, 11, 15, 21]

    summary = {}
    for record in selected:
        print("=" * 100)
        print(
            "solve #{:} rows={:} outerIters(live)={:} eta={:.2e}".format(
                record["solveCount"], record["rows"], record["outerIters"], record["eta"]
            )
        )
        A, b = loadSystem(record)
        blocks = record["blocks"]

        print("  -- full-system production replay, varying displacement Chebyshev degree --")
        rows = []
        for degree in degrees:
            relax = {"type": "chebyshev", "degree": degree, "power_iters": 300, "lower": 0.01}
            fieldPreconds = None if degree == 5 else {"displacement": {**_DEFAULT_VECTOR_PRECOND, "relax": relax}}
            label = "degree={:}{:}".format(degree, " (baseline)" if degree == 5 else "")
            result = productionReplay(A, b, blocks, record["eta"], fieldPreconds=fieldPreconds, sweeps=1, label=label)
            rows.append((degree, result["outerIters"], result["time"]))
        summary[record["solveCount"]] = rows
        print()

    print("=== summary: outerIters / time(s) by degree, full-system replay ===")
    header = "{:>10}".format("solve") + "".join("{:>16}".format("deg={:}".format(d)) for d in degrees)
    print(header)
    for solveCount, rows in summary.items():
        line = "{:>10}".format(solveCount) + "".join(
            "{:>16}".format("{:}/{:.1f}s".format(iters, t)) for _, iters, t in rows
        )
        print(line)


if __name__ == "__main__":
    main()
