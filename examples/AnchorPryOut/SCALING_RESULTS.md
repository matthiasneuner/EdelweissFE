# Thread Scaling Study: AnchorPryOut (Optimized Flags)

Model: `EdelweissFE/examples/AnchorPryOut` (Step 2, max 10 increments)
Environment: Conda `next_v2611`, Python 3.14 (free-threaded), AMGCL + OpenMP CSR v2 (C++20, -O3, -march=native)

| Threads | Total Wall Time (s) | Speedup (vs 4T) | Parallel Efficiency (%) | Status |
| :---: | :---: | :---: | :---: | :---: |
| 16 | 888.43 s | 1.00x | 25.0% | SUCCESS |
| 32 | 817.75 s | 1.09x | 13.6% | SUCCESS |
