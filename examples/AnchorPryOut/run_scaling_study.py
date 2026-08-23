import os
import shutil
import subprocess
import time

thread_counts = [16, 32]
base_dir = "/home/taylor/constitutive_modeling/next_v2611/EdelweissFE/examples/AnchorPryOut"
study_dir = os.path.join(base_dir, "scaling_study")

os.makedirs(study_dir, exist_ok=True)

conda_python = "/home/taylor/constitutive_modeling/miniforge3/envs/next_v2611/bin/python3.14"
edelweissfe_bin = "/home/taylor/constitutive_modeling/miniforge3/envs/next_v2611/bin/edelweissfe"

results = []

print("==========================================================================", flush=True)
print("Starting Thread Scaling Study for AnchorPryOut (Step 2, max 10 increments)", flush=True)
print(f"Threads to test: {thread_counts}", flush=True)
print("==========================================================================", flush=True)

for threads in thread_counts:
    run_dir = os.path.join(study_dir, f"run_threads_{threads}")
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir, exist_ok=True)

    # Copy required input files
    shutil.copy(os.path.join(base_dir, "test_scaling.inp"), os.path.join(run_dir, "test_scaling.inp"))
    shutil.copy(os.path.join(base_dir, "blockamg.json"), os.path.join(run_dir, "blockamg.json"))
    shutil.copy(os.path.join(base_dir, "zsymm_free_block.txt"), os.path.join(run_dir, "zsymm_free_block.txt"))
    if not os.path.exists(os.path.join(run_dir, "mesh")):
        os.symlink(os.path.join(base_dir, "mesh"), os.path.join(run_dir, "mesh"))

    env = os.environ.copy()
    env["PYTHON_GIL"] = "0"
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    env["OPENBLAS_NUM_THREADS"] = str(threads)
    env["MKL_ENABLE_INSTRUCTIONS"] = "AVX2"
    env["OMP_PROC_BIND"] = "spread"
    env["OMP_PLACES"] = "cores"

    log_file_path = os.path.join(run_dir, "run.log")

    print(f"\n---> Running with OMP_NUM_THREADS = {threads} ...", flush=True)
    start_time = time.perf_counter()

    with open(log_file_path, "w") as log_file:
        proc = subprocess.run(
            [conda_python, edelweissfe_bin, "test_scaling.inp"],
            cwd=run_dir,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

    elapsed_time = time.perf_counter() - start_time

    with open(log_file_path, "r") as f:
        log_content = f.read()

    success = proc.returncode == 0 or "Reached maximum number of increments" in log_content

    print(f"     Finished in {elapsed_time:.2f} seconds (Status: {'SUCCESS' if success else 'FAILED'})", flush=True)

    results.append(
        {"threads": threads, "total_wall_time": elapsed_time, "return_code": proc.returncode, "success": success}
    )

base_time = results[0]["total_wall_time"] if results else 1.0

print("\n==========================================================================", flush=True)
print("Thread Scaling Results Summary", flush=True)
print("==========================================================================", flush=True)
print(
    f"{'Threads':<10} | {'Total Time (s)':<16} | {'Speedup (vs 4T)':<18} | {'Efficiency (%)':<16} | {'Status':<10}",
    flush=True,
)
print("-" * 80, flush=True)

summary_md = "# Thread Scaling Study: AnchorPryOut (Optimized Flags)\n\n"
summary_md += "Model: `EdelweissFE/examples/AnchorPryOut` (Step 2, max 10 increments)\n"
summary_md += "Environment: Conda `next_v2611`, Python 3.14 (free-threaded), AMGCL + OpenMP CSR v2 (C++20, -O3, -march=native)\n\n"
summary_md += "| Threads | Total Wall Time (s) | Speedup (vs 4T) | Parallel Efficiency (%) | Status |\n"
summary_md += "| :---: | :---: | :---: | :---: | :---: |\n"

for res in results:
    threads = res["threads"]
    t_time = res["total_wall_time"]
    speedup = base_time / t_time if t_time > 0 else 0
    efficiency = (speedup / (threads / 4.0)) * 100.0 if threads > 0 else 0
    status_str = "SUCCESS" if res["success"] else "FAILED"

    print(f"{threads:<10} | {t_time:<16.2f} | {speedup:<18.2f} | {efficiency:<16.1f} | {status_str:<10}", flush=True)
    summary_md += f"| {threads} | {t_time:.2f} s | {speedup:.2f}x | {efficiency:.1f}% | {status_str} |\n"

with open(os.path.join(base_dir, "SCALING_RESULTS.md"), "w") as f:
    f.write(summary_md)

print("==========================================================================", flush=True)
