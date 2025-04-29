#!/usr/bin/env python3
"""
Orchestrate all benchmark experiments and update paper metrics.
1. Launchs dynamic and baseline scripts
2. Polls for completion
3. Calls summarizer to compute mean metrics
4. Reports metrics for manual insertion
"""
import subprocess, time, os
from pathlib import Path
from tqdm import tqdm

# Project root
ROOT = Path(__file__).parent

# Define experiment commands
EXPERIMENTS = [
    # Block-chunked
    "python3 agenten/nadoo_algorithmen/block_chunk_quant_paper/experiments/run_block_chunked_synthetic.py --layers 6 --chunks 12 --topk 3 --runs 5 --batch-size 1",
    "python3 agenten/nadoo_algorithmen/block_chunk_quant_paper/experiments/run_block_chunked_cnn.py --dataset cifar10 --quant-thresh 0.2 --runs 5 --batch-size 64 --output cnn_results.csv",
    # Activity-routing
    "python3 agenten/nadoo_algorithmen/activity_routing_paper/experiments/run_activity_routing_synthetic.py --samples 10000 --D 128 --layers 6 --threshold 0.1 --batch-size 64 --runs 5",
    "python3 agenten/nadoo_algorithmen/activity_routing_paper/experiments/run_activity_routing_classification.py --dataset mnist --threshold 0.1 --layers 6 --batch-size 64 --runs 5 --output real_results.csv",
    # Boundary-pruning
    "python3 agenten/nadoo_algorithmen/boundary_pruning_paper/experiments/run_boundary_pruning_synthetic.py --layers 6 --in-dim 128 --out-dim 128 --threshold 0.2 --runs 5 --batch-size 64 --output synthetic_pruning_results.csv",
    "python3 agenten/nadoo_algorithmen/boundary_pruning_paper/experiments/run_boundary_pruning_cnn.py --dataset cifar10 --threshold 0.2 --batch-size 64 --runs 5 --output real_pruning_results.csv",
    # Baselines
    "python3 agenten/nadoo_algorithmen/block_chunk_quant_paper/experiments/run_baseline_synthetic.py --layers 6 --in-dim 128 --out-dim 128 --runs 5 --batch-size 1 --output baseline_synthetic_results.csv",
    "python3 agenten/nadoo_algorithmen/block_chunk_quant_paper/experiments/run_baseline_cnn.py --batch-size 64 --runs 5 --output baseline_cnn_results.csv",
]

# Human-readable labels for each experiment
NAMES = [
    "Block-chunked synthetic MLP",
    "Block-chunked CNN (CIFAR-10)",
    "Activity-routing synthetic MLP",
    "Activity-routing real (MNIST)",
    "Boundary-pruning synthetic MLP",
    "Boundary-pruning CNN (CIFAR-10)",
    "Baseline synthetic MLP",
    "Baseline CNN (CIFAR-10)"
]

def run_experiments():
    procs = []
    for idx, (name, cmd) in enumerate(zip(NAMES, EXPERIMENTS), 1):
        print(f"[{idx}/{len(EXPERIMENTS)}] Starting experiment: {name}")
        p = subprocess.Popen(cmd, cwd=ROOT, shell=True)
        procs.append((name, cmd, p))
    # Progress bar
    pbar = tqdm(total=len(procs), desc="Experiments", dynamic_ncols=True)
    completed = set()
    while len(completed) < len(procs):
        for name, cmd, p in procs:
            if p.poll() is not None and name not in completed:
                completed.add(name)
                pbar.update(1)
                tqdm.write(f"[Completed] {name}")
        alive = [(name, p) for name, cmd, p in procs if p.poll() is None]
        alive_names = [n for n, p in alive]
        pbar.set_description(f"Running {len(alive_names)}/{len(procs)}")
        pbar.set_postfix_str(", ".join(alive_names))
        time.sleep(1)
    pbar.close()

if __name__ == '__main__':
    run_experiments()
    print("\n=== Summarizing Results ===")
    subprocess.run(["python3", "analysis/summarize_experiment_results.py"], cwd=ROOT)
    print("Done. Review metrics above and update paper placeholders.")
