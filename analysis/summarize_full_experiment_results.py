#!/usr/bin/env python3
"""
Full Summary of Experiment Results
Generates Markdown tables with mean, std, and median for each metric in the result CSVs.
"""
import pandas as pd
import os


def summarize_csv(path, title):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    df = pd.read_csv(path)
    metrics = [c for c in df.columns if c != 'run_id']
    stats = df[metrics].agg(['mean','std','median'])
    print(f"## {title}\n")
    print("| Metric | Mean | Std Dev | Median |")
    print("|---|---|---|---|")
    for m in metrics:
        print(f"| {m} | {stats.loc['mean', m]:.6f} | {stats.loc['std', m]:.6f} | {stats.loc['median', m]:.6f} |")
    print("\n")


if __name__ == '__main__':
    base = os.path.abspath(os.path.dirname(__file__))
    experiments = [
        ('../agenten/nadoo_algorithmen/block_chunk_quant_paper/experiments/synthetic_results.csv', 'Block-Chunked Synthetic MLP'),
        ('../agenten/nadoo_algorithmen/block_chunk_quant_paper/experiments/cnn_results.csv', 'Block-Chunked CNN (CIFAR-10)'),
        ('../agenten/nadoo_algorithmen/activity_routing_paper/experiments/synthetic_activity_results.csv', 'Activity-Routing Synthetic MLP'),
        ('../agenten/nadoo_algorithmen/activity_routing_paper/experiments/real_results.csv', 'Activity-Routing Classification'),
        ('../agenten/nadoo_algorithmen/boundary_pruning_paper/experiments/synthetic_pruning_results.csv', 'Boundary-Pruning Synthetic MLP'),
        ('../agenten/nadoo_algorithmen/boundary_pruning_paper/experiments/real_pruning_results.csv', 'Boundary-Pruning CNN (CIFAR-10)'),
        ('../agenten/nadoo_algorithmen/block_chunk_quant_paper/experiments/baseline_synthetic_results.csv', 'Baseline Synthetic MLP'),
        ('../agenten/nadoo_algorithmen/block_chunk_quant_paper/experiments/baseline_cnn_results.csv', 'Baseline CNN (CIFAR-10)'),
    ]
    for rel, title in experiments:
        csv_path = os.path.join(base, rel)
        summarize_csv(csv_path, title)
