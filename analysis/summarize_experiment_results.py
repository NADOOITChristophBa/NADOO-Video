#!/usr/bin/env python3
"""
Summarize experiment results for block-chunked, activity-routing, and boundary-pruning methods.
Generates mean metrics to help fill placeholders in paper drafts.
"""
import pandas as pd
import os
from pandas.errors import EmptyDataError

def summarize(path, metrics):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return {}
    try:
        df = pd.read_csv(path)
    except EmptyDataError:
        print(f"No data in file: {path}")
        return {}
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return {}
    summary = {}
    for m in metrics:
        if m in df.columns:
            summary[m] = df[m].mean()
        else:
            print(f"Metric '{m}' not found in {path}, available columns: {list(df.columns)}")
    return summary

if __name__ == '__main__':
    print("=== Block-Chunked Routing ===")
    print("Synthetic MLP:", summarize(
        'synthetic_results.csv',
        ['time_s', 'theoretical_macs']
    ))
    print("CNN:", summarize(
        'cnn_results.csv',
        ['time_s', 'peak_mem_mb', 'accuracy']
    ))

    print("\n=== Activity-Based Routing ===")
    print("Synthetic:", summarize(
        'synthetic_activity_results.csv',
        ['time_s', 'peak_mem_mb', 'block_compute_ratio']
    ))
    print("Classification:", summarize(
        'real_results.csv',
        ['time_s', 'peak_mem_mb', 'accuracy', 'skip_ratio']
    ))

    print("\n=== Boundary Pruning ===")
    print("Synthetic:", summarize(
        'synthetic_pruning_results.csv',
        ['time_s', 'peak_mem_mb', 'param_ratio']
    ))
    print("CNN:", summarize(
        'real_pruning_results.csv',
        ['time_s', 'peak_mem_mb', 'accuracy', 'param_ratio']
    ))

    # Baseline comparisons
    print("\n=== Baseline ===")
    print("Synthetic Baseline MLP:", summarize(
        'baseline_synthetic_results.csv',
        ['time_s', 'theoretical_macs']
    ))
    print("Baseline CNN:", summarize(
        'baseline_cnn_results.csv',
        ['time_s', 'peak_mem_mb', 'accuracy']
    ))
