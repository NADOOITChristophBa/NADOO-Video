#!/usr/bin/env python3
"""
Synthetic Baseline Benchmark for Dense MLP
Measures inference time, peak memory, and theoretical MACs on random Gaussian data.
"""
import argparse, os, sys, time, csv
from memory_profiler import memory_usage
import torch
import torch.nn as nn

# allow import from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from nadoo_algorithmen.datasets import synthetic_gaussian_loader


def parse_args():
    p = argparse.ArgumentParser(description="Synthetic Baseline MLP Benchmark")
    p.add_argument('--layers', type=int, default=6)
    p.add_argument('--in-dim', type=int, default=128)
    p.add_argument('--out-dim', type=int, default=128)
    p.add_argument('--runs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--output', type=str, default='baseline_synthetic_results.csv')
    return p.parse_args()


def build_dense_model(layers, in_dim, out_dim):
    dims = [in_dim] + [out_dim] * layers
    modules = []
    for i in range(layers):
        modules.append(nn.Linear(dims[i], dims[i+1]))
        modules.append(nn.ReLU())
    return nn.Sequential(*modules)


def measure(model, x):
    t0 = time.perf_counter()
    _ = model(x)
    t1 = time.perf_counter()
    mem = max(memory_usage((model, (x,)), interval=0.01, retval=False))
    return (t1 - t0), mem


def main():
    args = parse_args()
    model = build_dense_model(args.layers, args.in_dim, args.out_dim)
    model.eval()
    # prepare synthetic data loader and get one batch
    loader = synthetic_gaussian_loader(n_samples=args.runs * args.batch_size,
                                      D=args.in_dim,
                                      n_classes=args.out_dim,
                                      batch_size=args.batch_size)
    x, _ = next(iter(loader))
    macs_per_layer = args.batch_size * args.in_dim * args.out_dim
    total_macs = macs_per_layer * args.layers

    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['run_id', 'time_s', 'peak_mem_mb', 'theoretical_macs'])
        for run in range(1, args.runs+1):
            time_s, mem = measure(model, x)
            writer.writerow([run, f"{time_s:.6f}", f"{mem:.2f}", int(total_macs)])
            print(f"Run {run}: time={time_s:.4f}s, mem={mem:.2f} MiB, macs={int(total_macs)}")

if __name__ == '__main__':
    main()
