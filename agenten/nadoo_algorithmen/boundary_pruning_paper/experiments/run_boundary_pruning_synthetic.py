#!/usr/bin/env python3
"""
Synthetic Benchmark for Boundary-Based Dynamic Pruning & On-the-Fly Quantization
Measures inference time, peak memory, and remaining-parameter ratio on random Gaussian data.
"""
import argparse
import os
import sys
import time
import csv
from memory_profiler import memory_usage
import torch
import torch.nn as nn

# allow import from project
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from nadoo_algorithmen.datasets import synthetic_gaussian_loader
from nadoo_algorithmen.boundary_pruning import BoundaryPruner


def parse_args():
    p = argparse.ArgumentParser(description="Synthetic Boundary Pruning Benchmark")
    p.add_argument('--layers', type=int, default=6)
    p.add_argument('--in-dim', type=int, default=128)
    p.add_argument('--out-dim', type=int, default=128)
    p.add_argument('--threshold', type=float, default=0.2, help='Pruning threshold fraction')
    p.add_argument('--runs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--output', type=str, default='synthetic_pruning_results.csv')
    return p.parse_args()


def build_model(layers, in_dim, out_dim):
    dims = [in_dim] + [out_dim] * layers
    modules = []
    for i in range(layers):
        modules.append(nn.Linear(dims[i], dims[i+1]))
        modules.append(nn.ReLU())
    return nn.Sequential(*modules)


def measure(model, loader, device):
    model.to(device).eval()
    # single-batch measurement
    x, _ = next(iter(loader))
    x = x.to(device)
    def forward():
        return model(x)
    # time
    t0 = time.perf_counter()
    _ = model(x)
    t1 = time.perf_counter()
    # memory
    mem_list = memory_usage((forward, ()), interval=0.01, retval=False)
    peak_mem = max(mem_list)
    # remaining parameters ratio
    total = 0
    nonzero = 0
    for param in model.parameters():
        total += param.numel()
        nonzero += param.data.nonzero().size(0)
    ratio = nonzero / total if total > 0 else 0
    return (t1 - t0), peak_mem, ratio


def main():
    args = parse_args()
    # data loader
    loader = synthetic_gaussian_loader(n_samples=args.runs * args.batch_size,
                                       D=args.in_dim,
                                       n_classes=args.out_dim,
                                       batch_size=args.batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['run_id', 'time_s', 'peak_mem_mb', 'param_ratio'])
        for run in range(1, args.runs + 1):
            model = build_model(args.layers, args.in_dim, args.out_dim)
            pruner = BoundaryPruner(model)
            pruner.prune(args.threshold)
            time_s, peak_mem, ratio = measure(model, loader, device)
            writer.writerow([run, f"{time_s:.6f}", f"{peak_mem:.2f}", f"{ratio:.4f}"])
            print(f"Run {run}: time={time_s:.4f}s, mem={peak_mem:.2f} MiB, ratio={ratio:.2%}")

if __name__ == '__main__':
    main()
