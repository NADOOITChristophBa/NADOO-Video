#!/usr/bin/env python3
"""
Synthetic Benchmark for Dynamic Activity-Based Routing
Measures inference time, peak memory, and block computation ratio on random Gaussian data.
"""
import argparse
import os
import sys
import time
import csv
from memory_profiler import memory_usage
import torch
import torch.nn as nn
import numpy as np
_pyrapl = False
pyRAPL = None

# allow import from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from nadoo_algorithmen.datasets import synthetic_gaussian_loader
from nadoo_algorithmen.activity_routing import ActivityRoutedBlock


def parse_args():
    p = argparse.ArgumentParser(description="Synthetic Activity Routing Benchmark")
    p.add_argument('--samples', type=int, default=10000, help='Number of synthetic samples')
    p.add_argument('--D', type=int, default=128, help='Input dimension')
    p.add_argument('--layers', type=int, default=6, help='Number of routed blocks')
    p.add_argument('--threshold', type=float, default=0.1, help='Activity threshold')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--runs', type=int, default=5)
    p.add_argument('--output', type=str, default='synthetic_activity_results.csv')
    return p.parse_args()


class SyntheticRoutingModel(nn.Module):
    def __init__(self, layers, D, threshold):
        super().__init__()
        self.blocks = nn.ModuleList([
            ActivityRoutedBlock(D, D, threshold=threshold)
            for _ in range(layers)
        ])
        self.threshold = threshold

    def forward(self, x):
        computed = []
        for block in self.blocks:
            x, act = block(x)
            computed.append(act > self.threshold)
        return x, computed


def measure(model, loader, device):
    model.to(device).eval()
    times, mems, ratios = [], [], []
    if _pyrapl:
        meter = pyRAPL.Measurement('activity_routing')
        meter.begin()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            def forward(): return model(x)
            t0 = time.perf_counter()
            _, comp = model(x)
            t1 = time.perf_counter()
            mem = max(memory_usage((forward, ()), interval=0.01, retval=False))
            ratio = sum(comp)/len(comp)
            times.append(t1 - t0)
            mems.append(mem)
            ratios.append(ratio)
    if _pyrapl:
        meter.end()
        energy_uj = meter.result.pkg + meter.result.dram
        energy_mj = energy_uj / 1000.0
    else:
        energy_mj = 0.0
    arr_t = np.array(times)
    arr_m = np.array(mems)
    arr_r = np.array(ratios)
    metrics = {
        'mean_latency': arr_t.mean(),
        'median_latency': np.percentile(arr_t,50),
        'p99_latency': np.percentile(arr_t,99),
        'latency_std': arr_t.std(),
        'throughput': (len(arr_t)*loader.batch_size) / arr_t.sum(),
        'mem_p50': np.percentile(arr_m,50),
        'mem_p99': np.percentile(arr_m,99),
        'mem_std': arr_m.std(),
        'energy_per_sample': energy_mj / (len(arr_t)*loader.batch_size),
        'skip_q25': np.percentile(arr_r,25),
        'skip_median': np.percentile(arr_r,50),
        'skip_q75': np.percentile(arr_r,75),
    }
    return metrics


def main():
    global _pyrapl, pyRAPL
    args = parse_args()
    # optional energy measurement setup
    try:
        import pyRAPL as _pyrapl_module
        _pyrapl_module.setup()
        pyRAPL = _pyrapl_module
        _pyrapl = True
    except ImportError:
        _pyrapl = False
        print("Warning: pyRAPL not installed. Energy metrics disabled.")
    loader = synthetic_gaussian_loader(n_samples=args.samples,
                                       D=args.D,
                                       n_classes=2,
                                       batch_size=args.batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    metrics_keys = ['mean_latency','median_latency','p99_latency','latency_std','throughput','mem_p50','mem_p99','mem_std','energy_per_sample','skip_q25','skip_median','skip_q75']
    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['run_id'] + metrics_keys)
        for run in range(1, args.runs+1):
            model = SyntheticRoutingModel(args.layers, args.D, args.threshold)
            metrics = measure(model, loader, device)
            row = [run] + [f"{metrics[k]:.6f}" for k in metrics_keys]
            writer.writerow(row)
            print(f"Run {run}: " + ", ".join([f"{k}={metrics[k]:.4f}" for k in metrics_keys]))


if __name__ == '__main__':
    main()
