#!/usr/bin/env python3
"""
Synthetic Benchmark for Block-Chunked Routing
Measures inference time, peak memory, and theoretical MACs
"""
import argparse
import os
import sys
# ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import time
import csv
from memory_profiler import memory_usage
import torch
import torch.nn as nn
from nadoo_algorithmen.datasets import synthetic_gaussian_loader  # synthetic data loader
from nadoo_algorithmen.block_chunked import BlockChunkedRouting


def parse_args():
    p = argparse.ArgumentParser(description="Synthetic MLP Block-Chunked Benchmark")
    p.add_argument('--layers', type=int, default=6)
    p.add_argument('--in-dim', type=int, default=128)
    p.add_argument('--out-dim', type=int, default=128)
    p.add_argument('--chunks', type=int, default=12)
    p.add_argument('--topk', type=int, default=3)
    p.add_argument('--runs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--output', type=str, default='synthetic_results.csv')
    return p.parse_args()


class SyntheticModel(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        out = x
        for block in self.blocks:
            out, _ = block(out)
        return out


def build_model(layers, in_dim, out_dim, chunks, topk):
    dims = [in_dim] + [out_dim] * layers
    blocks = []
    for i in range(layers):
        blocks.append(BlockChunkedRouting(dims[i], dims[i+1], num_chunks=chunks, top_k=topk))
    return SyntheticModel(blocks)


def measure(model, x):
    # measure time
    t0 = time.perf_counter()
    out = model(x)
    t1 = time.perf_counter()
    # measure memory usage during forward
    def forward():
        return model(x)
    mem_list = memory_usage((forward, ()), interval=0.01, retval=False)
    peak_mem = max(mem_list)
    return (t1 - t0), peak_mem


def main():
    args = parse_args()
    model = build_model(args.layers, args.in_dim, args.out_dim, args.chunks, args.topk)
    model.eval()
    # prepare synthetic dataset loader and get one batch
    loader = synthetic_gaussian_loader(n_samples=args.runs * args.batch_size,
                                      D=args.in_dim,
                                      n_classes=args.out_dim,
                                      batch_size=args.batch_size)
    x, _ = next(iter(loader))
    # theoretical MACs
    macs_per_layer = args.batch_size * args.in_dim * (args.out_dim / args.chunks) * args.topk
    total_macs = macs_per_layer * args.layers

    # run experiments
    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['run_id', 'time_s', 'peak_mem_mb', 'theoretical_macs'])
        for run in range(1, args.runs + 1):
            time_s, peak_mem = measure(model, x)
            writer.writerow([run, f"{time_s:.6f}", f"{peak_mem:.2f}", int(total_macs)])
            print(f"Run {run}: time={time_s:.4f}s, mem={peak_mem:.2f} MiB, macs={int(total_macs)}")

if __name__ == '__main__':
    main()
