#!/usr/bin/env python3
"""
CIFAR-10 CNN Benchmark for Block-Chunked Routing & Dynamic Quantization
"""
import argparse
import os
import sys
import time
import csv
import torch
import torchvision
import torchvision.transforms as transforms
from memory_profiler import memory_usage
from tqdm import tqdm  # progress bar for visual feedback

# allow import from nadoo_algorithmen
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from nadoo_algorithmen.block_chunked import BlockChunkedRouting
from nadoo_algorithmen.dynamic_quantization import DynamicQuantizedLinear
from nadoo_algorithmen.datasets import get_loaders  # dynamic dataset support
import torch.nn as nn

# Custom head combining chunked routing and dynamic quantization
class ChunkedQuantHead(nn.Module):
    def __init__(self, in_feats, chunks, topk, thresh):
        super().__init__()
        self.chunked = BlockChunkedRouting(in_feats, 10, num_chunks=chunks, top_k=topk)
        self.quant = DynamicQuantizedLinear(10, 10, bits_fn=lambda act: 32 if act > thresh else 1)
    def forward(self, x):
        out, acts = self.chunked(x)
        # use maximum chunk activity for quantization decision
        scalar_act = max(acts) if isinstance(acts, list) else acts
        return self.quant(out, scalar_act)


def parse_args():
    p = argparse.ArgumentParser(description="CIFAR-10 Chunked CNN Benchmark")
    p.add_argument('--chunks', type=int, default=12)
    p.add_argument('--topk', type=int, default=3)
    p.add_argument('--quant-thresh', type=float, default=0.2)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--runs', type=int, default=5)
    p.add_argument('--output', type=str, default='cnn_results.csv')
    p.add_argument('--dataset', type=str, default='cifar10', help='Dataset to use (mnist, cifar10, etc.)')
    return p.parse_args()


def get_model(chunks, topk, thresh):
    # Base model
    model = torchvision.models.resnet18(pretrained=False, num_classes=10)
    in_feats = model.fc.in_features
    # Replace FC with chunked+quant head
    head = ChunkedQuantHead(in_feats, chunks, topk, thresh)
    model.fc = head
    return model


def measure(model, loader, device):
    model.to(device).eval()
    total_correct = 0
    total_samples = 0
    times = []
    mems = []
    with torch.no_grad():
        # show batch-level progress
        for images, labels in tqdm(loader, desc="Batches", leave=False):
            images, labels = images.to(device), labels.to(device)
            def forward():
                out = model(images)
                return out
            # time
            t0 = time.perf_counter()
            out = model(images)
            t1 = time.perf_counter()
            # memory
            mem_list = memory_usage((forward, ()), interval=0.01, retval=False)
            times.append(t1 - t0)
            mems.append(max(mem_list))
            # accuracy
            preds = out.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
    return sum(times)/len(times), max(mems), total_correct/total_samples


def main():
    args = parse_args()
    # Data
    _, loader = get_loaders(args.dataset, batch_size=args.batch_size)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Run experiments
    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['run_id', 'time_s', 'peak_mem_mb', 'accuracy'])
        for run in range(1, args.runs + 1):
            model = get_model(args.chunks, args.topk, args.quant_thresh)
            time_s, peak_mem, acc = measure(model, loader, device)
            writer.writerow([run, f"{time_s:.4f}", f"{peak_mem:.2f}", f"{acc:.4f}"])
            print(f"Run {run}: time={time_s:.4f}s, mem={peak_mem:.2f} MiB, acc={acc:.2%}")

if __name__ == '__main__':
    main()
