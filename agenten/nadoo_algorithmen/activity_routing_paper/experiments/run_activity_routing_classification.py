#!/usr/bin/env python3
"""
Classification Benchmark for Dynamic Activity-Based Routing
Measures inference time, peak memory, accuracy, and block-compute ratio on real datasets.
"""
import argparse, os, sys, time, csv
from memory_profiler import memory_usage
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# allow import from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from nadoo_algorithmen.datasets import get_loaders
from nadoo_algorithmen.activity_routing import ActivityRoutedBlock


def parse_args():
    p = argparse.ArgumentParser(description="Activity Routing Classification Benchmark")
    p.add_argument('--dataset', type=str, default='mnist', help='Dataset name: mnist, fashion_mnist, cifar10, etc.')
    p.add_argument('--threshold', type=float, default=0.1, help='Activity threshold')
    p.add_argument('--layers', type=int, default=6, help='Number of routed blocks')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--runs', type=int, default=5)
    p.add_argument('--output', type=str, default='real_results.csv')
    return p.parse_args()


class RoutingClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, layers, threshold):
        super().__init__()
        self.blocks = nn.ModuleList([
            ActivityRoutedBlock(input_dim, input_dim, threshold=threshold)
            for _ in range(layers)
        ])
        self.classifier = nn.Linear(input_dim, num_classes)
        self.threshold = threshold

    def forward(self, x):
        # Flatten input for MLP
        x = x.view(x.size(0), -1)
        skip_count = 0
        total = len(self.blocks)
        for block in self.blocks:
            x, act = block(x)
            if act > block.threshold:
                skip_count += 1
        logits = self.classifier(x)
        skip_ratio = skip_count / total if total>0 else 0
        return logits, skip_ratio


def measure(model, loader, device):
    model.to(device).eval()
    # one-batch measure
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)
    def forward(): return model(images)

    t0 = time.perf_counter()
    logits, skip_ratio = model(images)
    t1 = time.perf_counter()
    mem = max(memory_usage((forward,()), interval=0.01, retval=False))
    preds = logits.argmax(dim=1)
    acc = (preds == labels).float().mean().item()
    return (t1 - t0), mem, acc, skip_ratio


def main():
    args = parse_args()
    # data
    _, loader = get_loaders(args.dataset, batch_size=args.batch_size)
    # determine dims
    sample, _ = next(iter(loader))
    input_dim = sample.view(sample.size(0), -1).size(1)
    ds = loader.dataset
    if hasattr(ds, 'classes'):
        num_classes = len(ds.classes)
    else:
        mapping = {'mnist':10,'fashion_mnist':10,'cifar10':10,'cifar100':100,'svhn':10,'tiny_imagenet':200}
        num_classes = mapping.get(args.dataset, 10)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['run_id','time_s','peak_mem_mb','accuracy','skip_ratio'])
        for run in range(1, args.runs+1):
            model = RoutingClassifier(input_dim, num_classes, args.layers, args.threshold)
            time_s, mem, acc, skip_ratio = measure(model, loader, device)
            writer.writerow([run, f"{time_s:.6f}", f"{mem:.2f}", f"{acc:.4f}", f"{skip_ratio:.4f}"])
            print(f"Run {run}: time={time_s:.4f}s, mem={mem:.2f} MiB, acc={acc:.2%}, skip={skip_ratio:.2%}")

if __name__ == '__main__':
    main()
