#!/usr/bin/env python3
"""
CIFAR-10 CNN Benchmark for Boundary-Based Dynamic Pruning & On-the-Fly Quantization
Measures inference time, peak memory, accuracy, and remaining-parameter ratio.
"""
import argparse, os, sys, time, csv
from memory_profiler import memory_usage
import torch
import torch.nn as nn
import torchvision.models as models

# allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from nadoo_algorithmen.datasets import get_loaders
from nadoo_algorithmen.boundary_pruning import BoundaryPruner


def parse_args():
    p = argparse.ArgumentParser(description="CNN Boundary Pruning Benchmark")
    p.add_argument('--dataset', type=str, default='cifar10', help='Dataset name (mnist, cifar10, etc.)')
    p.add_argument('--threshold', type=float, default=0.2, help='Pruning threshold fraction')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--runs', type=int, default=5)
    p.add_argument('--output', type=str, default='real_pruning_results.csv')
    return p.parse_args()


def get_model(num_classes):
    return models.resnet18(pretrained=False, num_classes=num_classes)


def compute_param_ratio(model):
    total = sum(p.numel() for p in model.parameters())
    nonzero = sum(p.data.nonzero().size(0) for p in model.parameters())
    return nonzero / total if total > 0 else 0


def measure(model, loader, device):
    model.to(device).eval()
    total_correct = 0
    total_samples = 0
    times = []
    mems = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            def forward():
                return model(images)
            t0 = time.perf_counter()
            out = model(images)
            t1 = time.perf_counter()
            mem_list = memory_usage((forward, ()), interval=0.01, retval=False)
            times.append(t1 - t0)
            mems.append(max(mem_list))
            preds = out.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
    return sum(times)/len(times), max(mems), total_correct/total_samples


def main():
    args = parse_args()
    # Load data
    _, test_loader = get_loaders(args.dataset, batch_size=args.batch_size)
    # Determine number of classes
    ds = test_loader.dataset
    if hasattr(ds, 'classes'):
        num_classes = len(ds.classes)
    else:
        mapping = {'mnist':10, 'fashion_mnist':10, 'cifar10':10, 'cifar100':100, 'svhn':10, 'tiny_imagenet':200}
        num_classes = mapping.get(args.dataset, 10)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['run_id','time_s','peak_mem_mb','accuracy','param_ratio'])
        for run in range(1, args.runs + 1):
            model = get_model(num_classes)
            pruner = BoundaryPruner(model)
            pruner.prune(args.threshold)
            param_ratio = compute_param_ratio(model)
            time_s, peak_mem, acc = measure(model, test_loader, device)
            writer.writerow([run, f"{time_s:.4f}", f"{peak_mem:.2f}", f"{acc:.4f}", f"{param_ratio:.4f}"])
            print(f"Run {run}: time={time_s:.4f}s, mem={peak_mem:.2f} MiB, acc={acc:.2%}, ratio={param_ratio:.2%}")

if __name__ == '__main__':
    main()
