#!/usr/bin/env python3
"""
Baseline CNN Benchmark on CIFAR-10 for ResNet18
Measures inference time, peak memory, and accuracy on CIFAR-10.
"""
import argparse, os, sys, time, csv
from memory_profiler import memory_usage
import torch
import torchvision
import torchvision.transforms as transforms

# allow import from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))


def parse_args():
    p = argparse.ArgumentParser(description="Baseline ResNet18 CIFAR-10 Benchmark")
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--runs', type=int, default=5)
    p.add_argument('--output', type=str, default='baseline_cnn_results.csv')
    return p.parse_args()


def measure(model, loader, device):
    model.to(device).eval()
    total_correct, total_samples = 0, 0
    times, mems = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            def forward(): return model(images)
            t0 = time.perf_counter()
            out = model(images)
            t1 = time.perf_counter()
            mems.append(max(memory_usage((forward,()), interval=0.01, retval=False)))
            times.append(t1 - t0)
            preds = out.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
    return sum(times)/len(times), max(mems), total_correct/total_samples


def main():
    args = parse_args()
    # Data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,)*3, (0.5,)*3)])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    # Model
    model = torchvision.models.resnet18(pretrained=False, num_classes=10)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Measure
    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['run_id','time_s','peak_mem_mb','accuracy'])
        for run in range(1, args.runs+1):
            time_s, peak_mem, acc = measure(model, loader, device)
            writer.writerow([run, f"{time_s:.4f}", f"{peak_mem:.2f}", f"{acc:.4f}"])
            print(f"Run {run}: time={time_s:.4f}s, mem={peak_mem:.2f} MiB, acc={acc:.2%}")

if __name__ == '__main__':
    main()
