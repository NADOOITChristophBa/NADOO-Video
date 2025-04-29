# Experimental Protocol for Asynchronous Distributed Inference

This document describes the detailed steps to reproduce all experiments reported in the Async Distributed Inference paper.

## 1. Environment Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/ChristophBackhaus/NADOO-Video.git
   cd NADOO-Video
   ```
2. Create a Python environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Ensure PyTorch detects available devices (CPU/GPU):
   ```python
   import torch; print(torch.cuda.device_count(), torch.cuda.is_available())
   ```

## 2. Synthetic Benchmark
Evaluate raw throughput and latency on a dummy model.

1. Navigate to experiments folder:
   ```bash
   cd agenten/nadoo_algorithmen/async_distributed_paper/experiments
   ```
2. Run synthetic benchmark script:
   ```bash
   python run_async_synthetic.py --tasks 1000 --batch-size 32 --devices 1 2 4 8 --output synthetic_results.csv
   ```
3. This script:
   - Creates a dummy PyTorch model that multiplies input by 2.
   - Submits `tasks` independent inputs in parallel to N workers.
   - Logs throughput (tasks/sec) and average latency (ms) in `synthetic_results.csv`.

## 3. Real Model Benchmark
Measure performance on a real image classification model (ResNet18 on CIFAR-10).

1. Download CIFAR-10:
   ```bash
   python data/prepare_cifar10.py
   ```
2. Run real-model benchmark:
   ```bash
   python run_async_resnet.py --dataset cifar10 --batch-size 64 --devices 1 2 4 --output realmodel_results.csv
   ```
3. Logs `throughput, latency` per device count in CSV.

## 4. Data Analysis
1. Launch Jupyter notebook for plotting:
   ```bash
   jupyter notebook analysis/plot_async_results.ipynb
   ```
2. The notebook reads `synthetic_results.csv` and `realmodel_results.csv`, generating figures:
   - Throughput vs. number of devices
   - Latency distribution

All scripts and logs are version-controlled under `async_distributed_paper/experiments`. For detailed code, see `run_async_synthetic.py` and `run_async_resnet.py`.
