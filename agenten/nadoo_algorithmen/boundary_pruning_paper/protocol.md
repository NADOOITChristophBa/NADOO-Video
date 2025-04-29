# Experimental Protocol for Boundary-Based Dynamic Pruning & On-the-Fly Quantization

This document outlines the steps to reproduce all experiments reported in the Boundary-Based Dynamic Pruning paper.

## 1. Environment Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/ChristophBackhaus/NADOO-Video.git
   cd NADOO-Video
   ```
2. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 2. Synthetic Benchmark
Partition a simple MLP and apply boundary-based dynamic pruning and on-the-fly quantization:
```bash
cd agenten/nadoo_algorithmen/boundary_pruning_paper/experiments
python run_boundary_pruning_synthetic.py \
  --layers 6 --chunks 12 --quant-thresh 0.2 --runs 5 \
  --output synthetic_pruning_results.csv
```
Logged metrics: multiply-add count, inference time, peak memory usage.

## 3. Real-Model Benchmark
Evaluate on CIFAR-10 with a small CNN:
```bash
cd agenten/nadoo_algorithmen/boundary_pruning_paper/experiments
python run_boundary_pruning_cnn.py \
  --dataset cifar10 --chunks 12 --topk 3 \
  --quant-thresh 0.2 --runs 5 \
  --output real_pruning_results.csv
```
Logged metrics: accuracy, inference time, peak memory usage.

## 4. Data Analysis
Open the analysis notebook to generate figures and tables:
```bash
jupyter notebook analysis/plot_boundary_pruning_results.ipynb
```
This notebook reads `synthetic_pruning_results.csv` and `real_pruning_results.csv` to produce all visuals in the paper.

All experiment scripts and logs are located under `agenten/nadoo_algorithmen/boundary_pruning_paper/experiments`.
