# Experimental Protocol for Dynamic Activity-Based Routing

This document describes the detailed steps to reproduce all experiments reported in the Dynamic Activity-Based Routing paper.

## 1. Environment Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/ChristophBackhaus/NADOO-Video.git
   cd NADOO-Video
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 2. Synthetic Block-Skipping Benchmark
1. Navigate to protocol folder:
   ```bash
   cd agenten/nadoo_algorithmen/activity_routing_paper/experiments
   ```
2. Run synthetic benchmark script:
   ```bash
   python run_activity_routing_synthetic.py --samples 1000 --blocks 10 --thresholds 0.1 0.2 0.5 --output synthetic_results.csv
   ```
3. Metrics logged:
   - Block skip ratio per threshold
   - Inference time (mean ± std)

## 3. Real-Model Classification Benchmark
1. Prepare CIFAR-10 dataset:
   ```bash
   python data/prepare_cifar10.py
   ```
2. Run classification benchmark:
   ```bash
   python run_activity_routing_classification.py --dataset cifar10 --threshold 0.2 --batch-size 64 --runs 5 --output real_results.csv
   ```
3. Metrics logged:
   - Accuracy
   - Speedup vs baseline
   - Blocks computed ratio

## 4. Data Analysis
1. Open the analysis notebook:
   ```bash
   jupyter notebook analysis/plot_activity_routing_results.ipynb
   ```
2. Follow notebook cells to load CSV logs and generate figures:
   - Block skip ratio vs threshold
   - Speedup vs accuracy trade-off

All scripts and logs are stored under `activity_routing_paper/experiments`. Refer to code files for implementation details.
