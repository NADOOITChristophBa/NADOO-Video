# Experimental Protocol for Block-Chunked Routing & Dynamic Quantization

This document describes the detailed steps to reproduce all experiments reported in the Block-Chunked Routing & Dynamic Quantization paper.

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

## 2. Chunk Splitting
Partition each model layer’s weights into chunks:
```bash
python split_weights.py --model <model_name> --chunks <C>
```
Chunks are saved to `data/chunks/<model_name>/<layer>/chunk_<i>.npy`.

## 3. Synthetic MLP Benchmark
Measure computation reduction on a dummy MLP:
```bash
cd agenten/nadoo_algorithmen/block_chunk_quant_paper/experiments
python run_block_chunked_synthetic.py --layers 6 --chunks 12 --topk 3 --runs 5 --output synthetic_results.csv
```
Logs include per-run multiply-add count and inference time.

## 4. Real-Model CNN Benchmark
Evaluate on a small CNN for ImageNet subset:
```bash
python data/prepare_imagenet_subset.py
python run_block_chunked_cnn.py --dataset imagenet_sub --chunks 12 --topk 3 --quant-policy "32,8,1" --runs 5 --output real_cnn_results.csv
```
Metrics logged: inference time, peak memory, accuracy, model size.

## 5. Data Analysis
Open the analysis Jupyter notebook:
```bash
jupyter notebook analysis/plot_block_chunked_quant_results.ipynb
```
This notebook reads `synthetic_results.csv` and `real_cnn_results.csv` to generate all tables and figures in the paper.

All scripts and logs are stored under `block_chunk_quant_paper/experiments`. For implementation details, see the code comments in `run_block_chunked_synthetic.py` and `run_block_chunked_cnn.py`.
