# Experiment Plan for Block-Chunked Routing & Dynamic Quantization

This document captures my planning notes and steps for the experiments behind our paper.

## Objectives
- Validate theoretical computation reduction $T_{comb}=B\times k\times t$ on a synthetic MLP.
- Measure peak memory usage and multiply-add reduction across varying $k$.
- Benchmark accuracy and efficiency on a small CNN over CIFAR-10.

## Synthetic Benchmark
- **Model:** MLP with 6 hidden layers
- **Dimensions:** in=128, out=128 per layer
- **Chunking:** $C=12$, top-$k$ in \{2,3,4\}
- **Runs:** 5 runs per configuration, seed=42
- **Metrics:** inference time (`time.perf_counter()`), peak RSS (`memory-profiler`), MACs count
- **Output:** CSV with columns `run_id,k,time_s,peak_mem_mb,macs`

## Real-Model Benchmark
- **Model:** Simple CNN on CIFAR-10 (flattened convs via Block-Chunked layers)
- **Chunking:** same $C,k$ settings
- **Quantization Policy:** $(q_{high},q_{mid},q_{low})=(32,8,1)$, thresholds $\tau=0.2,0.5$
- **Runs:** 5 runs each
- **Metrics:** test accuracy, inference time, peak RSS
- **Output:** CSV `cnn_results.csv`

## Next Steps
1. Implement `run_block_chunked_synthetic.py` for MLP.
2. Run experiments locally, gather `synthetic_results.csv`.
3. Implement CNN script `run_block_chunked_cnn.py`.
4. Execute CIFAR-10 experiments, collect `cnn_results.csv`.
5. Generate plots in `analysis/plot_block_chunked_quant_results.ipynb`.

*Notes:* Ensuring reproducibility: fixed random seeds, clear logs, and environment specs as in paper.
