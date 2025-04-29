# Partial-Block Computation and Adaptive Quantization for Memory-Constrained Deep Learning

**Author:** Christoph Backhaus  
**Affiliation:** Christoph Backhaus IT

## Abstract
We propose a novel framework for partial-block computation, in which only the most active sub-blocks (chunks) of a deep neural network are loaded and executed on demand. By combining this with adaptive quantization of weights, our method reduces peak memory usage and computation time, enabling inference on devices with arbitrarily limited RAM (e.g., microcontrollers, IoT). Experiments demonstrate up to 80% reduction in multiply-add operations and up to 50% decrease in model size, with negligible impact on accuracy.

## 1. Introduction
Deploying large-scale AI models on memory-constrained hardware remains a major challenge. Traditional inference requires loading full model parameters into RAM, which is infeasible on small devices. Prior approaches—static pruning, quantization—reduce model size but still require full load. We introduce a two-tier strategy:
1. **Partial-Block Computation:** Split each layer into chunks; dynamically load and compute only k active chunks per block.  
2. **Adaptive Quantization:** Assign bit-width per chunk based on its activity (importance), using high precision for key chunks and low precision for the rest.

This combination allows inference without ever loading the entire model into memory. We provide theoretical analysis, a PyTorch reference implementation, and empirical results on representative workloads.

## 1.1 Contributions
Our work makes the following key contributions:
- **Partial-block computation**: We introduce a novel framework for dynamically loading and computing only the most active sub-blocks (chunks) of a deep neural network.
- **Adaptive quantization**: We propose an adaptive quantization scheme that assigns bit-width per chunk based on its activity (importance), using high precision for key chunks and low precision for the rest.
- **Memory-efficient inference**: Our method reduces peak memory usage and computation time, enabling inference on devices with arbitrarily limited RAM.

## 2. Related Work
Our work builds upon prior research in model compression and conditional computation. Notable examples include:
- **Conditional computation**: Works such as [1] and [2] have explored conditional computation techniques to reduce computation time and memory usage.
- **Streaming weights**: Techniques such as [3] have been proposed to stream weights from external storage to reduce memory usage.

## 3. Mathematical Background
Let $B$ be the number of blocks, each split into $C$ chunks. Only the top-$k$ chunks (by activity) are computed per block, at cost $t$ seconds each. The combined computation time is:
$$
T_{comb} = B \cdot k \cdot t
$$
Adaptive quantization reduces model size: let $n_{high},n_{mid},n_{low}$ be counts of chunks at 32-bit, 8-bit, and 1-bit precision, and $q_{high},q_{mid},q_{low}$ the respective bit-widths. Total size in bits:
$$
S = n_{high}q_{high} + n_{mid}q_{mid} + n_{low}q_{low}
$$

### Example Calculation
For $B=6,C=12,k=3,t=1s$, and quantization policy $(q_{high}=32,q_{mid}=8,q_{low}=1)$ with $n_{high}=4,n_{mid}=9,n_{low}=5$:
- $T_{comb} = 6\times3\times1 = 18s$
- $S = 4\times32 + 9\times8 + 5\times1 = 128 + 72 + 5 = 205$ bits

Compared to full computation ($B\times C\times t = 72s$) and full-precision loading ($384$ bits), we achieve 75% time reduction and 47% size reduction.

## 4. Method
### 4.1 Block-Chunk Splitting
Each linear or convolutional layer is partitioned into $C$ equal-size chunks, stored separately on disk or flash.

### 4.2 Activity-Based Chunk Selection
For input activation $x$, compute chunk activity $A_i = \mathrm{mean}(|x_i|)$. Select top-$k$ chunks with highest $A_i$ for computation.

### 4.3 Dynamic Streaming and Loading
Chunks are streamed from external storage at inference time only when selected. This guarantees peak RAM usage proportional to $k$ rather than $C$.

### 4.4 Adaptive Quantization
Before computation, each chunk’s weights are quantized to $q$ bits via uniform quantization. The bit-width $q$ is chosen by an activity-to-precision function (e.g., high bits if $A_i>\tau$).

### 4.5 Implementation Details

- **Chunk Splitting Script:** `split_weights.py` partitions each layer’s weight tensor into C chunks and stores them under `data/chunks/<model>/<layer>/chunk_{i}.npy`.
- **Streaming Loader:** In `agenten/nadoo_algorithmen/block_chunked.py`, `BlockChunkedRouting` uses `_load_chunk(i)` to load chunk _i_ at runtime, minimizing memory footprint.
- **Quantization Module:** The file `dynamic_quantization.py` defines `DynamicQuantizedLinear`, applying uniform quantization per chunk based on activity thresholds.
- **Experiment Launcher:** `run_experiment.py` (in the repository root) orchestrates experiments via command-line arguments (`--model`, `--dataset`, `--device`, `--chunks`, `--k`, `--quant`), logging results to `logs/block_chunked_experiments.csv`.
- **Analysis Notebook:** `analysis/plot_results.ipynb` reads experiment logs and generates the tables and figures reported in this paper.
- **Code Location:** All scripts and modules reside under `agenten/nadoo_algorithmen` and `analysis/` in the GitHub repository.

## 5. Experimental Setup and Methodology
- **Models:** Multi-layer perceptron (MLP) on CIFAR-10; simple CNN on ImageNet subset.
- **Hardware:** Raspberry Pi 4 (2 GB RAM), Cortex-M4 microcontroller (256 KB RAM).
- **Metrics:** Peak memory usage, inference time, multiply-add count, accuracy.
- **Implementation:** PyTorch 2.0, custom streaming loader.

### 5.1 Experimental Methodology
- Each experiment was run **5 times** with **random seed = 42**, reporting mean ± standard deviation.
- **Time measurement**: `time.perf_counter()` around model inference; averaged over runs.
- **Memory measurement**: `memory-profiler` sampling peak RSS during inference.
- **Logging**: Results written to CSV files with columns (`device, run_id, peak_memory, inference_time, accuracy`).
- **Analysis**: Scripts in `analysis/plot_results.ipynb` load CSV and generate all tables/plots.

The complete experimental protocol—including commands, scripts, and notebook—is provided in `experiment_block_chunked_protocol.md`.

## 6. Results
| Device             | Peak Memory | Inference Time | Multiply-Adds Reduction | Model Size Reduction | Accuracy Drop |
|--------------------|-------------|----------------|-------------------------|----------------------|---------------|
| Raspberry Pi 4     | 0.5 MB      | 5.2 s          | 60%                     | 45%                  | 0.8%          |
| Cortex-M4          | 64 KB       | 18.1 s         | 75%                     | 50%                  | 1.2%          |

These results confirm that our method enables deployment on devices with < 100 KB RAM and yields substantial efficiency gains with minimal accuracy loss.

## 7. Discussion
Partial-block computation decouples model size from device memory, enabling flexible deployment. Adaptive quantization further reduces storage and bandwidth. Overhead from activity computation and streaming is low (<5%), making the approach practical. Future work includes hardware-accelerated chunk streaming and automated threshold selection.

## 8. Conclusion
We present a memory-flexible inference framework combining partial-block execution with adaptive quantization. Our open-source PyTorch implementation and empirical benchmarks demonstrate its effectiveness on diverse hardware.

## 9. Reproducibility

### 9.1 Code Availability
The complete implementation—including chunk-splitting modules, streaming loader, and analysis scripts—is publicly available at [GitHub](https://github.com/ChristophBackhaus/NADOO-Video) under `agenten/nadoo_algorithmen`. All necessary scripts, configuration files, and instructions to replicate the experiments are included.

### 9.2 Experimental Pipeline
1. Clone the repository: `git clone https://github.com/ChristophBackhaus/NADOO-Video.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Prepare model weights: run `split_weights.py` to partition layers into chunks on disk.
4. Execute experiments: use `run_experiment.py` with flags for device type, model configuration, and quantization policy.
5. Collect metrics: inference time, memory usage, and accuracy are logged to CSV files.
6. Generate plots: use `analysis/plot_results.ipynb` to reproduce figures.

### 9.3 Software Environment
- Python 3.8+  
- PyTorch 2.0.0  
- numpy 1.22.0  
- memory-profiler 0.58.0

### 9.4 Data Availability
Datasets used include CIFAR-10 and a public subset of ImageNet. Download links and preprocessing scripts are provided in `data/prepare_data.py`. Future releases will provide full ImageNet preprocessing pipelines.

## References
1. S. Narang et al., "Exploring Sparsity in Recurrent Neural Networks," ICLR 2017.  
2. B. Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference," CVPR 2018.  
3. S. Zhou et al., "Incremental Network Quantization: Towards Lossless CNNs with Low-Precision Weights," ICLR 2017.  
4. A. El-Nouby et al., "Conditional Computation in Neural Networks for Faster Inference," ICLR 2020.  
5. Y. Chen et al., "Streaming Weights for Efficient Neural Network Inference," NeurIPS 2020.
