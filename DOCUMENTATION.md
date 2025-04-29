# NADOO-Video Experiment Framework Documentation

This document provides a comprehensive overview of the NADOO-Video project, including algorithms, papers, experiments, analysis, and utilities.

## Repository Structure
```
NADOO-Video/
├── run_all_experiments.py        # Orchestrates end-to-end experiments
├── requirements.txt              # Python dependencies
├── agenten/nadoo_algorithmen/    # Algorithm implementations & papers
│   ├── README.md                 # Overview of modules & experiments
│   ├── async_distributed_paper/  # Paper + experiments for async inference
│   ├── activity_routing_paper/   # Paper + synthetic & classification benchmarks
│   ├── block_chunk_quant_paper/  # Paper + synthetic & CNN benchmarks
│   ├── boundary_pruning_paper/   # Paper + synthetic & CNN benchmarks
│   ├── dynamic_quantization.py   # Dynamic quantized linear layer
│   ├── formulas.py               # Mathematical functions (activity, complexity, pruning, chunk selection)
│   └── tests/                    # Pytest for formulas (`test_formulas.py`)
├── analysis/                     # Result summarization and plots
│   ├── summarize_experiment_results.py    # Mean metrics for papers
│   ├── summarize_full_experiment_results.py  # Mean, std, median tables
│   ├── plot_activity_routing_results.ipynb
│   ├── plot_block_chunked_quant_results.ipynb
│   └── plot_boundary_pruning_results.ipynb
└── docs/ (optional)             # Additional documentation or design notes
```

## Installation
```bash
pip install -r requirements.txt
```

## Running Experiments
To execute all benchmarks and summarize metrics:
```bash
python3 run_all_experiments.py
```
This will launch all experiments (synthetic & real, dynamic vs. baseline), show a progress bar, and generate summary outputs under `analysis/`.

## Algorithm Modules & Papers
A detailed module-level overview is in `agenten/nadoo_algorithmen/README.md`. In brief:
- **Asynchronous Distributed Inference**: `async_distributed.py` + paper + `run_async_inference.py`
- **Activity-Based Routing**: `activity_routing.py` + paper + synthetic & classification scripts
- **Block-Chunked Routing**: `block_chunked.py` + paper + experiments
- **Boundary-Based Pruning**: `boundary_pruning.py` + paper + experiments
- **Dynamic Quantization**: `dynamic_quantization.py`
- **Formulas & Tests**: `formulas.py` + `tests/test_formulas.py`

## Analysis & Visualization
- **summarize_experiment_results.py**: quick means for placeholder metrics
- **summarize_full_experiment_results.py**: full stats (mean, std, median)
- **Notebooks**: interactive plots in `analysis/*.ipynb`

## Testing
From project root:
```bash
cd agenten/nadoo_algorithmen
pytest tests/test_formulas.py
```

## Formulas & Validation
Mathematical foundations for:
- Activity score: `A(x)=mean(|x|)`
- Complexity: `C_worst=O(B·d²)`, `C_expected=(1−s)·B·d²+B·d`
- Pruning threshold and mask
- Chunk top-k selection

Validated via pytest in `tests/test_formulas.py`.

## Reporting Results
After experiments complete, run:
```bash
python3 analysis/summarize_full_experiment_results.py
```
This prints Markdown tables with mean, std, median for each metric.

---
For further questions or modifications, see `docs/` or open an issue.
