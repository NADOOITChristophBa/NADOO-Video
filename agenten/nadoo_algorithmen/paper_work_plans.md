# Paper Development Work Plans

This document outlines detailed, iterative work plans for finalizing the three scientific papers:

## 1. Asynchronous Distributed Inference
- **Introduction:** Add citations on distributed inference and asynchronous systems.
- **Related Work:** Survey synchronous vs asynchronous frameworks.
- **Method:** Formalize algorithm pseudocode and complexity analysis.
- **Experimental Setup:** Define datasets (CIFAR-10, ImageNet), hardware specs (CPU/GPU types), software environment (PyTorch, Python).
- **Results:** Run benchmarks, collect throughput and latency tables, plot throughput vs number of devices.
- **Discussion:** Analyze scalability and limitations.
- **Conclusion & Abstract Refinement:** Update summary with real numbers.

## 2. Dynamic Activity-Based Routing
- **Introduction & Related Work:** Include conditional computation references.
- **Method:** Detail threshold selection strategies and activity metrics.
- **Experimental Setup:** Select benchmarks (image classification, NLP), hardware specs.
- **Results:** Measure % of blocks skipped, speedup, accuracy trade-offs.
- **Discussion:** Evaluate threshold sensitivity and overhead.
- **Conclusion & Abstract Update:** Incorporate measured results.

## 3. Block-Chunked Routing & Dynamic Quantization
- **Introduction & Related Work:** Cite block-sparse and quantization papers.
- **Method:** Formalize chunk splitting, top-k selection, quantization bit-width policy.
- **Experimental Setup:** Choose large linear and conv models, hardware, dataset.
- **Results:** Collect memory savings, computation reduction, accuracy metrics.
- **Discussion:** Discuss hardware friendliness and trade-offs.
- **Conclusion & Abstract Revision:** Summarize final gains.

---

Once the plan is approved, we will execute each step in order, updating the corresponding paper draft with real figures, tables, and citations.
