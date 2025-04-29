# Notizen: Entwicklung ohne CUDA-Beschleunigung

Dieser Leitfaden fasst Best Practices und Strategien zusammen, um effiziente Experimente und Prototypen auf CPU-basierten Systemen (kein GPU/CUDA) durchzuführen.

## Motivation
- Viele Nutzer haben nur CPU-Ressourcen oder remote Umgebungen ohne CUDA.
- Ziel: maximale Performance und Reproduzierbarkeit auf x86-Servern und Mac Studio.

## 1. Framework- und Sprachwahl
- Verwende PyTorch XLA oder native CPU-Backends mit MKL/OpenMP.
- Nutze `torch.jit.script` / TorchScript, um Graph-Optimierungen und Fusion zu aktivieren.
- Für kritische Kernschleifen: Numba (JIT-Compiler) oder Cython/C++-Extensions (PyBind11).

## 2. Vektorisierung & Kernel-Fusion
- Setze auf batchweise Matrix-Multiplikationen (BLAS/LAPACK). Keine Python-Loops über Tensor-Elemente.
- Aktiviere `torch.backends.mkldnn.enabled=True` und `torch.set_num_threads()` passend zur CPU-Kernzahl.
- Prüfe `torch.backends.openmp` und Umgebungsvariablen `OMP_NUM_THREADS`.

## 3. Parallelisierung
- DataLoader mit `num_workers>0` nutzen; batch-Vorbereitung entkoppeln.
- Python-Multiprocessing für unabhängige Modelle/Parameter-Sweeps.
- Vermeide GIL-Engpässe: reines Tensor-Rechnen in C/C++-Schichten.

## 4. Profiling & Monitoring
- `torch.profiler` (CPU-Modus) für detaillierte Timeline.
- Python `cProfile` / `pyinstrument` ergänzend.
- Betriebsmessungen mit `psutil` und `memory_profiler`.

## 5. Quantisierung & Sparsity
- CPU-Inferenz kann INT8- und FP16-Quantisierung (via `torch.quantization`) stark beschleunigen.
- Nutzen von Block-Pruning und Sparse-Matrizen (SciPy-Sparse + PyTorch-Sparse) für Memory-/Compute-Vorteile.

## 6. Bibliotheken & Tools
- **ONNX Runtime** mit CPU-Optimierungen (OpenVINO, MKL-DNN).
- **TVM** für Graph-Optimierung und Cross-Device-Code-Gen.
- **Intel® Extension for PyTorch**: zusätzlicher Layer für Vektor-Beschleunigung.

## 7. Reproduzierbarkeit
- Fixiere zufällige Seeds (`torch.manual_seed`, `np.random.seed`).
- Doku: CPU-Threading, BLAS-Konfiguration, PyTorch-Version.

## 8. Referenzen
1. Zhang et al., "Efficient CPU Inference with Mixed Precision" (NeurIPS 2023)
2. Kim & Park, "Sparse Linear Algebra on CPUs" (ICLR 2024)
3. PyTorch Docs: TorchScript CPU Optimization
4. Intel PyTorch Extension Guide
