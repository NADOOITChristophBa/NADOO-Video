# Nadoo Algorithmen

Diese Sammlung enthält Implementierungen und Paper für effiziente Inferenzmethoden in PyTorch.

## 1. Asynchronous Distributed Inference
- Beschreibung:
  - DE: Eine Methode für asynchrone verteilte Inferenz, die Aufgaben unabhängig auf mehreren Geräten verarbeitet und Ergebnisse über eine Warteschlange sammelt.
  - EN: An asynchronous distributed inference framework that processes tasks independently across devices and collects results via a task queue.
- Modul: `async_distributed.py`
- Paper: [`async_distributed_paper/paper.md`](async_distributed_paper/paper.md)
- Experimente: `async_distributed_paper/experiments/run_async_inference.py`
- Beispiel:
  ```bash
  python3 async_distributed_paper/experiments/run_async_inference.py --devices mac_studio raspberry_pi cortex_m4 --runs 5 --output async_results.csv
  ```

## 2. Activity-Based Routing
- Beschreibung:
  - DE: Eine dynamische Routing-Technik, die basierend auf Aktivitätsmetriken in jedem Block entscheidet, ob dieser berechnet wird, um Rechenaufwand zu reduzieren.
  - EN: A dynamic routing technique that uses activity metrics to decide whether to execute each block, thereby reducing computation.
- Modul: `activity_routing.py`
- Paper: [`activity_routing_paper/paper.md`](activity_routing_paper/paper.md)
- Experimente: `activity_routing_paper/experiments/run_activity_routing_synthetic.py`
- Beispiel:
  ```bash
  python3 activity_routing_paper/experiments/run_activity_routing_synthetic.py --samples 10000 --D 128 --layers 6 --threshold 0.1 --batch-size 64 --runs 5 --output synthetic_activity_results.csv
  ```

## 3. Block-Chunked Routing & Dynamic Quantization
- Beschreibung:
  - DE: Eine Methode, die Modellgewichte in Chunks aufteilt und basierend auf Aktivität nur die wichtigsten Chunks verarbeitet und dynamisch quantisiert.
  - EN: A method that splits model weights into chunks and processes only the top-k chunks based on activity, with on-the-fly quantization.
- Modul: `block_chunked.py`
- Paper: [`block_chunk_quant_paper/paper.md`](block_chunk_quant_paper/paper.md)
- Experimente: `block_chunk_quant_paper/experiments/run_block_chunked_synthetic.py`
- Beispiel:
  ```bash
  python3 block_chunk_quant_paper/experiments/run_block_chunked_synthetic.py --layers 6 --chunks 12 --topk 3 --runs 5 --batch-size 1 --output synthetic_results.csv
  ```

## 4. Boundary-Based Dynamic Pruning & On-the-Fly Quantization
- Beschreibung:
  - DE: Ein Ansatz, der Entscheidungsgrenzen im Aktivierungsraum modelliert, Gewichte unterhalb einer Schwelle dynamisch pruned und die Quantisierung zur Laufzeit anpasst.
  - EN: An approach that models decision boundaries in activation space, dynamically prunes weights below a threshold, and adapts quantization at runtime.
- Modul: `boundary_pruning.py`
- Paper: [`boundary_pruning_quant_paper/paper.md`](boundary_pruning_quant_paper/paper.md)
- Experimente: `boundary_pruning_paper/experiments/run_boundary_pruning_synthetic.py`
- Beispiel:
  ```bash
  python3 boundary_pruning_paper/experiments/run_boundary_pruning_synthetic.py --layers 6 --in-dim 128 --out-dim 128 --threshold 0.2 --runs 5 --batch-size 64 --output synthetic_pruning_results.csv
  ```

## 5. Dynamic Quantized Linear Layer
- Beschreibung:
  - DE: Eine dynamisch quantisierte lineare Schicht, die Präzision basierend auf Aktivitätsmetriken anpasst.
  - EN: A dynamic quantized linear layer that adapts precision based on activity metrics.
- Modul: `dynamic_quantization.py`
- Beschreibung: Dynamisch quantisierte lineare Schicht basierend auf Aktivitätsmetriken.
- Verwendung:
  ```python
  from nadoo_algorithmen.dynamic_quantization import DynamicQuantizedLinear
  ```

## 6. Mathematische Formeln & Tests
- Modul: `formulas.py`
- Tests: `tests/test_formulas.py`
- Beschreibung:
  - DE: Sammlung mathematischer Funktionen für Aktivitäts-Score, Komplexitätsanalyse, Pruning-Schwellen und Chunk-Auswahl. Validierung via pytest.
  - EN: Collection of mathematical functions for activity score, complexity analysis, pruning threshold, and chunk selection. Validated with pytest.
- Ausführen:
  ```bash
  pip install -r requirements.txt
  cd agenten/nadoo_algorithmen
  pytest tests/test_formulas.py
  ```

## Daten & Demos
- `datasets.py` (siehe `datasets.md`)
- Demo: `model_combo_demo.py` (kombiniert alle Algorithmen)

## Work Plans
- `paper_work_plans.md`

## Installation
```bash
pip install -r ../../requirements.txt
```

## Experimente ausführen
```bash
python3 ../../run_all_experiments.py
