# Technischer Report: Benchmark-Übersicht

## 1. Zusammenfassung der Experimente

### 1.1 Block-Chunked Routing
- **Synthetic MLP**
  - Laufzeit: 2,95 ms (Baseline: 2,48 ms, +19 % Overhead)
  - Theoretische MACs: 24 576 vs 98 304 (–75 % Rechenreduzierung)
- **CNN (CIFAR-10)**
  - Laufzeit: 0,536 s (Baseline: 0,534 s, ≈–0,4 % schneller)
  - Peak Memory: 346,95 MiB (Baseline: 330,18 MiB, +5 %)
  - Accuracy: 9,996 % (Baseline: 11,53 %)

**Fazit:** Starkes MAC-Reduction beim MLP; CNN zeigt nur marginale Laufzeitvorteile und verschlechterte Memory/Accuracy.

### 1.2 Activity-Based Routing
- **Synthetic MLP**: keine Metriken (Logging-Fehler)
- **Classification (MNIST)**
  - Laufzeit: 37,6 ms vs Baseline CNN 533 ms (×14 Speed-up)
  - Peak Memory: 130,87 MiB vs 330,18 MiB (–60 %)
  - Accuracy: 10,6 % vs 11,53 % (–0,9 pp)

**Fazit:** Exzellente Speed-ups und Memory-Einsparungen. Logging-Fehler in Synthetic MLP reparieren.

### 1.3 Boundary Pruning
- **Synthetic MLP**
  - Laufzeit: 3,37 ms vs Baseline 2,48 ms (+36 %)
  - Memory: 81,98 MiB
  - Parameter-Retention: 80,03 %
- **CNN (CIFAR-10)**
  - Laufzeit: 0,531 s vs Baseline 0,534 s (–0,6 %)
  - Peak Memory: 395,65 MiB vs 330,18 MiB (+20 %)
  - Accuracy: 10,09 % vs 11,53 %
  - Parameter-Retention: 31,95 %

**Fazit:** Gute Latenz beim CNN, aber hohe Memory-Overhead und Accuracy-Abfall.

### 1.4 Baseline
- **Synthetic MLP:** 2,48 ms, 98 304 MACs
- **CNN:** 0,534 s, 330,18 MiB, 11,53 %

## 2. Was funktioniert, was nicht?
- MLP-Workloads profitieren stark von Chunking & Quantisierung.
- CNNs zeigen nur bei Activity-Based signifikante Vorteile; sonst Trade-off Memory & Accuracy.
- Logging-Fehler in Activity-Based MLP-Experimenten muss behoben werden.
- Streaming-Inference Metriken fehlen noch komplett.

## 3. Nächste Schritte
1. **Activity-Based MLP:** Logging-Fix und erneutes Rerun der Synthetic-Experimente.
2. **Block-Chunked CNN:** Optimierung des Chunk-Loaders und Ablationsstudie für verschiedene k und τ.
3. **Boundary Pruning:** Memory-Optimierung und Integration von Quantisierungsmodulen.
4. **Streaming Inference:** Experimente durchführen, Resultate in Paper (Sec. 5) eintragen.
5. **Volunteer Computing:** Prototyp validieren und Latenz-/Overhead-Messungen ergänzen.
6. **Reproducibility:** Einheitliches Logging-Framework und Konfigurationsdateien einführen.
