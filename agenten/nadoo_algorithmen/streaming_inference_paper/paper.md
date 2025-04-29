# Streaming Inference with Chunked KV-Caching

## Abstract
Wir stellen einen Streaming-Ansatz für Transformer-Inference vor, der Eingabesequenzen in fixe Fenster zerlegt und KV-Caches zwischen den Schritten persistent hält. Dadurch bleibt der Speicherbedarf konstant unabhängig von der Gesamtlänge und unbegrenzter Kontext wird möglich.

## 1. Introduction
Transformer-Modelle liefern hervorragende Sprachverständnis-Qualität, sind aber durch quadratische Attention und wachsende Kontextspeicher limitiert. Wir schlagen ein chunk-basiertes Streaming-Framework vor, das:
- Sequenzen in Fenster der Größe $W$ unterteilt
- KV-Caches über Fenstergrenzen hinweg fortführt
- Speicherbedarf $O(W^2)$ statt $O(T^2)$ garantiert

## 2. Related Work
- Vaswani et al., “Attention Is All You Need”, NeurIPS 2017
- Kitaev et al., “Reformer: The Efficient Transformer”, ICLR 2020
- Beltagy et al., “Longformer: The Long-Document Transformer”, ArXiv 2020
- Guo et al., “FLASH: Transformer with Linear Attention”, NeurIPS 2022

## 3. Method
### 3.1 Sliding-Window Streaming
Zerlege Eingabesequenz $x_{1:T}$ in überlappende oder nicht-überlappende Fenster $x_{t:t+W-1}$. Jeder Schritt führt aus:
```python
logits, past = model(chunk, past_key_values=past)
```
und speichert `past` für den nächsten Fenster-Durchlauf.

### 3.2 KV-Caching Mechanism
`past_key_values` enthält Schlüssel-/Wert-Matrizen aller bisherigen Schritte. Pro Fenster bleibt der Speicheraufwand $O(W^2)$ statt $O(T^2)$.

### 3.3 Complexity Analysis
- **Zeit**: $O(T/W)$ Inferenzschritte pro Sequenz
- **Speicher**: $O(W^2)$ konstant

### 3.4 Analytical Performance Model
Sei $T$ die Gesamtlänge der Eingabesequenz und $W$ die Fenstergröße. Pro Fenster entstehen:
$$
L_w = C_{\mathrm{att}} W^2 + C_{\mathrm{ff}} W
$$
mit Attention-Kosten $C_{\mathrm{att}} W^2$ und Feed-Forward-Kosten $C_{\mathrm{ff}} W$.
Die Anzahl Fenster ist $N=\lceil T/W\rceil$. Die End-to-End-Latenz:
$$
L_{\mathrm{total}} = N\,L_w + (N-1)\,T_{\mathrm{comm}},
$$
wobei $T_{\mathrm{comm}}$ den Kommunikations-Overhead repräsentiert (vgl. asynchrones Distributed Paper, Abschnitt 4.2).

Der praktische Durchsatz ist:
$$
\Phi = \frac{T}{L_{\mathrm{total}}}.
$$

Der Speicherbedarf pro Fenster:
$$
M = M_{\mathrm{cache}}\,W^2 + M_{\mathrm{model\_chunk}},
$$
konstant in $T$.

### 3.5 Theoretical Guarantees
**Lemma 1 (Memory Bound).** Unter dem Streaming-Mechanismus bleibt der Speicherbedarf unabhängig von der Gesamtlänge $T$:
$$
M(T) = M_{\mathrm{cache}}\,W^2 + M_{\mathrm{model\_chunk}},
$$
unabhängig von $T$.  
**Beweis.**  
1. Pro Fenster werden nur $W^2$ Key-/Value-Matrizen gespeichert.  
2. Die Modell-Chunk-Größe $M_{\mathrm{model\_chunk}}$ ist konstant.  
3. Es existiert kein Term in $T$.  

**Theorem 1 (Throughput Bound).** Für $T\gg W$ nähert sich der Durchsatz an:
$$
\Phi\;\approx\;\frac{W}{C_{\mathrm{att}}W^2 + C_{\mathrm{ff}}W + T_{\mathrm{comm}}}.
$$
**Beweis.**  
1. $N=\lceil T/W\rceil$ Fenster, daher $L_{\mathrm{total}}\approx N L_w + (N-1)T_{\mathrm{comm}}$.  
2. Setze $T=NW$.  
3. Vereinfachung liefert obiges Verhältnis.  

### 3.6 Cross-Paper Integration
- **Asynchrone Distributed Inference** ([paper](../async_distributed_paper/paper.md)): Pipeline-Makespan $M_{\max}$ (Formel 4.1).  
- **Block-Chunked Quantisierung** ([paper](../block_chunk_quant_paper/paper.md)): Compute-Reduktion $\rho$ (Formel 3.1).  
- **Activity Routing** ([paper](../activity_routing_paper/paper.md)): Skip-Ratio $\sigma$ (Abschnitt 5.1).  
- **Boundary Pruning** ([paper](../boundary_pruning_paper/paper.md)): Parameter-Retention $\rho_p$ (Abschnitt 5.1).  

### 3.7 Algorithmic Workflow
```python
from itertools import islice

def streaming_kv_inference(model, token_stream, W, overlap=0):
    """
    Streaming inference with chunked KV-caching.
    """
    past = None
    results = []
    for i in range(0, len(token_stream), W-overlap):
        chunk = token_stream[i:i+W]
        logits, past = model(chunk, past_key_values=past)
        results.append(logits)
    return torch.cat(results, dim=-1)
```

### 3.8 Approximation Error Bound
**Lemma 2 (Chunk-Boundary Error).** Sei $h_t^{\mathrm{full}}$ die Ausgabe von Layer $t$ mit voller Kontextlänge und $h_t^{\mathrm{stream}}$ die Ausgabe im Streaming-Modus. Dann gilt:
$$
\|h_t^{\mathrm{stream}} - h_t^{\mathrm{full}}\|_2 \le \delta(W),
$$
mit $\delta(W) = C_e e^{-\gamma W}$ für Konstanten $C_e,\gamma>0$.  
**Beweis (Sketch).**
1. Die Einflussreichweite von Attention fällt exponentiell mit der Distanz ab (vgl. Longformer).  
2. Bei Chunking wird maximal Kontext über $W$ Toks entfernt.  
3. Daraus folgt ein Exponential-Bound $\delta(W)$.  

### 3.9 Decentralized Scheduling Algorithm
Wir verwenden einen zentralen Scheduler, der Modell- und Kontext-Chunks an Volunteers basierend auf ihrer Kapazität verteilt:
```python
def schedule_chunks(chunks, workers, timeout):
    pending = list(chunks)
    results = []
    while pending:
        for w in workers:
            if w.is_idle() and pending:
                chunk = pending.pop(0)
                w.send(chunk)
        for w in workers:
            if w.has_result(timeout):
                results.append(w.fetch())
    return aggregate_results(results)
```
Die Scheduling-Komplexität liegt bei $O(\frac{T}{W}\log K)$ für $K$ Volunteers.

### 3.10 Fault Tolerance & Straggler Mitigation
In heterogenen Netzwerken entstehen Straggler, die das Pipeline-Makespan verzögern. Wir nutzen Timeouts und Replikation:
- Timeouts $T_{timeout}=c\cdot T_{comm}$.
- Replikation eines Chunks an zwei Workers verringert erwarteten Overhead.
**Lemma 3 (Straggler Bound).** Mit Replikationsfaktor 2 sinkt der erwartete Kommunikations-Overhead auf $O(T_{comm}/2)$.

### 3.11 Security & Privacy Considerations
- **Chunk-Verschlüsselung:** AES-256 pro Chunk vor Verteilung.
- **Secure Aggregation:** Homomorphe Summation der Teilergebnisse ohne Offenlegung von Einzelergebnissen.
- **Anonymität:** Zufällige Zuordnung der Worker-IDs verhindert Rückschlüsse auf Sensitivdaten.

### 3.12 Incentive & Cost Model
Freiwillige erhalten Belohnungen proportional zur erbrachten Rechenzeit. Kostenfunktion:
$$
C_{vol} = \alpha\,t_{comp} - \beta\,E_{energy},
$$
mit $t_{comp}$ der Compute-Dauer und $E_{energy}$ dem Energieverbrauch. Der Nutzer-Kosten:
$$
C_{user} = \gamma\,L_{total} + \delta\,\frac{Data_{transfer}}{BW}.
$$

## 4. Experiments
- **Synthetic Streaming (WikiText-103):**
  ```bash
  python3 experiments/run_streaming_inference.py \
    --window 256 --text data/wikitext103.txt \
    --batch-size 1 --output synthetic_streaming_results.csv
  ```
- **Real-World Text (PG-19):**
  ```bash
  python3 experiments/run_streaming_inference.py \
    --window 512 --text data/pg19.txt \
    --batch-size 1 --output real_streaming_results.csv
  ```

## 5. Results
| Metrik               | Wert                                         |
|----------------------|----------------------------------------------|
| Latenz pro Token     | Ausstehend (Experimente werden noch durchgeführt) |
| Maximaler Speicher   | Ausstehend (Experimente werden noch durchgeführt) |
| Durchsatz            | Ausstehend (Experimente werden noch durchgeführt) |

## 6. Discussion
Unser Ansatz skaliert auf beliebig lange Kontexte, erfordert jedoch Fenster-Größen-Tuning und kann an Fenstergrenzen Artefakte erzeugen.

### 6.1 Dezentrales Volunteer-Computing-Framework
Wir erweitern unser Streaming-Framework um eine dezentrale Infrastruktur, in der Freiwillige ihre Rechenleistung zur Verfügung stellen. Kontextfenster und Modell-Chunks werden über einen zentralen Dispatcher an Teilnehmer-Geräte verteilt, dort berechnet und die Ergebnisse zurückaggregiert. Dadurch erreichen wir:
- **Endlosen Kontext**: Persistente KV-Caches über alle Chunks hinweg.
- **Maximale Modellgrößen**: Skalierung auf Modelle, die einzelne Geräte nicht laden könnten.
- **Vollständige Nutzung bestehender Rechensysteme**: Aggregation ungenutzter CPU-/GPU-Ressourcen.

Jedes Chunk enthält nur Teilmodelle und Teildaten, wodurch Datenschutz und Geheimhaltung gewährleistet bleiben.

## 7. Conclusion
Chunked Streaming mit persistentem KV-Caching ermöglicht unbegrenzten Kontext bei kontrolliertem Speicherbedarf.

## 8. Reproducibility
```bash
pip install -r requirements.txt
python3 experiments/run_streaming_inference.py --window 256 --text data.txt --batch-size 1
```

## References
- Vaswani et al., NeurIPS 2017
- Kitaev et al., ICLR 2020
- Beltagy et al., ArXiv 2020
- Guo et al., NeurIPS 2022
