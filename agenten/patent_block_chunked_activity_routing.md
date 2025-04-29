# Patent-Idee: Kombiniertes blockweises Streaming und aktivitätsbasiertes Routing für KI-Modelle

## Erfinder: Christoph Backhaus

## Technisches Prinzip

Das Modell wird in Blöcke und darin in Chunks unterteilt. Während der Inferenz werden nur die aktuell benötigten Blöcke nacheinander geladen (Streaming/seriell). Innerhalb jedes Blocks werden wiederum nur die aktivsten Chunks (z.B. Channels, Heads, Patches) berechnet. So können beliebig große Modelle speichereffizient und adaptiv genutzt werden – und innerhalb jedes Blocks wird nur das Nötigste gerechnet.

## Schutzwürdige Aspekte
- Kombination aus sequentiellem Block-Laden und aktivitätsbasiertem Routing innerhalb der Blöcke.
- Ermöglicht 4D-selektive, adaptive Inferenz: Nur die wahrscheinlich aktivsten Regionen werden geladen und berechnet.
- Extrem speichereffizient und skalierbar.

## Anwendung
- KI auf mobilen Geräten, Edge/IoT, große Modelle auf kleinen Geräten, energieeffiziente KI

## Beispiel-Algorithmus (Pseudocode)

```python
for block in model:
    # Lade Block in den Speicher
    load_block(block)
    activities = [measure_activity(chunk, input) for chunk in block.chunks]
    topk = select_topk_chunks(activities, k)
    for i in topk:
        output_chunk = block.chunks[i](input_chunk[i])
    # Entlade Block wieder
    unload_block(block)
```

## Mathematischer Beweis (Effizienz)

Sei $B$ die Anzahl Blöcke, $C$ die Chunk-Anzahl pro Block, $k$ die Zahl der aktiv genutzten Chunks pro Block. Die Gesamtrechenzeit ist:
$$
T_{comb} = \sum_{j=1}^B \sum_{i=1}^k t_{ji}
$$
Im klassischen Ansatz: $T_{full} = \sum_{j=1}^B \sum_{i=1}^C t_{ji}$. Für $k < C$ und sequentielles Block-Streaming ist $T_{comb} \ll T_{full}$. Speicherbedarf pro Schritt ist $\mathcal{O}(k)$ statt $\mathcal{O}(C)$.

---

Siehe auch: /agenten/aktivitaetsbasiertes_routing_beispiel.py (und Folge-Skript mit Chunking)
