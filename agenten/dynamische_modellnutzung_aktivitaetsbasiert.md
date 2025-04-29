# Dynamische, aktivitätsbasierte Modellnutzung in KI-Systemen

## Grundidee

Große neuronale Netze (z.B. Transformer, CNNs) sind bei einer konkreten Eingabe (Prompt, Bild, etc.) meist nur in bestimmten Teilen („Regionen“) wirklich aktiv. Viele Gewichte/Layer/Neuronen sind für einen bestimmten Input fast „tot“ (kaum Aktivierung, kaum Einfluss auf das Ergebnis).

## Algorithmischer Ansatz

1. **Aktivitätsmessung:**
   - Während der Inferenz werden die Aktivierungen in den Layern/Blöcken gemessen.

2. **Dynamisches Laden und Berechnen:**
   - Nur die Blöcke/Layers mit hoher Aktivität werden vollständig geladen und durchgerechnet.
   - Wenig aktive Blöcke werden übersprungen, sparsamer berechnet oder ganz ausgelassen.

3. **Ressourcen sparen:**
   - Große Teile des Modells werden für einen konkreten Input gar nicht erst geladen – das spart RAM, VRAM und Rechenzeit.

## Methoden

- **Top-K Aktivierung:** Pro Layer/Block werden nur die K aktivsten Neuronen/Heads/Channels berechnet.
- **Routing-Netzwerke:** Ein separates Controller-Netzwerk entscheidet, welche Blöcke für einen Input aktiviert werden (z.B. Mixture of Experts).
- **Pruning on-the-fly:** Während der Inferenz werden Layer mit niedriger Aktivität temporär abgeschaltet.
- **Attention-Masken:** Nur relevante Regionen (z.B. im Bild, im Prompt) werden durch das Modell propagiert.

## Praktische Umsetzung

- **Profiling-Phase:** Für viele Inputs werden Aktivitätsmuster gemessen (Heatmaps, Layer-wise Activations).
- **Laufzeit-Entscheidung:** Während der Inferenz entscheidet das System dynamisch, welche Blöcke geladen/genutzt werden.
- **Fallback:** Bei zu viel Inaktivität kann auf vollständige Inferenz zurückgefallen werden, um Genauigkeit zu sichern.

## Vorteile

- **Massive Einsparung an Speicher und Rechenzeit**
- **Skalierbarkeit:** Besonders effizient bei Streaming/Block-Laden
- **Potenzial für mobile/embedded KI**

## Herausforderungen

- **Genauigkeit:** Zu aggressives Pruning kann die Modellqualität verschlechtern.
- **Dynamik:** Aktive Regionen unterscheiden sich je nach Input.
- **Implementierung:** Erfordert dynamisches Routing und ein flexibles Framework.

## Forschung & Inspiration

- **Mixture of Experts (MoE):** Nur ein Teil der Experten wird pro Input aktiviert (GShard, Switch Transformer, etc.).
- **Dynamic Sparse Attention:** Nur relevante Attention-Maps werden berechnet.
- **Conditional Computation:** Adaptive Computation Time, Layer Skipping, etc.

## Fazit

Mit Streaming-Gewichten und blockweisem Laden ist aktivitätsbasiertes, dynamisches Routing besonders effizient. So können große Modelle noch sparsamer und schneller auf kleinen Geräten genutzt werden, ohne die Architektur grundlegend zu ändern.
