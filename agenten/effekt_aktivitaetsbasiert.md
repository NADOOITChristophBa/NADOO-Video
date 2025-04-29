# Mathematische Effekte: Dynamische, aktivitätsbasierte Modellnutzung

## Erklärungen & Auswirkungen

### Formel:
Effizienzsteigerung:
$$
\text{Effizienz} = \frac{M}{k}
$$

### Beispiel mit Zahlenwerten
- $M=10$ Blöcke, $k=3$ aktive Blöcke pro Inferenz
- Klassisch: 10 Berechnungen, Dynamisch: nur 3
- **Fazit:** 3,3x schneller und weniger Speicherbedarf

### User Story 1
*Als Mobile-Entwickler* möchte ich große Modelle auf Smartphones nutzen, indem ich nur die aktiven Blöcke berechne, um RAM und Akku zu sparen.

### User Story 2
*Als Data Scientist* möchte ich Modelle sparsamer rechnen lassen, damit ich mehr Experimente parallel durchführen kann.
