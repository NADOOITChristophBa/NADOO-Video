# Mathematische Effekte: Asynchrone, verteilte KI-Inferenz

## Erklärungen & Auswirkungen

### Formel:
Gesamtdurchsatzrate:
$$
R = \sum_{i=1}^n \frac{1}{T_i}
$$

### Beispiel mit Zahlenwerten
- 8 Geräte, Rechenzeiten: 1.2s, 1.5s, 1.1s, 2.0s, 1.8s, 1.3s, 1.7s, 1.4s
- Gesamtdurchsatz: $R = 1/1.2 + 1/1.5 + ... + 1/1.4 \approx 5.9$ Aufgaben/s
- Seriell (langsamstes Gerät): $R_{ser} = 1/2.0 = 0.5$ Aufgaben/s
- **Fazit:** Fast 12x schneller als das langsamste Gerät alleine.

### User Story 1
*Als Forscher* möchte ich 1000 Bilder auf 20 Handys verteilen, damit die Gesamtverarbeitung trotz langsamer Einzelgeräte in akzeptabler Zeit abgeschlossen ist.

### User Story 2
*Als Entwickler* möchte ich eine KI-Inferenz auf viele Edge-Geräte verteilen, um Energie zu sparen und die Verarbeitung zu beschleunigen, ohne einen teuren Server zu betreiben.
