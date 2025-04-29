# TODO-Übersicht

## Sofortige Bugfixes
- [x] Flattening: Input vor MLP-Modulen in `run_activity_routing_classification.py` flachlegen (Bug behoben).

## Logging & Reporting
- [ ] Strukturiertes Logging in JSON/CSV statt Freitext
- [ ] Automatische Plots in `summarize_experiment_results.py`
- [ ] TQDM-Fortschrittsbalken pro Experiment

## Code-Qualität
- [ ] Type Hints & Docstrings ergänzen
- [ ] Unit Tests für `measure()` & `synthetic_gaussian_loader()`
- [ ] Code-Style via `black` & `flake8`

## Konfigurationsmanagement
- [ ] Hydra / YAML-Konfiguration für Skripte
- [ ] `requirements.txt` aktualisieren & Dockerfile erstellen

## Performance-Optimierung
- [ ] `torch.jit.script` oder Numba für Hotspot-Loops
- [ ] BLAS- & Thread-Settings dokumentieren & anpassen

## Reproduzierbarkeit
- [ ] Seeds fixieren (`torch.manual_seed`, `np.random.seed`)
- [ ] README um Hardware- & BLAS-Infos ergänzen
