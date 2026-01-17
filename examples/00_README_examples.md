# Examples JSON (copy/paste ready)

## How to run
From project root:
- put the JSON you want in the root or keep it in `examples/`
- call `run_from_json("path/to/file.json")` (in `run.py`)

## Files
- 01_full_multi.json
  End-to-end example: trains (pattern_streams) + line + route, with transfers, tie-breakers, boxplots export.

- 02_scheduled_offset.json
  Scheduled service + offset optimization (topology="scheduled" with offset_range_min > 0).

- 03_loop.json
  Loop topology (cycle_time + turnaround).

- 04_line_phi_optimize.json
  Line topology with phi optimization (phi_minutes="optimize").

- 05_route_multistop.json
  Route topology (multi-stops closed loop) with legs + dwells.

- 06_singletrack_main.json
  Singletrack topology calling `singletrack_v2` + a separate line config file.

- 06_singletrack_line_config_R81.json
  Singletrack infrastructure/line constraints (run times sets, dwell sets, forbidden meet stops, etc.).

## Note about JSON comments
`.json` files are valid JSON and runnable.
`.jsonc` files include comments (// ...) for documentation and are not strict JSON.

Oui, tu as raison d’être “pas sûre” — et là on peut répondre de façon **factuelle** avec ton `engine_runner.py` :

## Où se passent les exports, dans TON code actuel ?

Dans ce fichier, il y a **2 exports** explicitement gérés :

### 1) Export CSV (via l’optimizer)

Dans `run_from_json` :

```py
export_cfg = ((cfg.get("export") or {}).get("csv") or {})
csv_path = export_cfg.get("path") if export_cfg.get("enabled") else None

results = optimize_services(..., csv_path=csv_path)
```

➡️ Donc `export.csv` est lu ici, mais **l’écriture du CSV est faite dans `optimizer.py`** (`_export_csv` + `_export_all_candidates_csv`).

✅ JSON attendu :

```json
"export": {
  "csv": { "enabled": true, "path": "out_results.csv" }
}
```

Et ça produit :

* `out_results.csv` (gagnants seulement)
* `out_results_all.csv` (tous les candidats testés) — car ton optimizer fait `csv_path.replace(".csv", "_all.csv")`.

---

### 2) Export “boxplots payload” (JSON)

Toujours dans `run_from_json` :

```py
dists = extract_all_transfer_distributions(results, transfers)
export_box = ((cfg.get("export") or {}).get("boxplots") or {})
if export_box.get("enabled"):
    out_path = export_box.get("path", "boxplot_payload.json")
    payload = {"case": case_name, "transfers": dists}
    json.dump(payload, ...)
```

✅ JSON attendu :

```json
"export": {
  "boxplots": { "enabled": true, "path": "boxplot_payload.json" }
}
```

---

## Donc “tous les exports”, c’est bien ici… mais réparti sur 2 endroits

* **lecture des options d’export** : `engine_runner.py`
* **écriture du CSV** : `optimizer.py`
* **écriture du boxplots payload** : `engine_runner.py`

👉 Si tu avais d’autres exports (PDF, figures, logs…), ils seraient soit :

* dans `engine_runner.py` (après `optimize_services`), soit
* dans un autre runner/script (ex. `run.py`, `reporting.py`, etc.), soit
* dans les modules appelés (ex. un module “boxplots_maker” qui lit `out_*_boxplots.json`).

Là, avec ce que tu m’as collé, **il n’y en a pas d’autres**.

---


## Exports

Exports are configured under `export` at the top-level of the case JSON.

### 1) CSV export (results + all candidates)
Config:
```json
"export": {
  "csv": { "enabled": true, "path": "out_results.csv" }
}
````

What it produces:

* `out_results.csv`: one row per optimized service (winner only)
* `out_results_all.csv`: all candidates tested (debug / analysis)

Where it happens:

* option is read in `engine_runner.py`
* files are written in `optimizer.py` (`_export_csv`, `_export_all_candidates_csv`)

### 2) Boxplots payload export (transfer wait distributions)

Config:

```json
"export": {
  "boxplots": { "enabled": true, "path": "boxplot_payload.json" }
}
```

What it produces:

* a JSON payload with:

  * `case`: case name
  * `transfers`: for each transfer key, the lists:

    * `values_up`, `weights_up`
    * `values_down`, `weights_down`
    * `values_all`, `weights_all`
      plus meta fields (walk_time, min_margin, w_up, w_down, ...)

Where it happens:

* computed in `engine_runner.py` using `extract_all_transfer_distributions(...)`
* written in `engine_runner.py`
