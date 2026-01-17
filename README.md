
---

# Project — Correspondance Optimization (Semester Project)

Ce projet optimise des correspondances intermodales (train / funiculaire / bus, etc.) sur une fenêtre temporelle, en testant des horaires candidats et en minimisant une fonction objectif basée sur les **temps d’attente de correspondance** (montée et descente), avec gestion des ex æquo via **tie-breakers**.

Le code est conçu pour être piloté via un fichier **JSON** (cas d’étude), puis exécuté via `run.py` (ou un équivalent).

---

## 1) Quickstart

### Prérequis

* Python 3.10+ recommandé
* Dépendances standard (pas de packages externes obligatoires, sauf si tu en ajoutes)

### Exécution

1. Mettre un fichier JSON de configuration dans le dossier de travail (ex. `Sierre_train_bus_BS1.json`).
2. Dans `run.py`, changer la variable `CONFIG = "..."`.
3. Lancer :

```bash
python run.py
```

Sorties typiques :

* logs console (résumé des résultats)
* export boxplots si activé dans le JSON
* export CSV si activé côté engine (selon implémentation)

---

## 2) Vue d’ensemble de l’architecture

Le projet est structuré autour de 4 briques :

1. **`transit_core.py`**
   Modèle de données + génération des horaires (topologies) + évaluation des transferts.

2. **`optimizer.py`**
   Génération de candidats, évaluation de l’objectif, sélection du meilleur candidat (avec tie-breakers).
   Contient aussi une logique spécifique “singletrack”.

3. **`singletrack_v2.py`**
   Sous-modèle dédié aux lignes ferroviaires à voie unique (“singletrack”) : génération de patterns faisables + validation exacte (reconstruction interne).

4. **`engine_runner.py`**
   Lecture JSON + construction des `Service`/`Transfer` + lancement de l’optimisation + reporting/export.

Fichiers utilitaires :

* **`tie_breakers.py`** : registre de tie-breakers (plugins).
* **`run.py`** : script minimal de lancement d’un cas JSON.

---

## 3) Concepts et objets principaux

### 3.1 `Service`, `Anchor`, `Transfer` (dans `transit_core.py`)

* **Anchor** : un point physique (gare/arrêt) où un service a des événements :

  * `arrivals` : liste de temps d’arrivée **en minutes absolues**
  * `departures` : liste de temps de départ **en minutes absolues**
  * `arrival_weights`, `departure_weights` : poids (même longueur que arrivals/departures)
  * `ensure_weights()` garantit que les poids existent (par défaut 1.0)

* **Service** : un mode de transport (train/funi/bus…) :

  * `name`, `category`
  * `anchors: Dict[str, Anchor]` (au minimum un anchor `"default"`)
  * `template: ServiceTemplate` si le service est généré/optimisé (loop/line/route/singletrack/scheduled+offset)
  * `pattern_streams` : pour les services “fixes” (trains par exemple) définis par minutes dans l’heure

* **Transfer** : une correspondance entre deux anchors :

  * `from_service`, `from_anchor`
  * `to_service`, `to_anchor`
  * `walk_time`, `min_margin`
  * pondérations : `w_up` (montée), `w_down` (descente)
  * tie-break : `tie_breaker`, `tie_params`

---

## 4) Fenêtre temporelle, padding et évaluation

Le projet travaille sur une fenêtre globale `[WINDOW_START, WINDOW_END)` définie dans `transit_core.py`.

Fonctions importantes :

* `set_window("06:00", "09:00")` : définit la fenêtre.
* `set_padding(minutes)` : padding autour de la fenêtre (utile pour éviter les effets de bord).
* `_padded_window()` (interne) : renvoie `[WINDOW_START - PADDING, WINDOW_END + PADDING)`.

### Évaluation d’un transfert

`evaluate_transfer(src_anchor, dst_anchor, walk_time, min_margin, w_up, w_down)` calcule :

* **Montée (UP)** : `arrivals(src) -> departures(dst)`
* **Descente (DOWN)** : `departures(src) -> arrivals(dst)` (formulation symétrique)

Le score d’un transfert est :

```
score = w_up * avg_up_w + w_down * avg_down_w
```

où `avg_up_w` / `avg_down_w` sont des moyennes pondérées (par les poids des événements).

Dans `optimizer.py`, `evaluate_transfer_guarded` ajoute un comportement plus “tolérant” :

* si un sens est demandé mais qu’un côté n’a pas d’événements, il ne crash pas automatiquement
* il ne lève une erreur que si les deux côtés ont des événements mais qu’aucune correspondance n’est trouvée

---

## 5) Topologies supportées

Les topologies sont définies par `ServiceTemplate.topology`.

### 5.1 `scheduled` (service fixe)

* L’horaire est donné via `pattern_streams` (déployé par `expand_pattern_streams`).
* Optionnel : mode “offset” si `offset_range_min > 0`, l’optimizer teste des décalages globaux.

Champs utiles :

* `base_offset_min`, `offset_range_min`, `offset_step_min`

### 5.2 `loop`

Un seul anchor, cycle complet.
Champs utiles :

* `cycle_time`, `turnaround`
* `n_per_hour`, `equal_headway`

Génération : `expand_loop_anchor(anchor, pattern_minutes, tmpl)`

### 5.3 `line` (A ↔ B)

Deux anchors (terminus A et B).
Champs utiles :

* `anchors = ["A","B"]`
* `tt_AB`, `tt_BA`
* `turnaround_A`, `turnaround_B`
* `phi_minutes` : phase entre les départs (B = A + phi mod 60), peut être int ou `"optimize"`

Génération : `expand_line_anchors(anchor_A, anchor_B, pattern_A, phi, tmpl)`

### 5.4 `route` (boucle multi-stops fermée)

Boucle définie par une liste d’arrêts + temps de parcours entre arrêts.
Champs utiles :

* `stops: ["S0","S1",...,"S(N-1)"]`
* `leg_minutes`: N valeurs incluant la dernière jambe `S(N-1) -> S0`
* `dwells`: temps d’arrêt par stop (0 si absent)
* `n_per_hour`, `equal_headway`

Génération : `expand_route_anchors(anchors, pattern_minutes, tmpl)`

### 5.5 `singletrack` (voie unique, sous-modèle dédié)

Cas particulier : l’horaire est contraint par la faisabilité de croisement/dépassement sur une voie unique.
L’optimizer :

1. génère des patterns candidats via `singletrack_v2.compute_singletrack_hourly_patterns`
2. évalue ces candidats comme les autres
3. **valide exactement** le meilleur (ou runner-ups) via `singletrack_v2.validate_singletrack_candidate`
4. si OK, “finalise” les anchors de tous les stops via `fill_singletrack_anchors_from_exact_details(...)`

Champs typiques dans `ServiceTemplate` :

* `st_config_path` : chemin vers le JSON du modèle singletrack
* `st_n_dep_per_hour`, `st_min_headway`
* `st_forbidden_meet_stops`, `st_enable_no_overtaking`, `st_extra_terminus_slack`
* `base_offset_min`, `offset_range_min`, `offset_step_min` (offset global en plus)

---

## 6) Optimisation et séquentialité (`optimizer.py`)

### 6.1 Pipeline par service : `find_best_for_service`

Pour un service cible :

1. Génère les candidats (`candidates_for_service`)
2. Pour chaque candidat :

   * reconstruit les anchors depuis l’état “orig”
   * réalise l’horaire (`realize_service_schedule` ou `shift_service_inplace` en mode offset)
   * évalue l’objectif (`eval_service_objective`)
3. Récupère le meilleur score + liste des ex æquo (tie tolérance `tie_tol`)
4. Applique un tie-breaker si nécessaire
5. **Si topology == singletrack** : validation exacte itérative (best score puis runner-ups)

Retourne :

* `chosen` : meilleur candidat (EvalRecord)
* `ties` : liste des EvalRecord ex æquo
* `ties_dicts` : métriques agrégées (format plugin)
* `records` : tous les candidats testés

### 6.2 Optimisation multi-services : `optimize_services`

`optimize_services(services, transfers, targets, ...)` optimise les services dans l’ordre de `targets`.

Important : `eval_service_objective` ignore certains transferts si le service “contrepartie” est cadencé mais encore vide (optimisation séquentielle).

---

## 7) Tie-breakers (`tie_breakers.py`)

Les tie-breakers permettent de départager des solutions ex æquo (même score à tolérance près).

### Utilisation

Dans le JSON :

```json
"tie_breakers": { "module": "tie_breakers" }
```

Dans un transfert :

```json
"tie_breaker": "penalty_sum",
"tie_params": { "max_wait": 10, "lambda": 2.0 }
```

Stratégies disponibles (selon ton `tie_breakers.py`, PS: d'autres stratégies peuvent être ajoutées) :

* `none`
* `penalty`
* `min_max`
* `min_variance`
* `penalty_sum`
* `chain_strategies` (optionnel, cascade de règles)

---

## 8) Formats JSON

### 8.1 Structure générale d’un cas

Champs typiques :

* `window` : start/end
* `case_name`
* `tie_breakers.module`
* `modes` : liste de services (certains fixes via streams, d’autres via template)
* `transfers` : liste des correspondances à optimiser
* `targets` : ordre d’optimisation

### 8.2 Exemple “scheduled streams” (service fixe)

```json
{
  "name": "Trains",
  "category": "rail",
  "pattern_streams": [
    { "label": "R91→Sion", "depart_minutes": [8,36], "arr_offset_min": 0, "w": 0.2 }
  ]
}
```

### 8.3 Exemple “line”

```json
{
  "name": "Funi A–B",
  "category": "funi",
  "template": {
    "topology": "line",
    "anchors": ["A","B"],
    "n_per_hour": 4,
    "equal_headway": false,
    "tt_AB": 14,
    "tt_BA": 14,
    "turnaround_A": 4,
    "turnaround_B": 4,
    "phi_minutes": 0
  }
}
```

### 8.4 Exemple “route”

```json
{
  "name": "Bus loop",
  "category": "bus",
  "template": {
    "topology": "route",
    "stops": ["S0","S1","S2","S3"],
    "leg_minutes": [9,16,17,8],
    "dwells": { "S0": 3, "S2": 7 },
    "n_per_hour": 3,
    "equal_headway": false
  }
}
```

### 8.5 Exemple “singletrack”

```json
{
  "name": "R81",
  "category": "rail",
  "template": {
    "topology": "singletrack",
    "st_config_path": "config_singletrack_R81.json",
    "base_offset_min": 0,
    "offset_range_min": 0,
    "offset_step_min": 1,
    "st_n_dep_per_hour": 2,
    "st_min_headway": 20,
    "st_forbidden_meet_stops": ["Martigny", "Le Châble VS"],
    "st_enable_no_overtaking": true
  }
}
```

---

## 9) Fichiers de sortie / exports

Selon la config du runner :

* export boxplots (JSON) si activé
* export CSV des gagnants (`_export_csv`)
* export CSV complet de tous les candidats testés (`_export_all_candidates_csv`) si activé

---

## 10) Notes pratiques / Debug

* Les temps sont manipulés en **minutes absolues** (pas de modulo 24h dans les shifts).
* Le **padding** est important pour éviter que des correspondances proches des bords de fenêtre soient évaluées de manière incohérente.
* En singletrack, la validation exacte peut imprimer des blocs “debug” (reconstruction interne, événements par stop).

---

