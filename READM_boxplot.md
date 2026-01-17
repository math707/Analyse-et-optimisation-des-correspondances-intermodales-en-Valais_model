
---

## README — dossier `Boxplot/`


# Boxplots (visualisation des distributions de temps d’attente)

Ce dossier sert UNIQUEMENT à transformer des fichiers `out_*_boxplots.json` (payloads) en figures `.png` (boxplots pondérés).
Il est volontairement séparé du moteur d’optimisation : l’optimisation produit des résultats + (optionnellement) un payload JSON, et ici on fabrique les figures.

---

## 1) Organisation du dossier

- `outputs_boxplots_case_1/`
- `outputs_boxplots_case_2/`
- `outputs_boxplots_case_3/`
  -> Dossiers contenant les payloads JSON à comparer (un fichier JSON = un “case”).

- `plots_boxplots_case_1/`
- `plots_boxplots_case_2/`
- `plots_boxplots_case_3/`
  -> Dossiers de sortie des figures `.png` générées.

- `BOXPlotMaker.py`
  -> Script principal :
     - lit tous les `.json` présents dans un dossier `outputs_*`
     - calcule des boxplots *pondérés* (quantiles, moyenne, moustaches)
     - génère des figures par “correspondance” ou par “case”

- `run_boxplots.py`
  -> Petit runner “manuel” :
     - tu choisis un INPUT_DIR + OUTPUT_DIR
     - tu choisis un mode (1..6) pour générer les figures voulues

---

## 2) Format attendu des fichiers d’entrée (`out_*_boxplots.json`)

Chaque fichier JSON doit ressembler à :

```json
{
  "case": "NomDuCas",
  "transfers": {
    "FromService:FromAnchor->ToService:ToAnchor": {
      "values_up":   [ ... ],
      "weights_up":  [ ... ],
      "values_down": [ ... ],
      "weights_down":[ ... ],
      "values_all":  [ ... ],
      "weights_all": [ ... ],
      "walk_time": 3,
      "min_margin": 1,
      "w_up": 1.0,
      "w_down": 0.0,
      "n_up": 12,
      "n_down": 0
    }
  }
}
````

Notes :

* Les clés de transferts doivent avoir la forme :
  `FromService:FromAnchor->ToService:ToAnchor`
* `values_*` = liste des temps d’attente (en minutes)
* `weights_*` = poids associés à chaque observation (même longueur que `values_*`)
* `*_all` est le mélange UP + DOWN (utile si tu veux une seule distribution globale par correspondance)

---

## 3) Workflow (manuel) — ce que tu fais actuellement

1. Choisir un “case” (ex. case_1) et préparer un dossier d’entrée :

   * `Boxplot/outputs_boxplots_case_1/`

2. Copier/coller (ou glisser-déposer) dedans les fichiers :

   * `out_Sierre_case_ref_boxplots.json`
   * `out_Sierre_case_4_funi_boxplots.json`
   * etc.

3. Ouvrir `run_boxplots.py` et régler :

```python
INPUT_DIR = "outputs_boxplots_case_1"
OUTPUT_DIR = "plots_boxplots_case_1"
```

4. Lancer :

```bash
python run_boxplots.py
```

Par défaut, le script fait :

* `run(1)` => 1 figure par correspondance, en utilisant `values_all/weights_all` (recommandé)

---

## 4) Modes disponibles dans `run_boxplots.py`

Le choix `run(choice)` pilote 2 dimensions :

* Mode de regroupement :

  * `by_transfer` : 1 figure par correspondance (comparaison de plusieurs cases)
  * `by_case`     : 1 figure par case (comparaison des correspondances à l’intérieur d’un case)

* Champ utilisé :

  * `all`  : mélange UP+DOWN (souvent ce que tu veux)
  * `up`   : uniquement le sens “arrival -> departure”
  * `down` : uniquement le sens inverse

Mapping actuel :

1. by_transfer + all   (recommandé)
2. by_case + all
3. by_transfer + up
4. by_transfer + down
5. by_case + up
6. by_case + down

---

## 5) Ce que fait `BOXPlotMaker.py` (résumé)

### Lecture des payloads

* `read_payloads(input_dir)` lit tous les `.json` dans `input_dir`.

### Calcul des stats “boxplot pondéré”

* `make_weighted_boxstats(label, values, weights)` calcule :

  * Q1 / médiane / Q3 via quantiles pondérés
  * moyenne pondérée (μ) affichée au-dessus de la moustache
  * moustaches type “1.5×IQR” (adaptées aux données disponibles)

### Génération des figures

* `mode_by_transfer(...)` :
  -> pour chaque clé de transfert, on trace un boxplot par case (comparaison inter-cases).
* `mode_by_case(...)` :
  -> pour chaque case, on trace un boxplot par transfert (comparaison intra-case).

---

## 6) Sorties

Les figures sont écrites dans `OUTPUT_DIR` avec des noms du type :

* `by_transfer__all__<transfer_key_sanitized>.png`
* `by_case__all__<case_name>.png`

