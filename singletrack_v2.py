# singletrack_v2.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any
import json


# =========================
# 1) Structures de base
# =========================

@dataclass
class SegmentRange:
    """Plage de temps de parcours pour un tronçon (en minutes)."""
    t_min: float
    t_max: float


@dataclass
class LineConfig:
    """
    Représentation simplifiée de la ligne single-track pour la V2
    (centrée sur l'ancre A, ici Martigny).
    """
    name: str
    stops: List[str]

    # Tronçons vers le haut (stops[0] -> stops[-1]) et vers le bas
    run_up: List[SegmentRange]
    run_down: List[SegmentRange]

    run_values_up: List[List[int]]
    run_values_down: List[List[int]]

    # Dwells min / max par arrêt
    dwell_min: Dict[str, float]
    dwell_max: Dict[str, float]

    dwell_times: Dict[str, List[float]]

    forbidden_meet_stops: List[str] = None  # si tu veux garder simple

    # Période de cadencement (en minutes, typiquement 60)
    period: int = 60

    @property
    def terminus_A(self) -> str:
        return self.stops[0]

    @property
    def terminus_B(self) -> str:
        return self.stops[-1]

    @property
    def intermediate_stops(self) -> List[str]:
        return self.stops[1:-1]


@dataclass
class TerminusPattern:
    """
    Pattern au terminus A (Martigny) sur une période :
      - arr_minutes : 2 arrivées modulo period (0..period-1)
      - dep_minutes : 2 départs modulo period (0..period-1)
      - dwell_minutes : dwell choisi à A pour chaque arrivée (>= dwell_A_min)
    Important : l'association est 1-à-1 par index (arr[i] -> dep[i]).
    """
    arr_minutes: List[int]
    dep_minutes: List[int]
    dwell_minutes: List[int]


# =========================
# 2) Helpers de parsing
# =========================

def _to_segment_ranges(run_times_field: List[Any]) -> List[SegmentRange]:
    """
    Convertit un champ "run_times_up" ou "run_times_down" du JSON en
    liste de SegmentRange.

    Exemples:
      [ [2,3,4], [6,7], [4,5,6], [8,9] ]
      -> [SegmentRange(2,4), SegmentRange(6,7), SegmentRange(4,6), SegmentRange(8,9)]
    """
    segs: List[SegmentRange] = []
    for entry in run_times_field:
        # entry peut être [2,3,4] ou directement un nombre
        if isinstance(entry, (int, float)):
            vals = [float(entry)]
        else:
            vals = [float(v) for v in entry]
        segs.append(SegmentRange(min(vals), max(vals)))
    return segs


def _to_dwell_min_max(dwell_times_field: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Convertit le champ "dwell_times" du JSON en 2 dicts dwell_min / dwell_max.

    Important:
      - Pour les terminus, tu déclares souvent un unique temps (ex: [3]),
        qu'on interprète comme un MINIMUM: on peut attendre plus.
      - Pour les arrêts intermédiaires, une liste (ex: [0,1]) donne min=0, max=1.
    """
    dmin: Dict[str, float] = {}
    dmax: Dict[str, float] = {}

    for stop, vals in dwell_times_field.items():
        if isinstance(vals, (int, float)):
            nums = [float(vals)]
        else:
            nums = [float(v) for v in vals]
        dmin[stop] = min(nums)
        dmax[stop] = max(nums)

    return dmin, dmax


def load_line_config(path: str) -> LineConfig:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    line_cfg = cfg["line"]

    forbidden = cfg.get("forbidden_meet_stops", []) or []
    forbidden = [str(x) for x in forbidden]

    name = line_cfg.get("name", "SingleTrackLine")
    stops = list(line_cfg["stops"])

    run_up_field = line_cfg["run_times_up"]
    run_down_field = line_cfg.get("run_times_down", run_up_field)

    def _as_int_list(entry) -> List[int]:
        if isinstance(entry, (int, float)):
            return [int(round(entry))]
        return [int(round(v)) for v in entry]

    # UP: A -> B (ordre JSON)
    run_values_up = [_as_int_list(e) for e in run_up_field]
    run_up = _to_segment_ranges(run_up_field)

    # DOWN JSON est aussi A -> B, MAIS le code DOWN travaille en B -> A,
    # donc on inverse l'ordre des segments.
    run_down_field_rev = list(reversed(run_down_field))
    run_values_down = [_as_int_list(e) for e in run_down_field_rev]
    run_down = _to_segment_ranges(run_down_field_rev)

    dwell_times_field_raw = line_cfg.get("dwell_times", {}) or {}

    # normaliser: stop -> List[float]
    dwell_times_field: Dict[str, List[float]] = {}
    for stop, vals in dwell_times_field_raw.items():
        if isinstance(vals, (int, float)):
            dwell_times_field[stop] = [float(vals)]
        else:
            dwell_times_field[stop] = [float(v) for v in vals]

    dwell_min, dwell_max = _to_dwell_min_max(dwell_times_field)

    # --- NEW: terminus_dwell_max (override du max aux terminus) ---
    term_max_raw = line_cfg.get("terminus_dwell_max", {}) or {}
    if isinstance(term_max_raw, dict) and term_max_raw:
        A = stops[0]
        B = stops[-1]

        def _as_float(x):
            try:
                return float(x)
            except Exception:
                return None

        for term in (A, B):
            if term in term_max_raw:
                v = _as_float(term_max_raw[term])
                if v is not None:
                    # max >= min, sécurité
                    dwell_max[term] = max(v, float(dwell_min.get(term, 0.0)))


    period = int(line_cfg.get("horizon", 60))

    return LineConfig(
        name=name,
        stops=stops,
        run_up=run_up,
        run_down=run_down,
        dwell_min=dwell_min,
        dwell_max=dwell_max,
        dwell_times=dwell_times_field,
        period=period,
        forbidden_meet_stops=forbidden,
        run_values_up=run_values_up,
        run_values_down=run_values_down,

    )




# =========================
# 3) Bornes de cycle A -> B -> A
# =========================

def compute_cycle_bounds(
    line: LineConfig,
    extra_terminus_slack: float = 0,
) -> Tuple[float, float]:
    """
    Calcule une borne (approx) du temps de cycle entre:
      départ terminus_A -> arrivée terminus_A
      en faisant A -> B -> A.

    - On utilise:
        * temps de parcours min/max sur chaque tronçon
        * dwell min/max aux arrêts intermédiaires
        * dwell min aux terminus (A et B) pour le cycle_min
        * dwell max aux terminus + un slack supplémentaire
          pour le cycle_max (car aux terminus, ta dwell est un minimum).

    Remarque:
      - L'idée n'est pas d'être ultra précis mais d'avoir des bornes
        "physiquement raisonnables" pour filtrer les patterns
        (depart/arrivée à A) avant de vérifier la faisabilité détaillée.
    """
    A = line.terminus_A
    B = line.terminus_B
    inter = line.intermediate_stops

    # Somme des temps de parcours min/max vers le haut / vers le bas
    up_min = sum(seg.t_min for seg in line.run_up)
    up_max = sum(seg.t_max for seg in line.run_up)
    down_min = sum(seg.t_min for seg in line.run_down)
    down_max = sum(seg.t_max for seg in line.run_down)

    # Dwells aux arrêts intermédiaires, supposés symétriques UP/DOWN
    dw_inter_min = sum(line.dwell_min.get(s, 0.0) for s in inter)
    dw_inter_max = sum(line.dwell_max.get(s, 0.0) for s in inter)

    # Dwells aux terminus: ce que tu déclares est un MINIMUM
    dwell_A_min = line.dwell_min.get(A, 0.0)
    dwell_B_min = line.dwell_min.get(B, 0.0)

    dwell_A_max = max(line.dwell_max.get(A, dwell_A_min), dwell_A_min)
    dwell_B_max = max(line.dwell_max.get(B, dwell_B_min), dwell_B_min)

    # Pour un cycle départ A -> arrivée A :
    #
    #   A (dep) -> ... inter ... -> B (arr)  : up_min + dw_inter_min
    #   B (dwell)                             : dwell_B_min
    #   B (dep) -> ... inter ... -> A (arr)  : down_min + dw_inter_min
    #
    # On n'inclut PAS la dwell_A dans le cycle_min, car on veut
    # la durée entre le départ à A et l'arrivée à A.
    #
    cycle_min = (
        up_min
        + down_min
        + 2.0 * dw_inter_min
        + dwell_B_min
    )

    # Pour cycle_max, on autorise:
    #   - temps de parcours max
    #   - dwells max aux inter
    #   - dwell max à B
    #   - et un slack supplémentaire aux terminus (A et B)
    #     pour représenter le fait qu'on peut attendre plus longtemps.
    #
    cycle_max = (
        up_max
        + down_max
        + 2.0 * dw_inter_max
        + dwell_B_max
        + extra_terminus_slack  # marge de sécurité (paramétrable)
    )

    return cycle_min, cycle_max


@dataclass
class StopOffsets:
    # offsets depuis le départ au terminus origine (arr/dep) : bornes min/max
    arr_min: Dict[str, float]
    arr_max: Dict[str, float]
    dep_min: Dict[str, float]
    dep_max: Dict[str, float]


@dataclass
class OffsetsAlongLine:
    """Offsets (dep origin -> arr/dep stop) en bornes min/max."""
    arr_min: Dict[str, float]
    arr_max: Dict[str, float]
    dep_min: Dict[str, float]
    dep_max: Dict[str, float]


def _build_offsets_along_line(
    stops: List[str],
    run: List[SegmentRange],
    dwell_min: Dict[str, float],
    dwell_max: Dict[str, float],
) -> OffsetsAlongLine:
    """
    Construit des offsets bornés depuis un départ à stops[0] (origin):
      - arr_min/arr_max[s] : arrivée au stop s
      - dep_min/dep_max[s] : départ du stop s (après dwell)
    Hypothèses:
      - run a longueur len(stops)-1
      - dwell s’applique à tous les stops (0 si absent)
    """
    assert len(run) == len(stops) - 1, "run doit être de longueur len(stops)-1"

    arr_min: Dict[str, float] = {stops[0]: 0.0}
    arr_max: Dict[str, float] = {stops[0]: 0.0}
    dep_min: Dict[str, float] = {stops[0]: 0.0}
    dep_max: Dict[str, float] = {stops[0]: 0.0}

    tmin = 0.0
    tmax = 0.0

    for i in range(1, len(stops)):
        prev = stops[i - 1]
        cur  = stops[i]
        seg  = run[i - 1]

        # départ prev -> arrivée cur
        # (on suppose que dep(prev) est déjà inclus via dep_min/dep_max)
        tmin = dep_min[prev] + seg.t_min
        tmax = dep_max[prev] + seg.t_max

        arr_min[cur] = tmin
        arr_max[cur] = tmax

        dmin = float(dwell_min.get(cur, 0.0))
        dmax = float(dwell_max.get(cur, dmin))
        dmax = max(dmax, dmin)

        dep_min[cur] = tmin + dmin
        dep_max[cur] = tmax + dmax

    return OffsetsAlongLine(arr_min=arr_min, arr_max=arr_max, dep_min=dep_min, dep_max=dep_max)



from itertools import combinations
from typing import List


def _circular_gaps_ok(pattern: List[int], min_gap: float, period: int = 60) -> bool:
    """
    Vérifie que TOUTES les gaps circulaires entre départs successifs
    sont ≥ min_gap, sur un cercle de longueur 'period'.

    pattern : liste triée de minutes dans l'heure (0..period-1).
    """
    if len(pattern) <= 1:
        return True

    m = sorted(int(x) % period for x in pattern)
    for i in range(len(m) - 1):
        if (m[i + 1] - m[i]) < min_gap:
            return False

    # gap entre le dernier et le premier en faisant le tour
    if (m[0] + period - m[-1]) < min_gap:
        return False

    return True


def generate_minute_patterns(
    n_events: int,
    min_headway: float,
    period: int = 60,
) -> List[List[int]]:
    n = int(n_events)
    if n <= 0:
        return []

    if min_headway <= 0:
        min_headway = 0.0

    if n * min_headway > period + 1e-9:
        return []

    minutes = range(period)
    patterns: List[List[int]] = []

    for comb in combinations(minutes, n):
        pat = list(comb)
        if _circular_gaps_ok(pat, min_headway, period):
            patterns.append(pat)

    return patterns




def generate_arr_dep_patterns_at_A(
    line: LineConfig,
    arr_patterns: List[List[int]],
    min_headway_dep: float,
    dwell_A_min: int | None = None,
    dwell_A_max: int = 15,
    period: int = 60,
) -> List[TerminusPattern]:
    """
    Génère des quatuors cohérents à A sous forme de 2 paires (arr -> dep).

    Input:
      - arr_patterns: patterns d'arrivées à A (2/h) respectant headway sur les ARR
      - dwell_A_min: min dwell à A (si None: pris depuis JSON)
      - dwell_A_max: max dwell autorisé à A (pour limiter combinatoire)
      - min_headway_dep: headway minimal entre les 2 DEPARTS à A (circulaire)

    Output:
      - TerminusPattern(arr_minutes, dep_minutes, dwell_minutes)
        avec association indexée arr[i] -> dep[i].
    """
    A = line.terminus_A

    if dwell_A_min is None:
        dwell_list = line.dwell_times.get(A, [0.0]) or [0.0]
        dwell_A_min = int(round(min(dwell_list)))

    dwell_A_min = int(dwell_A_min)
    dwell_A_max = int(max(dwell_A_max, dwell_A_min))

    # liste discrète des dwells possibles
    dwell_values = list(range(dwell_A_min, dwell_A_max + 1))

    out: List[TerminusPattern] = []

    for arr_pat in arr_patterns:
        arr_sorted = sorted(int(x) % period for x in arr_pat)

        # on choisit un dwell pour chacune des 2 arrivées
        for dwells in product(dwell_values, repeat=len(arr_sorted)):
            dep = [ (a + int(d)) % period for a, d in zip(arr_sorted, dwells) ]

            # headway minimal sur les départs aussi
            if not _circular_gaps_ok(dep, min_headway_dep, period):
                continue

            out.append(
                TerminusPattern(
                    arr_minutes=arr_sorted,
                    dep_minutes=[int(x) % period for x in dep],
                    dwell_minutes=[int(d) for d in dwells],
                )
            )

    return out




def _pair_arr_dep_to_abs(arr_minutes, dep_minutes, dwell_minutes, period=60):
    arr_mod = [int(x) % period for x in arr_minutes]
    dep_mod = [int(x) % period for x in dep_minutes]
    dwell   = [int(x) for x in dwell_minutes]

    arr_abs = []
    dep_abs = []

    for a, d, dw in zip(arr_mod, dep_mod, dwell):
        a_abs = a
        d_abs = d

        # dep doit être après arr + dwell (autorise wrap)
        while d_abs < a_abs + dw:
            d_abs += period

        arr_abs.append(a_abs)
        dep_abs.append(d_abs)

    return arr_abs, dep_abs







def _intervals_overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
    """Overlap entre [a0,a1] et [b0,b1]."""
    lo = max(a0, b0)
    hi = min(a1, b1)
    return lo <= hi + 1e-9


from typing import List, Dict, Tuple

def filter_patterns_by_two_meets_conservative(
    line: LineConfig,
    patterns: List[TerminusPattern],
    forbidden_meet_stops: List[str],
    extra_terminus_slack: float = 0.0,
    enable_no_overtaking: bool = True,
) -> List[TerminusPattern]:
    """
    Filtre CONSERVATIF (pas de faux négatifs) mais PLUS FORT :

    On impose 2 croisements nécessaires (meet1 + meet2), avec test des 2 appariements possibles
    entre (dep0, dep1) et (arr0, arr1) afin d’éviter les faux négatifs.

    Optionnel: no-overtaking (nécessaire) -> élimine beaucoup les départs trop proches
    quand les temps internes sont variables.

    forbidden_meet_stops : gares interdites pour croiser.
    """
    A = line.terminus_A
    B = line.terminus_B
    period = int(line.period)

    forbidden = set(forbidden_meet_stops or [])
    candidate_meet_stops = [s for s in line.stops if s not in forbidden]
    if not candidate_meet_stops:
        return []

    # Offsets UP (A->B)
    up_off = _build_offsets_along_line(
        stops=line.stops,
        run=line.run_up,
        dwell_min=line.dwell_min,
        dwell_max=line.dwell_max,
    )

    # Offsets DOWN (B->A) : on inverse la ligne, et on applique run_down
    stops_rev = list(reversed(line.stops))
    down_off_rev = _build_offsets_along_line(
        stops=stops_rev,
        run=line.run_down,
        dwell_min=line.dwell_min,
        dwell_max=line.dwell_max,
    )

    # mapping stop -> offsets depuis un départ de B (sens B->A)
    down_arr_min_from_B = {s: down_off_rev.arr_min[s] for s in stops_rev}
    down_arr_max_from_B = {s: down_off_rev.arr_max[s] for s in stops_rev}
    down_dep_min_from_B = {s: down_off_rev.dep_min[s] for s in stops_rev}
    down_dep_max_from_B = {s: down_off_rev.dep_max[s] for s in stops_rev}

    # bornes totales DOWN (dep B -> arr A)
    down_total_min = float(down_arr_min_from_B[A])
    down_total_max = float(down_arr_max_from_B[A])

    dwell_B_min = float(line.dwell_min.get(B, 0.0))

    def presence_windows_up(depA: float) -> Dict[str, tuple[float, float]]:
        """Fenêtre présence conservatrice UP depuis départ A=depA : [arr_earliest, dep_latest]."""
        w: Dict[str, tuple[float, float]] = {}
        for s in line.stops:
            if s == A:
                w[s] = (depA, depA)  # point
            else:
                a0 = depA + float(up_off.arr_min[s])
                d1 = depA + float(up_off.dep_max[s])
                w[s] = (a0, d1)
        return w

    def presence_windows_down(depB_lo: float, depB_hi: float) -> Dict[str, tuple[float, float]]:
        """Fenêtre présence conservatrice DOWN pour depB dans [lo,hi] : union -> [earliest, latest]."""
        w: Dict[str, tuple[float, float]] = {}
        for s in line.stops:
            a0 = depB_lo + float(down_arr_min_from_B[s])
            d1 = depB_hi + float(down_dep_max_from_B[s])
            w[s] = (a0, d1)
        return w

    def has_meet(w1: Dict[str, tuple[float, float]], w2: Dict[str, tuple[float, float]]) -> bool:
        for s in candidate_meet_stops:
            if s not in w1 or s not in w2:
                continue
            a0, a1 = w1[s]
            b0, b1 = w2[s]
            if _intervals_overlap(a0, a1, b0, b1):
                return True
        return False

    def no_overtaking_up(dep0: float, dep1: float) -> bool:
        """
        Condition nécessaire conservative :
        le 2e UP ne doit pas pouvoir arriver avant le 1er à un stop (risque de rattrapage).
        On impose: earliest(dep1->arr s) >= latest(dep0->arr s) pour tous les stops.
        Si violé quelque part, on rejette (car overtaking possible).
        """
        for s in line.stops[1:]:  # exclut A
            earliest_1 = dep1 + float(up_off.arr_min[s])
            latest_0   = dep0 + float(up_off.arr_max[s])
            if earliest_1 < latest_0 - 1e-9:
                return False
        return True

    def no_overtaking_down(depB0_lo: float, depB0_hi: float, depB1_lo: float, depB1_hi: float) -> bool:
        """
        Condition nécessaire conservative sur DOWN:
        même idée: le 2e DOWN ne doit pas pouvoir “rattraper” le 1er.
        On compare (earliest of 2) vs (latest of 1) aux stops.
        Comme depB est un intervalle, on prend le cas le plus dangereux:
          - earliest du train2 = depB2_lo + arr_min
          - latest du train1   = depB1_hi + arr_max
        """
        # ordre : train0 puis train1 en DOWN
        for s in line.stops[:-1]:  # exclut B
            earliest_2 = depB1_lo + float(down_arr_min_from_B[s])
            latest_1   = depB0_hi + float(down_arr_max_from_B[s])
            if earliest_2 < latest_1 - 1e-9:
                return False
        return True

    kept: List[TerminusPattern] = []

    for pat in patterns:
        if len(pat.dep_minutes) != 2:
            # on se concentre sur ton cas 2 dép/h
            continue

        if len(pat.arr_minutes) != 2 or len(pat.dep_minutes) != 2:
            continue

        # (arr -> dep) en absolu, avec unwrap pour gérer 51->17
        arrA_abs, depA_abs = _pair_arr_dep_to_abs(
            arr_minutes=pat.arr_minutes,
            dep_minutes=pat.dep_minutes,
            dwell_minutes=pat.dwell_minutes,
            period=period,
        )

        # On veut 2 UP (départs) => on trie les dep_abs
        # et on permute arr_abs de la même manière si besoin.
        order_dep = sorted(range(2), key=lambda i: depA_abs[i])
        dep_sorted = [depA_abs[i] for i in order_dep]
        arr_sorted = [arrA_abs[i] for i in order_dep]  # utile si tu veux garder cohérence

        # on construit 2 appariements possibles entre {dep0,dep1} et {arrA0,arrA1}
        # afin de ne pas créer de faux négatifs.
        ok_any = False

        # On construit 2 UP : départs à A
        d0 = float(dep_sorted[0])
        d1 = float(dep_sorted[1])


        up0 = presence_windows_up(d0)
        up1 = presence_windows_up(d1)

        # Les 2 DOWN doivent arriver à A à arrA_abs = arrA_abs[i]
        # Ici, arrA_abs est par index "train" (association),
        # mais pour les meets, ce qui importe ce sont les 2 arrivals en absolu.
        a0 = float(arrA_abs[0])
        a1 = float(arrA_abs[1])

        # bornes sur depB pour chaque arrivée à A
        def depB_bounds_for_arrA_with_shifts(arrA_mod: float) -> List[tuple[float, float]]:
            """Intervalles depB possibles pour arrA_mod + k*period, k ∈ {-1,0,1}."""
            out: List[tuple[float, float]] = []
            for k in (-1, 0, 1):
                arrA = float(arrA_mod) + k * period
                lo = arrA - down_total_max
                hi = arrA - down_total_min
                if extra_terminus_slack > 0:
                    hi += float(extra_terminus_slack)
                if lo <= hi:
                    out.append((lo, hi))
            return out

        b0_list = depB_bounds_for_arrA_with_shifts(a0)  # a0 = pat.arr_minutes[0] (mod 60)
        b1_list = depB_bounds_for_arrA_with_shifts(a1)

        # s'il n'y a aucun shift possible, on skip
        if not b0_list or not b1_list:
            continue

        ok_any_local = False
        for depB0_lo, depB0_hi in b0_list:
            down0 = presence_windows_down(depB0_lo, depB0_hi)
            for depB1_lo, depB1_hi in b1_list:
                down1 = presence_windows_down(depB1_lo, depB1_hi)

                meet1 = has_meet(up0, down1)
                meet2 = has_meet(up1, down0)

                if meet1 and meet2:
                    ok_any_local = True
                    break
            if ok_any_local:
                break

        if ok_any_local:
            ok_any = True

        if ok_any:
            kept.append(pat)

    return kept




# =========================
# 4) Conversion en patterns horaires (par arrêt)
# =========================

def build_hourly_patterns_at_A(
    line: LineConfig,
    term_patterns: List[TerminusPattern],
) -> List[Dict[str, Dict[str, List[int]]]]:
    """
    Convertit une liste de TerminusPattern en structures de type
    'st_hourly_patterns' compatibles avec l'optimizer.

    Pour chaque TerminusPattern, on construit un dict :

        {
          "<terminus_A>": {
              "arr": [.. minutes dans l'heure ..],
              "dep": [.. minutes dans l'heure ..],
          }
        }

    Pour l'instant :
      - On ne met QUE le terminus A (Martigny),
      - On ignore les autres arrêts et le terminus B.
        -> suffisant tant que les transferts ne se font qu'à Martigny.
      - On garde la correspondance 1-à-1 entre départs et durées de cycle,
        mais ici on ne transmet que les minutes dans l'heure.
    """
    terminus_A = line.terminus_A
    period = line.period

    hourly_patterns: List[Dict[str, Dict[str, List[int]]]] = []

    for pat in term_patterns:
        # On replie proprement dans [0, period-1] (même si c'est déjà le cas)
        dep = sorted(int(x) % period for x in pat.dep_minutes)
        arr = sorted(int(x) % period for x in pat.arr_minutes)

        combo_pat: Dict[str, Dict[str, List[int]]] = {
            terminus_A: {
                "dep": dep,
                "arr": arr,
            }
        }
        hourly_patterns.append(combo_pat)

    return hourly_patterns




# =========================
# 5) Pipeline complet pour singletrack V2
# =========================

def compute_singletrack_hourly_patterns(
    config_path: str,
    n_dep_per_hour: int,
    min_headway: float,
    extra_terminus_slack: float = 0.0,
    forbidden_meet_stops: List[str] | None = None,
    enable_no_overtaking: bool = True,
) -> Tuple[LineConfig, List[Dict[str, Dict[str, List[int]]]], List[TerminusPattern], int]:


    """
    Pipeline complet V2 à partir d'un JSON singletrack :

      1) charge la ligne (LineConfig),
      2) calcule les bornes de cycle (cycle_min, cycle_max),
      3) génère tous les patterns de DEPART possible à A
         (n_dep_per_hour, headway circulaire >= min_headway),
      4) pour chaque pattern de départ, génère les combos (dep,arr) à A
         compatibles avec les bornes de cycle,
      5) convertit ces TerminusPattern en 'hourly patterns'
         au format attendu par l'optimizer (st_hourly_patterns).

    Pour l'instant :
      - On ne renseigne que le terminus A (Martigny) dans les patterns horaires.
      - Aucun filtrage encore sur les croisements / dépassements internes :
        ce sera la prochaine étape.

    Retourne :
      - line        : le LineConfig
      - hourly_pats : liste de dicts { stop -> {"arr":[..], "dep":[..]} }
      - period      : la période de cadencement (en minutes)
    """
    # 1) Charger la ligne
    line = load_line_config(config_path)



    # 2) Patterns d'ARRIVÉES à A
    arr_patterns = generate_minute_patterns(
        n_events=n_dep_per_hour,  # 2 arrivals/h
        min_headway=min_headway,  # même headway que les départs (tu peux séparer après)
        period=line.period,
    )

    print("arr_patterns =", len(arr_patterns))


    # 3) Convertir (arr -> dwell -> dep) en TerminusPattern
    term_patterns = generate_arr_dep_patterns_at_A(
        line=line,
        arr_patterns=arr_patterns,
        min_headway_dep=min_headway,  # headway aussi sur dep
        dwell_A_min=None,  # pris depuis JSON (=3)
        dwell_A_max=int(round(line.dwell_max.get(line.terminus_A, 15))),
        period=line.period,
    )

    if forbidden_meet_stops is None:
        forbidden_meet_stops = getattr(line, "forbidden_meet_stops", []) or []



    print("term_patterns (avant filtre meets) =", len(term_patterns))

    term_patterns = filter_patterns_by_two_meets_conservative(
        line=line,
        patterns=term_patterns,
        forbidden_meet_stops=forbidden_meet_stops or [],
        extra_terminus_slack=extra_terminus_slack,
        enable_no_overtaking=enable_no_overtaking,
    )
    print("forbidden_meet_stops =", forbidden_meet_stops)

    print("term_patterns (après filtre meets) =", len(term_patterns))
    # 4) Conversion pour l'optimizer
    hourly_pats = build_hourly_patterns_at_A(line, term_patterns)
    print("hourly_pats (envoyés à optimizer) =", len(hourly_pats))


    return line, hourly_pats, term_patterns, line.period







# =========================
# 5) reconstruction/verification
# =========================





def _strict_overlap(a0: float, a1: float, b0: float, b1: float, eps: float = 1e-9) -> bool:
    # overlap strict (autorise contact à l’extrémité)
    return max(a0, b0) < min(a1, b1) - eps

def _stop_presence_windows(times_arr: dict, times_dep: dict) -> dict[str, tuple[float, float]]:
    # présence au stop = [arr, dep]
    return {s: (float(times_arr[s]), float(times_dep[s])) for s in times_arr.keys()}

def _compute_trip_times_A_to_B(
    stops: list[str],
    run_times: list[int],
    dwell_choice: dict[str, int],
    depA: int,
    terminus_B: str,
    dwellB: int,
) -> tuple[dict[str, int], dict[str, int], int]:
    """
    Construit horaires UP: depA à A -> ... -> B (arr/dep).
    Retourne (arr_times, dep_times, depB).
    Convention:
      - A : arr=dep=depA (point)
      - chaque stop intermédiaire s: dep = arr + dwell_choice[s]
      - B : depB = arrB + dwellB (turnaround)
    """
    A = stops[0]
    arr = {A: int(depA)}
    dep = {A: int(depA)}
    t = int(depA)
    for k in range(len(stops) - 1):
        s0 = stops[k]
        s1 = stops[k+1]
        t = int(dep[s0]) + int(run_times[k])
        arr[s1] = t
        if s1 == terminus_B:
            dep[s1] = t + int(dwellB)
        else:
            dep[s1] = t + int(dwell_choice.get(s1, 0))
    return arr, dep, int(dep[terminus_B])

def _compute_trip_times_B_to_A(
    stops: list[str],
    run_times_down: list[int],
    dwell_choice: dict[str, int],
    depB: int,
    terminus_A: str,
    terminus_B: str,
) -> tuple[dict[str, int], dict[str, int], int]:
    """
    Construit horaires DOWN: depB à B -> ... -> A (arr/dep).
    Convention:
      - B : arr=dep=depB (point)
      - chaque stop intermédiaire s: dep = arr + dwell_choice[s]
      - A : arrA est retourné, depA_down non utilisé (on met dep[A]=arr[A])
    """
    stops_rev = list(reversed(stops))
    # down run_times est aligné sur (B->...->A) donc longueur len(stops)-1
    B = terminus_B
    arr = {B: int(depB)}
    dep = {B: int(depB)}
    for k in range(len(stops_rev) - 1):
        s0 = stops_rev[k]
        s1 = stops_rev[k+1]
        # segment k correspond au tronçon s0->s1 en sens DOWN
        t = int(dep[s0]) + int(run_times_down[k])
        arr[s1] = t
        if s1 == terminus_A:
            dep[s1] = t
        else:
            dep[s1] = t + int(dwell_choice.get(s1, 0))
    return arr, dep, int(arr[terminus_A])

def _segment_intervals_from_times(
    stops: list[str],
    arr_times: dict[str, int],
    dep_times: dict[str, int],
) -> list[tuple[int, float, float]]:
    """
    Retourne une liste d’intervalles (seg_index, t_start, t_end) pour chaque tronçon k:
      [dep(stop[k]), arr(stop[k+1])]
    """
    out = []
    for k in range(len(stops)-1):
        s0 = stops[k]
        s1 = stops[k+1]
        out.append((k, float(dep_times[s0]), float(arr_times[s1])))
    return out

def _shift_segs(segs: list[tuple[int, float, float]], dt: int) -> list[tuple[int, float, float]]:
    return [(k, a0 + dt, a1 + dt) for (k, a0, a1) in segs]




from itertools import product

def validate_and_reconstruct_exact(
    line: LineConfig,
    pat: TerminusPattern,
    forbidden_meet_stops: list[str] | None = None,
    enable_no_overtaking: bool = True,
    require_two_meets: bool = True, debug_print: bool = False) -> tuple[bool, dict]:


    A = line.terminus_A
    B = line.terminus_B
    P = int(line.period)

    n = len(pat.dep_minutes)
    if n <= 0 or len(pat.arr_minutes) != n or len(pat.dwell_minutes) != n:
        return False, {"status": "fail", "reason": "pattern length mismatch"}

    forbidden = set((forbidden_meet_stops if forbidden_meet_stops is not None else (line.forbidden_meet_stops or [])) or [])
    meet_candidates = [s for s in line.stops if s not in forbidden]
    if not meet_candidates:
        return False, {"status": "fail", "reason": "no meet candidate stops (all forbidden)"}

    # 1) arr/dep absolus à A
    arrA_abs, depA_abs = _pair_arr_dep_to_abs(
        arr_minutes=pat.arr_minutes,
        dep_minutes=pat.dep_minutes,
        dwell_minutes=pat.dwell_minutes,
        period=P,
    )
    target_arrA = [int(a + P) for a in arrA_abs]

    # 2) Discrétisations
    inter_stops = [s for s in line.stops[1:-1]]

    dwell_options = {}
    for s in inter_stops:
        vals = line.dwell_times.get(s, [0.0]) or [0.0]
        dwell_options[s] = sorted({int(round(v)) for v in vals})

    dwellB_min = int(round(line.dwell_min.get(B, 0.0)))
    dwellB_max = int(round(line.dwell_max.get(B, dwellB_min)))
    if dwellB_max < dwellB_min:
        dwellB_max = dwellB_min
    dwellB_values = list(range(dwellB_min, dwellB_max + 1))

    run_up_choices = list(product(*[tuple(vs) for vs in line.run_values_up]))
    run_dn_choices = list(product(*[tuple(vs) for vs in line.run_values_down]))

    if inter_stops:
        dwell_inter_choices = list(product(*[tuple(dwell_options[s]) for s in inter_stops]))
    else:
        dwell_inter_choices = [()]

    def make_dwell_choice(tup) -> dict[str, int]:
        return {s: int(v) for s, v in zip(inter_stops, tup)}

    # --------
    # BLOC A : options par train
    # --------
    options_per_train: list[list[dict]] = [[] for _ in range(n)]

    for i in range(n):
        depA = int(depA_abs[i])
        tgtA = int(target_arrA[i])



        for run_up in run_up_choices:
            for run_dn in run_dn_choices:
                for dw_inter_tup in dwell_inter_choices:
                    dwell_choice = make_dwell_choice(dw_inter_tup)
                    for dwellB in dwellB_values:

                        up_arr, up_dep, depB = _compute_trip_times_A_to_B(
                            stops=line.stops,
                            run_times=list(run_up),
                            dwell_choice=dwell_choice,
                            depA=depA,
                            terminus_B=B,
                            dwellB=int(dwellB),
                        )

                        dn_arr, dn_dep, arrA = _compute_trip_times_B_to_A(
                            stops=line.stops,
                            run_times_down=list(run_dn),
                            dwell_choice=dwell_choice,
                            depB=depB,
                            terminus_A=A,
                            terminus_B=B,
                        )

                        if int(arrA) != tgtA:
                            continue

                        up_segs = _segment_intervals_from_times(line.stops, up_arr, up_dep)
                        stops_rev = list(reversed(line.stops))
                        dn_segs_rev = _segment_intervals_from_times(stops_rev, dn_arr, dn_dep)
                        L = len(line.stops) - 1
                        dn_segs = [(L-1-k, t0, t1) for (k, t0, t1) in dn_segs_rev]

                        options_per_train[i].append({
                            "i": i,
                            "depA": depA,
                            "arrA": int(arrA),
                            "up_arr": up_arr, "up_dep": up_dep,
                            "dn_arr": dn_arr, "dn_dep": dn_dep,
                            "up_segs": up_segs,
                            "dn_segs": dn_segs,
                            "run_up": list(map(int, run_up)),
                            "run_down": list(map(int, run_dn)),
                            "dwell_inter": dict(dwell_choice),
                            "dwellB": int(dwellB),
                        })

        if not options_per_train[i]:
            return False, {"status": "fail", "reason": f"train {i}: no internal realization matches target_arrA"}

    # --------
    # BLOC B : assemblage (backtracking simple)
    # --------
    chosen: list[dict] = []

    def ok_no_overtaking(trajs: list[dict]) -> bool:
        if not enable_no_overtaking or len(trajs) < 2:
            return True
        # UP : ordre par depA
        order = sorted(range(len(trajs)), key=lambda k: trajs[k]["depA"])
        for s in line.stops[1:]:
            for a, b in zip(order[:-1], order[1:]):
                if trajs[a]["up_arr"][s] > trajs[b]["up_arr"][s]:
                    return False
        # DOWN : ordre par départ à B (dn_dep[B])
        order_dn = sorted(range(len(trajs)), key=lambda k: trajs[k]["dn_dep"][B])
        for s in reversed(line.stops[:-1]):
            for a, b in zip(order_dn[:-1], order_dn[1:]):
                if trajs[a]["dn_arr"][s] > trajs[b]["dn_arr"][s]:
                    return False
        return True

    def ok_segments(trajs: list[dict]) -> bool:
        # check conflicts opposés UP(i) vs DOWN(j) en tenant compte des copies périodiques
        shifts = (-P, 0, +P)  # élargis à (-2P,-P,0,+P,+2P) si besoin

        for ii in range(len(trajs)):
            for jj in range(len(trajs)):
                if ii == jj:
                    continue

                up0 = trajs[ii]["up_segs"]
                dn0 = trajs[jj]["dn_segs"]

                for su in shifts:
                    up = _shift_segs(up0, su)
                    for sd in shifts:
                        dn = _shift_segs(dn0, sd)

                        for (k, a0, a1) in up:
                            for (kk, b0, b1) in dn:
                                if kk != k:
                                    continue
                                if _strict_overlap(a0, a1, b0, b1):
                                    return False
        return True

    def meet_stop(i_up: dict, j_dn: dict) -> str | None:
        shifts = (-P, 0, +P)

        w_up = _stop_presence_windows(i_up["up_arr"], i_up["up_dep"])

        for sd in shifts:
            dn_arr_s = {s: (t + sd) for s, t in j_dn["dn_arr"].items()}
            dn_dep_s = {s: (t + sd) for s, t in j_dn["dn_dep"].items()}
            w_dn = _stop_presence_windows(dn_arr_s, dn_dep_s)

            for s in meet_candidates:
                a0, a1 = w_up.get(s, (None, None))
                b0, b1 = w_dn.get(s, (None, None))
                if a0 is None or b0 is None:
                    continue
                # overlap ou contact
                if _strict_overlap(a0, a1, b0, b1) or abs(max(a0, b0) - min(a1, b1)) < 1e-9:
                    return s

        return None

    def ok_meets(trajs: list[dict]) -> tuple[bool, list[str | None]]:
        if not require_two_meets:
            return True, []
        if n != 2:
            return False, []
        m1 = meet_stop(trajs[0], trajs[1])  # UP0 vs DN1
        m2 = meet_stop(trajs[1], trajs[0])  # UP1 vs DN0
        return (m1 is not None and m2 is not None), [m1, m2]

    def backtrack(i: int) -> tuple[bool, dict]:
        if i == n:
            if not ok_no_overtaking(chosen):
                return False, {}
            if not ok_segments(chosen):
                return False, {}
            okm, meets = ok_meets(chosen)
            if not okm:
                return False, {}
            # succès
            return True, {"meets": meets}

        for opt in options_per_train[i]:
            chosen.append(opt)
            ok, det = backtrack(i + 1)
            if ok:
                return True, det
            chosen.pop()
        return False, {}

    ok, det = backtrack(0)
    if not ok:
        return False, {"status": "fail", "reason": "no feasible internal reconstruction found (exact search)"}

    details = {
        "status": "ok",
        "n_trains": n,
        "meet_stops": det.get("meets", []),
        "A": A, "B": B, "period": P,
        "trains": [
            {
                "i": tr["i"],
                "depA": tr["depA"],
                "arrA": tr["arrA"],
                "run_up": tr["run_up"],
                "run_down": tr["run_down"],
                "dwell_inter": tr["dwell_inter"],
                "dwellB": tr["dwellB"],
                "up_arr": tr["up_arr"],
                "up_dep": tr["up_dep"],
                "dn_arr": tr["dn_arr"],
                "dn_dep": tr["dn_dep"],
            }
            for tr in chosen
        ],
    }

    if debug_print:
        print("\n[SINGLETRACK EXACT] FOUND FEASIBLE INTERNAL RECONSTRUCTION")
        print(f"  meet_stops (if computed) = {det.get('meets', [])}")
        for tr in chosen:
            _print_solution_compact(line, chosen, show_run_times=True)

    return True, details





def validate_singletrack_candidate(
    config_path: str,
    term_pattern: TerminusPattern | None = None,
    forbidden_meet_stops: list[str] | None = None,
    enable_no_overtaking: bool = True,
    require_two_meets: bool = True,
    debug_print: bool = False,   # <-- ajout
    **kwargs,
) -> tuple[bool, dict]:

    if term_pattern is None:
        return False, {"status": "fail", "reason": "term_pattern is None"}

    line = load_line_config(config_path)

    ok, details = validate_and_reconstruct_exact(
        line=line,
        pat=term_pattern,
        forbidden_meet_stops=forbidden_meet_stops,
        enable_no_overtaking=enable_no_overtaking,
        require_two_meets=require_two_meets,
        debug_print=debug_print,   # <-- piloté par optimizer
    )
    return bool(ok), details


# --- compact debug printing: ONLY modulo times + per-stop arrivals/departures ---
from collections import defaultdict
from typing import Dict, List, Tuple

def _m(t: int, period: int = 60) -> int:
    return int(t) % int(period)

def _fmt_mod(t: int, period: int = 60) -> str:
    # "07" style (minute in hour)
    return f"{_m(t, period):02d}"

def _collect_events_by_stop(line: LineConfig, tr: dict):
    """
    Evénements "physiques" uniquement :
      - UP : à A -> DEP seulement ; à B -> ARR+DEP ; ailleurs ARR+DEP
      - DOWN : à B -> DEP seulement ; à A -> ARR seulement ; ailleurs ARR+DEP
    """
    P = int(line.period)
    A = line.terminus_A
    B = line.terminus_B

    events = {s: {"arr": [], "dep": []} for s in line.stops}

    tid = tr.get("i", "?")

    up_arr = tr.get("up_arr", {}) or {}
    up_dep = tr.get("up_dep", {}) or {}
    dn_arr = tr.get("dn_arr", {}) or {}
    dn_dep = tr.get("dn_dep", {}) or {}

    def add_arr(stop: str, tag: str, t: int):
        events[stop]["arr"].append((tag, int(t) % P))

    def add_dep(stop: str, tag: str, t: int):
        events[stop]["dep"].append((tag, int(t) % P))

    # ---------- UP ----------
    for s in line.stops:
        tag = f"T{tid}U"
        if s == A:
            # à A: seulement départ UP
            if s in up_dep: add_dep(s, tag, up_dep[s])
        elif s == B:
            # à B: arrivée + départ (turnaround)
            if s in up_arr: add_arr(s, tag, up_arr[s])
            if s in up_dep: add_dep(s, tag, up_dep[s])
        else:
            # intermédiaire
            if s in up_arr: add_arr(s, tag, up_arr[s])
            if s in up_dep: add_dep(s, tag, up_dep[s])

    # ---------- DOWN ----------
    for s in line.stops:
        tag = f"T{tid}D"
        if s == B:
            # à B: seulement départ DOWN (départ du retour)
            if s in dn_dep: add_dep(s, tag, dn_dep[s])
        elif s == A:
            # à A: seulement arrivée finale DOWN
            if s in dn_arr: add_arr(s, tag, dn_arr[s])
        else:
            # intermédiaire
            if s in dn_arr: add_arr(s, tag, dn_arr[s])
            if s in dn_dep: add_dep(s, tag, dn_dep[s])

    # tri
    for s in line.stops:
        events[s]["arr"].sort(key=lambda x: (x[1], x[0]))
        events[s]["dep"].sort(key=lambda x: (x[1], x[0]))
    return events


def _print_solution_compact(line: LineConfig, trains: List[dict], show_run_times: bool = True) -> None:
    """
    Prints a compact view to visually check meetings/conflicts:
      - per train: run_up/run_down + dwellB + dwell_inter (optional)
      - per stop: ALL arrivals and departures (modulo only), tagged by train + direction
    """
    P = int(line.period)

    print("\n" + "=" * 80)
    print(f"COMPACT SINGLETRACK DEBUG  | period={P}  | stops={line.stops}")
    print("=" * 80)

    if show_run_times:
        print("\n--- Train internal choices (run times / dwells) ---")
        for tr in trains:
            tid = tr.get("i", "?")
            ru = tr.get("run_up")
            rd = tr.get("run_down")
            di = tr.get("dwell_inter", {})
            db = tr.get("dwellB", None)
            print(f"  T{tid}: run_up={ru}  run_down={rd}  dwellB={db}  dwell_inter={di}")

    # Aggregate events across trains
    agg: Dict[str, Dict[str, List[Tuple[str,int]]]] = {s: {"arr": [], "dep": []} for s in line.stops}
    for tr in trains:
        ev = _collect_events_by_stop(line, tr)
        for s in line.stops:
            agg[s]["arr"].extend(ev[s]["arr"])
            agg[s]["dep"].extend(ev[s]["dep"])

    # Sort aggregated
    for s in line.stops:
        agg[s]["arr"].sort(key=lambda x: (x[1], x[0]))
        agg[s]["dep"].sort(key=lambda x: (x[1], x[0]))

    print("\n--- Per-stop events (minute-in-hour only) ---")
    for s in line.stops:
        arr_str = " ".join([f"{t:02d}({tag})" for tag, t in agg[s]["arr"]]) or "-"
        dep_str = " ".join([f"{t:02d}({tag})" for tag, t in agg[s]["dep"]]) or "-"
        print(f"\n{s}:")
        print(f"  ARR: {arr_str}")
        print(f"  DEP: {dep_str}")

def _print_single_train_compact(line: LineConfig, tr: dict) -> None:
    """
    If you still want a per-train compact view (modulo only) without segment intervals.
    """
    P = int(line.period)
    tid = tr.get("i", "?")
    print("\n" + "-" * 60)
    print(f"T{tid}  run_up={tr.get('run_up')}  run_down={tr.get('run_down')}  dwellB={tr.get('dwellB')}  dwell_inter={tr.get('dwell_inter')}")
    print("-" * 60)

    for s in line.stops:
        ua = tr["up_arr"].get(s); ud = tr["up_dep"].get(s)
        da = tr["dn_arr"].get(s); dd = tr["dn_dep"].get(s)
        # print only what's available, modulo
        parts = []
        if ua is not None: parts.append(f"U.arr={_fmt_mod(ua,P)}")
        if ud is not None: parts.append(f"U.dep={_fmt_mod(ud,P)}")
        if da is not None: parts.append(f"D.arr={_fmt_mod(da,P)}")
        if dd is not None: parts.append(f"D.dep={_fmt_mod(dd,P)}")
        if parts:
            print(f"{s:12s}  " + "  ".join(parts))

# -----------------
# How to use in your validate_and_reconstruct_exact debug_print block:
#
#   if debug_print:
#       _print_solution_compact(line, chosen, show_run_times=True)
#
# Or if you want per train:
#   for tr in chosen:
#       _print_single_train_compact(line, tr)
# -----------------
