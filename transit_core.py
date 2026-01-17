# transit_core.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from math import inf

# =========================
# 1) Time & window helpers
# =========================

WINDOW_START: int = 0   # minutes since 00:00 (global)
WINDOW_END:   int = 24*60

# Marge optionnelle (minutes) autour de [WINDOW_START, WINDOW_END)
EVAL_GRACE_BEFORE: int = 0
EVAL_GRACE_AFTER: int = 0

def set_eval_grace(before_min: int = 0, after_min: int = 0) -> None:
    global EVAL_GRACE_BEFORE, EVAL_GRACE_AFTER
    EVAL_GRACE_BEFORE = max(0, int(before_min))
    EVAL_GRACE_AFTER  = max(0, int(after_min))

def set_window(start_hhmm: str, end_hhmm: str) -> None:
    """Set global time window [start,end) in absolute minutes."""
    global WINDOW_START, WINDOW_END
    WINDOW_START = hhmm_to_min(start_hhmm)
    WINDOW_END   = hhmm_to_min(end_hhmm)

def hhmm_to_min(s: str) -> int:
    s = s.strip()
    h, m = s.split(":")
    return int(h)*60 + int(m)

def min_to_hhmm(t: int) -> str:
    t = int(t) % (24*60)
    return f"{t//60:02d}:{t%60:02d}"

def generate_hourly_times(minutes_in_hour: List[int], start_min: int, end_min: int) -> List[int]:
    """Repeat given minutes inside each hour over [start_min, end_min)."""
    minutes = sorted({int(x) % 60 for x in minutes_in_hour})
    times: List[int] = []
    first_hour = start_min // 60
    last_hour  = (end_min - 1) // 60
    for h in range(first_hour, last_hour + 1):
        base = h * 60
        for mm in minutes:
            t = base + mm
            if start_min <= t < end_min:
                times.append(t)
    return sorted(times)

def next_time(times: List[int], t_candidate: int) -> Optional[int]:
    """First time >= t_candidate (times sorted)."""
    # Could be bisect for perf; linear is fine for now.
    for t in times:
        if t >= t_candidate:
            return t
    return None

def prev_time(times: List[int], t_candidate: int) -> Optional[int]:
    """Last time <= t_candidate (times sorted)."""
    prev = None
    for t in times:
        if t > t_candidate:
            break
        prev = t
    return prev

# Padding (minutes) pour étendre la génération au-delà de la fenêtre
PADDING_MIN: int = 120  # tu peux mettre 90/120 selon tes cas

def set_padding(minutes: int) -> None:
    global PADDING_MIN
    PADDING_MIN = max(0, int(minutes))

def _padded_window() -> tuple[int, int]:
    return (WINDOW_START - PADDING_MIN, WINDOW_END + PADDING_MIN)

def compute_recommended_padding(services: Dict[str, "Service"], transfers: List["Transfer"]) -> int:
    max_cycle_or_tt = 0.0
    for s in services.values():
        t = s.template
        if not t:
            continue
        if t.topology == "loop" and t.cycle_time:
            max_cycle_or_tt = max(max_cycle_or_tt, float(t.cycle_time))
        elif t.topology == "line":
            if t.tt_AB: max_cycle_or_tt = max(max_cycle_or_tt, float(t.tt_AB))
            if t.tt_BA: max_cycle_or_tt = max(max_cycle_or_tt, float(t.tt_BA))
        elif t.topology == "route":
            # somme des legs + somme des dwells = cycle total
            legs  = sum(int(x) for x in (t.leg_minutes or []))
            dw    = sum(int(v) for v in (t.dwells or {}).values())
            max_cycle_or_tt = max(max_cycle_or_tt, float(legs + dw))

    max_walk_margin = 0.0
    for tr in transfers:
        max_walk_margin = max(max_walk_margin, float(tr.walk_time) + float(tr.min_margin))

    return int(60 + max_cycle_or_tt + max_walk_margin + 15)



# =========================
# 2) Data model
# =========================

@dataclass
class Anchor:
    """A physical interchange point of a Service (station/terminus/stop)."""
    name: str
    arrivals: List[int] = field(default_factory=list)
    departures: List[int] = field(default_factory=list)
    arrival_weights: List[float] = field(default_factory=list)   # same len as arrivals (auto=1.0)
    departure_weights: List[float] = field(default_factory=list) # same len as departures (auto=1.0)

    def ensure_weights(self) -> None:
        if not self.arrival_weights:
            self.arrival_weights = [1.0] * len(self.arrivals)
        if not self.departure_weights:
            self.departure_weights = [1.0] * len(self.departures)

@dataclass
class ServiceTemplate:
    """Parameters for services that need schedule generation / optimization."""
    # Topology: "scheduled" (no generation), "loop", or "line"
    topology: str                               # "scheduled" | "loop" | "line"
    # Common cadence params
    n_per_hour: int = 0                         # candidates per hour (if 0 → infer by H_min later if needed)
    equal_headway: bool = False                 # if True, use regular pattern (offset search)
    # Loop params
    cycle_time: Optional[float] = None          # minutes: a full cycle back to same anchor
    turnaround: Optional[float] = None          # minutes: dwell/buffer at anchor for loop or line endpoints
    # Line params (bidirectional A↔B)
    anchors: Optional[List[str]] = None         # must be length 2 for "line"
    tt_AB: Optional[float] = None               # run time A→B
    tt_BA: Optional[float] = None               # run time B→A
    turnaround_A: Optional[float] = None        # dwell @A
    turnaround_B: Optional[float] = None        # dwell @B
    # Phase between terminals (minutes at B = minutes at A + phi mod 60)
    phi_minutes: Optional[int] | str = None     # int or "optimize"

    stops: list[str] | None = None  # ex: ["Aminona","Montana_gare_in","CransForest","Montana_gare_out"]
    leg_minutes: list[int] | None = None  # longueurs = len(stops)  (dernier -> premier compris)
    dwells: dict[str, int] | None = None  # temps de rebroussement/arrêt par stop (par défaut 0)

    # Offset optimisation (pour topology="scheduled")
    base_offset_min: int = 0  # centre de la recherche (en minutes)
    offset_range_min: int = 0  # demi-plage : si 0 → pas d’optimisation par offset
    offset_step_min: int = 1  # pas entre deux offsets testés

    # --- Singletrack : config pour appeler le modèle local ---
    st_config_path: Optional[str] = None  # ex: "config_singletrack_3.json"
    st_period_min: int = 60  # en principe 60

    # On garde les arrêts de la ligne telle que définie dans le modèle single-track
    st_line_stops: List[str] = field(default_factory=list)

    # Liste de patterns horaires par option (A seul, A+B, etc.)
    # st_hourly_patterns[k][stop] = {"arr": [..min..], "dep": [..min..]} (minutes dans l'heure)
    st_hourly_patterns: List[Dict[str, Dict[str, List[int]]]] = field(default_factory=list)

    # NOUVEAU : paramètres V2 pour la génération singletrack
    st_n_dep_per_hour: int = 2  # nombre de départs/h à l’ancre A
    st_min_headway: float = 5.0  # headway minimal entre départs à A (en minutes)
    st_forbidden_meet_stops: List[str] = field(default_factory=list)
    st_enable_no_overtaking: bool = True
    st_extra_terminus_slack: float = 0.0
    # --- Singletrack debug / validation ---
    st_term_patterns: list = field(default_factory=list)  # liste de TerminusPattern alignée avec st_hourly_patterns
    st_line_config: Any = None  # LineConfig chargé (optionnel mais pratique)


@dataclass
class Service:
    """A transport service (train/bus/funicular/...) with 1+ anchors."""
    name: str
    category: str                       # "train"|"bus"|"funi"|...
    anchors: Dict[str, Anchor]          # at least {"default": Anchor(...)} for simple cases
    template: Optional[ServiceTemplate] = None
    # For scheduled services from streams
    pattern_streams: List[Dict] = field(default_factory=list)  # stays compatible with your JSON

    def get_anchor(self, name: Optional[str]) -> Anchor:
        key = name or "default"
        if key not in self.anchors:
            self.anchors[key] = Anchor(name=key)
        return self.anchors[key]

@dataclass
class Transfer:
    """A transfer between two specific anchors."""
    from_service: str
    from_anchor: str
    to_service: str
    to_anchor: str
    walk_time: float
    min_margin: float
    w_up: float = 0.5
    w_down: float = 0.5
    tie_breaker: str = "none"
    tie_params: Dict = field(default_factory=dict)

# =========================
# 3) Streams: scheduled & pattern_streams
# =========================

def _sort_times_with_weights(times: List[int], weights: List[float]) -> Tuple[List[int], List[float]]:
    pairs = sorted(zip(times, weights), key=lambda x: x[0])
    return [t for t,_ in pairs], [w for _,w in pairs]

def expand_pattern_streams(service: Service, anchor_name: Optional[str] = None) -> None:
    """
    Déploie les 'pattern_streams' d'un service en horaires absolus (arrivées/départs).

    Chaque stream représente une ligne/direction cadencée et contient :
      {
        "label": "nom ou direction",
        "depart_minutes": [8, 36],     # minutes dans l'heure
        "arr_offset_min": -1,          # décalage (arrivée - départ) en minutes
        "w": 1.0                       # poids global, ou "w_arr"/"w_dep" distincts
      }

    Cette fonction :
      - répète ces minutes dans l’heure sur la fenêtre étendue (padding inclus),
      - applique le décalage d’arrivée,
      - fusionne tous les streams dans un seul anchor,
      - trie les événements et garantit que les poids sont cohérents.

    Si aucun pattern_stream n’est défini, la fonction ne fait rien.
    """
    if not service.pattern_streams:
        return

    # Récupère ou crée l’ancrage (souvent "default")
    anc = service.get_anchor(anchor_name)

    # Listes cumulatives
    dep_all: List[int] = []
    arr_all: List[int] = []
    w_dep_all: List[float] = []
    w_arr_all: List[float] = []

    # Fenêtre étendue pour éviter les effets de bord (ex: correspondances en dehors)
    start, end = _padded_window()

    for s in service.pattern_streams:
        # 1) Récupère les minutes de départ et normalise (0–59)
        if "depart_minutes" not in s:
            raise ValueError(f"pattern_stream sans clé 'depart_minutes' dans {s}")
        dep_minutes = sorted({int(x) % 60 for x in s["depart_minutes"]})

        # 2) Décalage d'arrivée (positif ou négatif)
        arr_off = int(s.get("arr_offset_min", 0))

        # 3) Pondérations (w commun, ou séparé w_arr/w_dep)
        w_arr = float(s.get("w_arr", s.get("w", 1.0)))
        w_dep = float(s.get("w_dep", s.get("w", 1.0)))

        # 4) Génère tous les horaires de départs dans la fenêtre étendue
        dep_times = generate_hourly_times(dep_minutes, start, end)

        # 5) Applique le décalage pour les arrivées
        arr_times = [t + arr_off for t in dep_times]

        # 6) Ajoute aux listes cumulatives
        dep_all.extend(dep_times)
        arr_all.extend(arr_times)
        w_dep_all.extend([w_dep] * len(dep_times))
        w_arr_all.extend([w_arr] * len(arr_times))

    # 7) Trie et aligne les poids
    anc.departures, anc.departure_weights = _sort_times_with_weights(dep_all, w_dep_all)
    anc.arrivals,   anc.arrival_weights   = _sort_times_with_weights(arr_all, w_arr_all)

    # 8) Vérification finale
    anc.ensure_weights()


# =========================
# 4) Cadence generators
# =========================

def circular_gaps_ok(minutes: List[int], min_gap: float) -> bool:
    """Check circular gaps (0..59) >= min_gap."""
    m = sorted({int(x) % 60 for x in minutes})
    if len(m) <= 1:
        return True
    for i in range(len(m)-1):
        if (m[i+1] - m[i]) < min_gap:
            return False
    if (m[0] + 60 - m[-1]) < min_gap:
        return False
    return True

def headway_min_loop(tmpl: ServiceTemplate) -> float:
    """H_min for loops = cycle_time + turnaround (minimum safe separation at anchor)."""
    cycle = float(tmpl.cycle_time or 0)
    turn  = float(tmpl.turnaround or 0)
    return max(1.0, cycle + turn)

def headway_min_line(tmpl: ServiceTemplate) -> float:
    """
    A conservative feasibility bound for bidirectional line at an anchor.
    We use min of endpoint constraints; keep simple & safe for candidate pruning.
    """
    # This can be refined; we keep a safe positive lower bound
    min_tt = min(float(tmpl.tt_AB or 0), float(tmpl.tt_BA or 0))
    min_turn = min(float(tmpl.turnaround_A or 0), float(tmpl.turnaround_B or 0))
    return max(1.0, min_tt + min_turn)

def generate_equal_headway_patterns(n_per_hour: int) -> List[List[int]]:
    """Regular patterns: for step=60/n, test all integer offsets in [0, step)."""
    step = 60 / max(1, n_per_hour)
    cands: List[List[int]] = []
    # Offsets as integers for determinism
    for offset in range(int(step) if step >= 1 else 1):
        pattern = [int((offset + round(i * step)) % 60) for i in range(n_per_hour)]
        cands.append(sorted(pattern))
    # deduplicate
    uniq = []
    seen = set()
    for p in cands:
        key = tuple(p)
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq

def generate_combinatorial_patterns(n_per_hour: int, hmin: float) -> List[List[int]]:
    """All C(60, n) patterns satisfying circular headway >= hmin."""
    from itertools import combinations
    cands = []
    for comb in combinations(range(60), n_per_hour):
        if circular_gaps_ok(list(comb), hmin):
            cands.append(list(comb))
    return cands

def expand_loop_anchor(anchor: Anchor, pattern_minutes: List[int], tmpl: ServiceTemplate) -> None:
    assert tmpl.cycle_time is not None, "Loop requires 'cycle_time' in template."
    start, end = _padded_window()
    dep = generate_hourly_times(pattern_minutes, start, end)         # <-- pad
    arr = [t + int(tmpl.cycle_time) for t in dep]
    anchor.departures = dep
    anchor.arrivals   = arr
    anchor.ensure_weights()


def expand_line_anchors(anchor_A: Anchor, anchor_B: Anchor,
                        pattern_A: List[int], phi: int,
                        tmpl: ServiceTemplate) -> None:
    assert tmpl.tt_AB is not None and tmpl.tt_BA is not None, "Line requires tt_AB & tt_BA"
    start, end = _padded_window()
    dep_A = generate_hourly_times(pattern_A, start, end)             # <-- pad
    pattern_B = [ (m + (phi % 60)) % 60 for m in pattern_A ]
    dep_B = generate_hourly_times(pattern_B, start, end)             # <-- pad
    arr_A = [t + int(tmpl.tt_BA) for t in dep_B]
    arr_B = [t + int(tmpl.tt_AB) for t in dep_A]

    anchor_A.departures, anchor_A.arrivals = dep_A, arr_A
    anchor_B.departures, anchor_B.arrivals = dep_B, arr_B
    anchor_A.ensure_weights()
    anchor_B.ensure_weights()


def expand_route_anchors(anchors: dict[str, Anchor], pattern_minutes: list[int], t: ServiceTemplate) -> None:
    """
    Génère les horaires d'une boucle multi-stops fermée.
    Convention voulue (pas de décalage au départ de S0) :
      dep[S0] = pattern
      arr[Si] = dep[S{i-1}] + leg(S{i-1}→Si)
      dep[Si] = arr[Si] + dwell(Si)  (pour i>=1)
      arr[S0] = dep[S{N-1}] + leg(S{N-1}→S0)   # arrivée à S0 issue de la dernière jambe

    Hypothèses :
      - stops: [S0..S{N-1}] (S0 est le stop "référence")
      - leg_minutes: len==N, temps S_i -> S_{(i+1) mod N} (⚠️ dernière jambe S_{N-1} -> S0 INCLUSE)
      - dwells: dict stop->minutes (0 si absent)
      - pattern_minutes s'applique aux DÉPARTS de S0 (sans dwell(S0) ajouté).
    """
    assert t.stops and t.leg_minutes and len(t.stops) == len(t.leg_minutes), "route: config invalide"

    stops = list(t.stops)
    legs  = [int(x) for x in t.leg_minutes]  # N éléments, inclut S_{N-1} -> S0
    N = len(stops)
    dw = {s: int((t.dwells or {}).get(s, 0)) for s in stops}

    # Offsets par rapport à un départ de S0 (sans dwell appliqué à S0).
    # Définition :
    #   - pour i>=1 :
    #       arr_offset[i] = dep_offset[i-1] + leg(i-1 -> i)
    #       dep_offset[i] = arr_offset[i] + dwell(Si)
    #   - pour S0 :
    #       dep_offset[0] = 0              (clé de voûte : PAS de dwell sur S0 pour les départs)
    #       arr[S0] sera gérée séparément via la dernière jambe (voir plus bas).
    arr_offset = [0] * N
    dep_offset = [0] * N
    dep_offset[0] = 0  # <--- PAS de dwell(S0) ajouté au départ

    for i in range(1, N):
        arr_offset[i] = dep_offset[i-1] + legs[i-1]
        dep_offset[i] = arr_offset[i] + dw.get(stops[i], 0)

    # Durée jusqu'à l'arrivée à S0 (depuis un départ de S_{N-1}) :
    last_leg_to_S0 = legs[N-1]
    # Arrivée à S0 à partir d'un départ de S0 :
    # on part de S0 -> ... -> S{N-1} -> S0(arr)
    # temps cumulé = dep_offset[N-1] + last_leg_to_S0
    cycle_arrival_at_S0 = dep_offset[N-1] + last_leg_to_S0

    # Fenêtre élargie
    total_cycle = cycle_arrival_at_S0 + dw.get(stops[0], 0)  # info (pas utilisé pour offset S0)
    start, end = _padded_window()

    # Départs base de S0 (répétés dans [start, end))
    pm = sorted({int(x) % 60 for x in pattern_minutes})
    dep_S0 = generate_hourly_times(pm, start, end)  # dep[S0] = pattern (sans dwell S0)

    # Construit les séries d'arrivées/départs de chaque stop
    raw_arrs: dict[str, List[int]] = {}
    raw_deps: dict[str, List[int]] = {}

    for i, s in enumerate(stops):
        if i == 0:
            # S0 : départs bruts = pattern ; arrivées via la dernière jambe fermant la boucle
            arr_i = [d + cycle_arrival_at_S0 for d in dep_S0]
            dep_i = dep_S0[:]  # PAS de dwell sur S0 appliqué ici
        else:
            arr_i = [d + arr_offset[i] for d in dep_S0]
            dep_i = [d + dep_offset[i] for d in dep_S0]

        # Rogne à la fenêtre réelle
        raw_arrs[s] = arr_i
        raw_deps[s] = dep_i

    # Affecte aux anchors et garantit les poids
    for s in stops:
        a = anchors.get(s) or Anchor(s)
        a.arrivals   = raw_arrs[s]
        a.departures = raw_deps[s]
        if not a.arrival_weights:
            a.arrival_weights = [1.0] * len(a.arrivals)
        if not a.departure_weights:
            a.departure_weights = [1.0] * len(a.departures)
        anchors[s] = a




# =========================
# 5) Evaluation (generic)
# =========================

@dataclass
class TransferStats:
    avg_up_w: float
    avg_down_w: float
    score: float
    avg_up: float
    avg_down: float
    n_up: int
    n_down: int
    max_up: float
    max_down: float
    std_up: float
    std_down: float
    waits_up: List[float] = field(default_factory=list)
    waits_down: List[float] = field(default_factory=list)
    weights_up: List[float] = field(default_factory=list)
    weights_down: List[float] = field(default_factory=list)


def evaluate_transfer(src: Anchor, dst: Anchor, walk_time: float, min_margin: float,
                      w_up: float, w_down: float) -> TransferStats:
    """Generic evaluation: montée (arrival→departure), descente (arrival←departure)."""
    from statistics import pstdev

    src.ensure_weights()
    dst.ensure_weights()

    # montée: arrivals(src) -> departures(dst)
    waits_up: List[float] = []
    w_up_weights: List[float] = []
    for ta, w in zip(src.arrivals, src.arrival_weights):
        if not (WINDOW_START - EVAL_GRACE_BEFORE <= ta < WINDOW_END + EVAL_GRACE_AFTER):
            continue
        ready = ta + walk_time + min_margin
        d = next_time(dst.departures, ready)
        if d is not None:
            waits_up.append(d - (ta + walk_time))
            w_up_weights.append(w)

    n_up = len(waits_up)
    avg_up_w  = (sum(w*v for w, v in zip(w_up_weights, waits_up)) / sum(w_up_weights)) if w_up_weights else inf
    avg_up    = (sum(waits_up) / n_up) if n_up > 0 else inf
    max_up    = max(waits_up) if waits_up else inf
    std_up    = pstdev(waits_up) if n_up > 1 else 0.0

    # descente: arrivals(dst) -> departures(src)
    waits_down: List[float] = []
    w_down_weights: List[float] = []
    for td, w in zip(src.departures, src.departure_weights):
        if not (WINDOW_START - EVAL_GRACE_BEFORE <= td < WINDOW_END + EVAL_GRACE_AFTER):
            continue
        latest_arrival = td - (walk_time + min_margin)
        a = prev_time(dst.arrivals, latest_arrival)
        if a is not None:
            waits_down.append(td - (a + walk_time))
            w_down_weights.append(w)

    n_down = len(waits_down)
    avg_down_w = (sum(w*v for w, v in zip(w_down_weights, waits_down)) / sum(w_down_weights)) if w_down_weights else inf
    avg_down   = (sum(waits_down) / n_down) if n_down > 0 else inf
    max_down   = max(waits_down) if waits_down else inf
    std_down   = pstdev(waits_down) if n_down > 1 else 0.0

    score = w_up*avg_up_w + w_down*avg_down_w
    return TransferStats(avg_up_w, avg_down_w, score, avg_up, avg_down,
                         n_up, n_down, max_up, max_down, std_up, std_down,
                         waits_up, waits_down,  w_up_weights, w_down_weights)



# =========================
# 6) Offset helpers (used by optimizer)
# =========================

def _shift_times(times: List[int], offset_min: int) -> List[int]:
    """
    Décale une liste d'horaires absolus de offset_min (en minutes).

    Remarque : pas de modulo 24h ici.
    On travaille en minutes absolues et la fenêtre + padding fait le filtrage.
    """
    if offset_min == 0 or not times:
        return times
    return [t + int(offset_min) for t in times]


def shift_anchor_inplace(anchor: Anchor, offset_min: int) -> None:
    """Décale en place tous les horaires d'un Anchor."""
    if offset_min == 0:
        return
    anchor.arrivals   = _shift_times(anchor.arrivals, offset_min)
    anchor.departures = _shift_times(anchor.departures, offset_min)
    anchor.ensure_weights()


def shift_service_inplace(service: Service, offset_min: int) -> None:
    """
    Décale en place tous les anchors d'un Service.

    Usage typique :
      - l'optimizer sauvegarde l'état de base des anchors,
      - applique shift_service_inplace(..., offset) pour un candidat,
      - évalue les transferts, puis restaure l'état de base.
    """
    if offset_min == 0:
        return
    for anc in service.anchors.values():
        shift_anchor_inplace(anc, offset_min)

