# optimizer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Callable
from math import inf
import importlib

# IMPORTANT : on lit la fenêtre dynamiquement via le module
import transit_core as core
from transit_core import (
    Service, ServiceTemplate, Anchor, Transfer, TransferStats,
    generate_equal_headway_patterns, generate_combinatorial_patterns,
    expand_loop_anchor, expand_line_anchors,
    evaluate_transfer, min_to_hhmm, shift_service_inplace,
)

from singletrack_v2 import compute_singletrack_hourly_patterns





import json

# -------------------------
#
# -------------------------

@dataclass
class Candidate:
    pattern: List[int]
    phi: Optional[int] = None  # pour topology="line"
    offset_min: int = 0
    st_combo_index: Optional[int] = None




def _ensure_singletrack_hourly_patterns(t: ServiceTemplate):
    """
    V2 : génère (une seule fois) les patterns horaires à l’ancre A
    via singletrack_v2, puis les met dans t.st_hourly_patterns.

    On NE fait que :
      - charger la ligne V2,
      - calculer les bornes de cycle,
      - générer les patterns (dep,arr) à l’ancre A,
      - les convertir en minutes-dans-l’heure pour cette ancre.

    Pas de patterns B, pas de détails internes pour l’instant.
    """

    # Si déjà fait → on ne refait pas
    if t.st_hourly_patterns:
        return t.st_hourly_patterns

    if not t.st_config_path:
        raise ValueError("singletrack: st_config_path manquant dans ServiceTemplate.")

    # Paramètres de génération V2 (avec défauts raisonnables)
    n_dep = int(getattr(t, "st_n_dep_per_hour", 2) or 2)
    min_hw = float(getattr(t, "st_min_headway", 5.0) or 0.0)

    # Appel au pipeline V2
    out = compute_singletrack_hourly_patterns(
        config_path=t.st_config_path,
        n_dep_per_hour=n_dep,
        min_headway=min_hw,
        extra_terminus_slack=float(getattr(t, "st_extra_terminus_slack", 0.0) or 0.0),
        forbidden_meet_stops=getattr(t, "st_forbidden_meet_stops", None),
        enable_no_overtaking=bool(getattr(t, "st_enable_no_overtaking", True)),
    )

    # compat: ancien return (line, hourly_pats, period) vs nouveau (line, hourly_pats, term_patterns, period)
    if len(out) == 3:
        line, hourly_pats, period = out
        term_patterns = None
    else:
        line, hourly_pats, term_patterns, period = out

    # On stocke les infos dans le template pour plus tard
    t.st_period_min = period
    t.st_line_stops = list(line.stops)
    t.st_hourly_patterns = hourly_pats
    t.st_term_patterns = term_patterns
    t.st_line_config = line

    return hourly_pats







def _candidates_singletrack(svc: Service) -> List[Candidate]:
    t = svc.template
    if t is None:
        return []

    pats = _ensure_singletrack_hourly_patterns(t)
    if not pats:
        return []

    # Paramètres d'offset global, comme pour "scheduled+offset"
    offset_range = int(getattr(t, "offset_range_min", 0) or 0)
    center = int(getattr(t, "base_offset_min", 0) or 0)
    step = int(getattr(t, "offset_step_min", 1) or 1)
    if step <= 0:
        step = 1

    if offset_range <= 0:
        offsets = [center]
    else:
        offsets = list(range(center - offset_range, center + offset_range + 1, step))

    # Pour le champ pattern (juste pour debug), on se base sur le 1er arrêt de la ligne
    ref_stop = None
    if t.st_line_stops:
        ref_stop = t.st_line_stops[0]
    elif svc.anchors:
        ref_stop = next(iter(svc.anchors.keys()), None)

    cands: List[Candidate] = []

    for idx, combo_pat in enumerate(pats):
        debug_pat: List[int] = []
        if ref_stop and ref_stop in combo_pat:
            debug_pat = combo_pat[ref_stop]["dep"] or combo_pat[ref_stop]["arr"] or []

        for off in offsets:
            cands.append(
                Candidate(
                    pattern=list(debug_pat),
                    phi=None,
                    offset_min=off,
                    st_combo_index=idx,
                )
            )

    return cands


def _expand_hourly_pattern(
    mins_in_hour: List[int],
    offset_min: int,
    period: int,
    start_min: int,
    end_min: int,
) -> List[int]:
    """
    À partir d'une liste de minutes dans l'heure (0..period-1),
    d'un offset global et d'une fenêtre [start_min, end_min),
    génère les temps absolus dans cette fenêtre.
    """
    times: List[int] = []
    for m in mins_in_hour:
        t = int(m) + int(offset_min)
        # On saute jusqu'à atteindre la fenêtre
        while t < start_min:
            t += period
        while t < end_min:
            times.append(t)
            t += period
    return sorted(times)




# -------------------------
# Types
# -------------------------


@dataclass
class EvalRecord:
    service_name: str
    candidate: Candidate
    score: float
    details: Dict[str, Any]  # {"transfers": {...}}

# Plugin API : list[dict] -> dict
TieBreakerFn = Callable[[List[dict], Dict[str, Any]], dict]

# -------------------------
# Tie-breakers (fallbacks)
# -------------------------
def _tb_identity(cands: List[dict], params: Dict[str, Any]) -> dict:
    return sorted(cands, key=lambda d: (d.get("pattern", []), d.get("phi", -1)))[0]

def _tb_min_max(cands: List[dict], params: Dict[str, Any]) -> dict:
    def mx(d): return max(d.get("max_up", inf), d.get("max_down", inf))
    return min(cands, key=lambda d: (mx(d), d.get("avg_down_w", inf), d.get("avg_up_w", inf), d.get("pattern", [])))

def _tb_penalty(cands: List[dict], params: Dict[str, Any]) -> dict:
    T = params.get("max_wait"); lam = float(params.get("lambda", 0.0))
    if T is None or lam <= 0.0: return _tb_identity(cands, params)
    def key(d):
        over = max(d.get("max_up", inf), d.get("max_down", inf)) - T
        pen = lam * max(0.0, over)
        return (pen, d.get("avg_down_w", inf), d.get("avg_up_w", inf), d.get("pattern", []))
    return min(cands, key=key)

DEFAULT_PLUGIN_TB: Dict[str, TieBreakerFn] = {
    "none": _tb_identity,
    "min_max": _tb_min_max,
    "penalty": _tb_penalty,
}

def load_tie_breakers(module_name: Optional[str]) -> Dict[str, TieBreakerFn]:
    """Charge un module externe exposant TIE_BREAKERS (compatible avec ton tie_breakers.py)."""
    if not module_name: return {}
    try:
        mod = importlib.import_module(module_name)
        reg = getattr(mod, "TIE_BREAKERS", {})
        return {k: v for k, v in reg.items() if callable(v)} if isinstance(reg, dict) else {}
    except Exception as e:
        print(f"[tie-breakers] ⚠️ '{module_name}': {e}")
        return {}

# -------------------------
# Candidats & réalisation
# -------------------------
def candidates_for_service(svc: Service) -> List[Candidate]:
    """
    Génère la liste des candidats (patterns de minutes dans l'heure, et éventuellement phi)
    pour un service à optimiser, selon sa topologie :
      - loop  : pattern(s) sur un stop de référence
      - line  : pattern(s) + phase phi (0..59 ou fixé)
      - route : pattern(s) sur le 1er stop de 'stops' (multi-stops), propagés par leg_minutes + dwells
    """
    t: Optional[ServiceTemplate] = svc.template
    if t is None:
        return []

    topo = t.topology


    # --- Cas singletrack : patterns pré-computés en minutes-dans-l'heure ---
    if topo == "singletrack":
        return _candidates_singletrack(svc)


    # --- Cas "scheduled" : offset only (pas de génération de pattern) ---
    offset_range = int(getattr(t, "offset_range_min", 0) or 0)
    if topo == "scheduled":
        if offset_range <= 0:
            # service prévu pour rester tel quel (aucune optimisation)
            return []
        center = int(getattr(t, "base_offset_min", 0) or 0)
        step = int(getattr(t, "offset_step_min", 1) or 1)
        if step <= 0:
            step = 1
        offsets = list(range(center - offset_range, center + offset_range + 1, step))
        if not offsets:
            offsets = [center]
        # pattern=[] car on ne génère rien de nouveau, on ne fait que décaler
        return [Candidate(pattern=[], phi=None, offset_min=o) for o in offsets]

    # --- Headway minimal par topologie (pour inférence n_per_hour & combinatoire) ---
    def _hmin_loop(tt: ServiceTemplate) -> float:
        # headway opérationnel minimal pour un loop classique : cycle + turnaround
        # (si tu as un helper headway_min_loop, tu peux l'appeler ici)
        cycle = float(getattr(tt, "cycle_time", 0) or 0)
        turnaround = float(getattr(tt, "turnaround", 0) or 0)
        return max(1.0, cycle + turnaround)

    def _hmin_line(tt: ServiceTemplate) -> float:
        # headway opérationnel minimal pour une ligne A<->B : tt_AB + turnaround_B et tt_BA + turnaround_A
        # (si tu as un helper headway_min_line, tu peux l'appeler ici)
        tt_ab = float(getattr(tt, "tt_AB", 0) or 0) + float(getattr(tt, "turnaround_B", 0) or 0)
        tt_ba = float(getattr(tt, "tt_BA", 0) or 0) + float(getattr(tt, "turnaround_A", 0) or 0)
        return max(1.0, tt_ab, tt_ba)

    def _hmin_route(tt: ServiceTemplate) -> float:
        # Pour une boucle multi-stops : on prend le cycle total = somme(legs) + somme(dwells)
        legs = [int(x) for x in (tt.leg_minutes or [])]
        dw = sum(int(v) for v in (tt.dwells or {}).values())
        cycle_total = sum(legs) + dw
        return max(1.0, float(cycle_total))

    if topo == "loop":
        hmin = _hmin_loop(t)
    elif topo == "line":
        hmin = _hmin_line(t)
    elif topo == "route":
        hmin = _hmin_route(t)
    else:
        raise ValueError(f"Unknown topology: {topo}")

    # --- Inférence de n_per_hour si non fourni ---
    nph = int(t.n_per_hour or 0)
    if nph <= 0:
        # même logique que le code existant : on part d'un 1.05 * hmin
        nph = max(1, int(60 // max(1.0, 1.05 * hmin)))

    # --- Génération des patterns (minutes dans l'heure) ---
    # equal_headway -> patterns équidistants ; sinon -> combinatoire avec contrainte de gap
    if t.equal_headway:
        patterns = generate_equal_headway_patterns(nph)
    else:
        # Pour "route", on ne veut pas surcontraindre : gap minimal = 1 minute est un choix raisonnable,
        # mais si tu préfères la robustesse "opérationnelle", tu peux mettre hmin.
        min_gap = hmin if topo in ("loop", "line") else 1.0
        patterns = generate_combinatorial_patterns(nph, min_gap)

    if not patterns:
        raise ValueError(f"Aucun pattern faisable pour '{svc.name}' (n_per_hour={nph}, topo={topo}).")

    # --- Assemblage des candidats ---
    if topo == "loop":
        # Pas de phi ; pattern s'applique sur le stop de référence
        return [Candidate(p) for p in patterns]

    if topo == "route":
        # Multi-stops : pas de phi non plus ; on applique le pattern au stop S0, le reste est déduit par legs+dwells
        # (les ancres de chaque stop sont réalisées plus tard par expand_route_anchors)
        # Sécurité : s'assurer que stops/legs sont valides a déjà été fait côté build_services.
        return [Candidate(p) for p in patterns]

    # topo == "line"
    if not (t.anchors and len(t.anchors) == 2):
        raise ValueError(f"'{svc.name}' (line) requiert 2 anchors dans template.anchors")


    # phi : soit un entier fixé, soit on balaie 0..59
    phi_list = [t.phi_minutes % 60] if isinstance(t.phi_minutes, int) else list(range(60))
    return [Candidate(p, phi) for p in patterns for phi in phi_list]


def realize_service_schedule(svc: Service, cand: Candidate, inplace: bool = True) -> Dict[str, Anchor]:
    t = svc.template
    if t is None or t.topology == "scheduled":
        for a in svc.anchors.values(): a.ensure_weights()
        return svc.anchors
    if t.topology == "loop":
        name = next(iter(svc.anchors.keys()), "default")
        a = svc.anchors.get(name) or Anchor(name)
        expand_loop_anchor(a, cand.pattern, t)
        if inplace: svc.anchors[name] = a
        return {name: a}
    if t.topology == "line":
        A, B = t.anchors
        aA = svc.anchors.get(A) or Anchor(A)
        aB = svc.anchors.get(B) or Anchor(B)
        expand_line_anchors(aA, aB, cand.pattern, int(cand.phi or 0) % 60, t)
        if inplace: svc.anchors[A], svc.anchors[B] = aA, aB
        return {A: aA, B: aB}
    if t.topology == "route":  # <-- NOUVEAU
        # génère toutes les ancres de la boucle multi-stops
        from transit_core import expand_route_anchors
        expand_route_anchors(svc.anchors, cand.pattern, t)
        return svc.anchors

    if t.topology == "singletrack":
        # On récupère les patterns minutes/h pour toutes les options
        pats = _ensure_singletrack_hourly_patterns(t)
        if not pats:
            return svc.anchors

        period = getattr(t, "st_period_min", 60)
        combo_idx = cand.st_combo_index or 0
        if combo_idx < 0 or combo_idx >= len(pats):
            combo_idx = 0
        combo_pat = pats[combo_idx]

        # Fenêtre absolue = fenêtre + padding (défini dans transit_core)
        start_min, end_min = core._padded_window()

        new_anchors: Dict[str, Anchor] = {}

        for stop_name, pat in combo_pat.items():
            arr_mins = pat.get("arr", [])
            dep_mins = pat.get("dep", [])

            arr_abs = _expand_hourly_pattern(arr_mins, cand.offset_min, period, start_min, end_min)
            dep_abs = _expand_hourly_pattern(dep_mins, cand.offset_min, period, start_min, end_min)

            anc = svc.anchors.get(stop_name) or Anchor(stop_name)
            anc.arrivals = arr_abs
            anc.departures = dep_abs
            anc.arrival_weights = [1.0] * len(arr_abs)
            anc.departure_weights = [1.0] * len(dep_abs)
            anc.ensure_weights()

            new_anchors[stop_name] = anc

        if inplace:
            svc.anchors.update(new_anchors)
            return svc.anchors
        return new_anchors



    raise ValueError(f"Unknown topology: {t.topology}")



# -------------------------
# Évaluation (stricte) + agrégation plugin
# -------------------------
def _anchor_has_events(a: Anchor) -> bool: return bool(a.arrivals or a.departures)



def evaluate_transfer_guarded(src: Anchor, dst: Anchor, tr: Transfer) -> TransferStats:
    """
    Variante 'tolérante' :
    - Si un sens est demandé (w_up/w_down > 0) MAIS les listes nécessaires sont vides
      (ex: pas d'arrivées source OU pas de départs destination), on NE lève pas d'erreur.
      On retourne simplement des stats avec n=0 pour ce sens.
    - On ne lève une ValueError que si les listes existent et qu'aucune correspondance
      n'est trouvée malgré la présence d'événements des deux côtés.
    """
    st = evaluate_transfer(src, dst, tr.walk_time, tr.min_margin, tr.w_up, tr.w_down)

    win = f"[{min_to_hhmm(core.WINDOW_START)},{min_to_hhmm(core.WINDOW_END)})"
    problems: List[str] = []

    # UP demandé ?
    if tr.w_up > 0:
        need_src_arr = bool(src.arrivals)
        need_dst_dep = bool(dst.departures)
        # si les deux côtés existent mais n_up==0 → vrai problème
        if need_src_arr and need_dst_dep and st.n_up == 0:
            problems.append(
                f"UP: {src.name}(arr)→{dst.name}(dep) introuvable sur {win} "
                f"| arr(src): {len(src.arrivals)} | dep(dst): {len(dst.departures)} "
                f"| walk+margin={tr.walk_time+tr.min_margin:.1f}"
            )

    # DOWN demandé ?
    if tr.w_down > 0:
        need_dst_arr = bool(dst.arrivals)
        need_src_dep = bool(src.departures)
        if need_dst_arr and need_src_dep and st.n_down == 0:
            problems.append(
                f"DOWN: {dst.name}(arr)→{src.name}(dep) introuvable sur {win} "
                f"| arr(dst): {len(dst.arrivals)} | dep(src): {len(src.departures)} "
                f"| walk+margin={tr.walk_time+tr.min_margin:.1f}"
            )

    if problems:
        raise ValueError("Transfert impossible:\n  - " + "\n  - ".join(problems))

    return st


def eval_service_objective(svc_name: str, services: Dict[str, Service], transfers: List[Transfer]) -> Tuple[float, Dict[str, Any]]:
    """Somme les scores des transferts impliquant svc_name. Ignore un transfert si la contrepartie est cadencée et encore vide (optimisation séquentielle)."""
    total = 0.0
    det: Dict[str, Any] = {"transfers": {}}
    for tr in transfers:
        if not (tr.from_service == svc_name or tr.to_service == svc_name): continue
        sf, st_ = services[tr.from_service], services[tr.to_service]
        af = sf.anchors.get(tr.from_anchor) or sf.anchors.get("default")
        at = st_.anchors.get(tr.to_anchor) or st_.anchors.get("default")
        if af is None or at is None: continue
        counterpart = st_ if tr.from_service == svc_name else sf
        counterpart_anchor = at if tr.from_service == svc_name else af
        if counterpart.template and counterpart.template.topology != "scheduled" and not _anchor_has_events(counterpart_anchor):
            continue  # on l’utilisera quand ce service sera réalisé
        stats = evaluate_transfer_guarded(af, at, tr)
        total += float(stats.score)
        key = f"{tr.from_service}:{tr.from_anchor}->{tr.to_service}:{tr.to_anchor}"
        det["transfers"][key] = {
            "avg_up_w": stats.avg_up_w, "avg_down_w": stats.avg_down_w,
            "avg_up": stats.avg_up, "avg_down": stats.avg_down,
            "n_up": stats.n_up, "n_down": stats.n_down,
            "max_up": stats.max_up, "max_down": stats.max_down,
            "std_up": stats.std_up, "std_down": stats.std_down,
            "waits_up": stats.waits_up, "waits_down": stats.waits_down,
            "weights_up": stats.weights_up,
            "weights_down": stats.weights_down,

        }
    return total, det

def _aggregate_for_plugin(rec: EvalRecord) -> dict:
    """Aplatis les métriques d’un EvalRecord au format attendu par le plugin."""
    trs = rec.details.get("transfers", {})
    if not trs:
        return {
            "pattern": sorted(rec.candidate.pattern),
            "phi": rec.candidate.phi,
            "offset": rec.candidate.offset_min,
            "score": rec.score,
            "avg_up_w": inf, "avg_down_w": inf,
            "avg_up": inf, "avg_down": inf,
            "n_up": 0, "n_down": 0,
            "max_up": inf, "max_down": inf,
            "std_up": 0.0, "std_down": 0.0,
            "waits_up": [], "waits_down": [],
        }

    def mean(xs): return sum(xs)/len(xs) if xs else inf
    avg_up_w   = mean([d["avg_up_w"]   for d in trs.values()])
    avg_down_w = mean([d["avg_down_w"] for d in trs.values()])
    avg_up     = mean([d["avg_up"]     for d in trs.values()])
    avg_down   = mean([d["avg_down"]   for d in trs.values()])
    n_up       = sum(int(d["n_up"])   for d in trs.values())
    n_down     = sum(int(d["n_down"]) for d in trs.values())
    max_up     = max(float(d["max_up"])   for d in trs.values())
    max_down   = max(float(d["max_down"]) for d in trs.values())
    std_up     = mean([d["std_up"]   for d in trs.values()])
    std_down   = mean([d["std_down"] for d in trs.values()])
    wu, wd = [], []
    for d in trs.values():
        wu.extend(d.get("waits_up", []) or [])
        wd.extend(d.get("waits_down", []) or [])
    return {
        "pattern": sorted(rec.candidate.pattern),
        "phi": rec.candidate.phi,
        "offset": rec.candidate.offset_min,
        "score": rec.score,
        "avg_up_w": avg_up_w, "avg_down_w": avg_down_w,
        "avg_up": avg_up, "avg_down": avg_down,
        "n_up": n_up, "n_down": n_down,
        "max_up": max_up, "max_down": max_down,
        "std_up": std_up, "std_down": std_down,
        "waits_up": wu, "waits_down": wd,
    }


# -------------------------
# Recherche du meilleur (ex æquo + plugin)
# -------------------------
def find_best_for_service(
    svc: Service,
    services: Dict[str, Service],
    transfers: List[Transfer],
    external_tie_breakers: Optional[Dict[str, TieBreakerFn]] = None,
    tie_tol: float = 1e-9,
) -> Tuple[Optional[EvalRecord], List[EvalRecord], List[dict], List[EvalRecord]]:
    """
    Identique à avant, MAIS avec une étape optionnelle de validation "exacte" pour singletrack :
      - on évalue tous les candidats -> scores
      - on trie par score
      - si topo == singletrack : on prend le meilleur, on tente une validation exacte.
          * si OK -> on le garde
          * sinon -> on essaie le runner-up, etc.
      - si aucune validation exacte disponible (pas implémentée), on retombe sur l'ancien comportement.

    Retourne:
      chosen, ties, ties_dicts, records
    """
    t = svc.template
    if t is None:
        return None, [], [], []

    # Mode offset = topology="scheduled" + offset_range_min>0
    offset_range = int(getattr(t, "offset_range_min", 0) or 0)
    is_offset_mode = (t.topology == "scheduled" and offset_range > 0)

    # Si scheduled mais sans offset -> rien à optimiser
    if t.topology == "scheduled" and not is_offset_mode:
        return None, [], [], []

    cands = candidates_for_service(svc)
    if not cands:
        return None, [], [], []

    # Snapshot de l'état d'origine des anchors du service
    orig = {
        n: Anchor(a.name, a.arrivals[:], a.departures[:],
                  a.arrival_weights[:], a.departure_weights[:])
        for n, a in svc.anchors.items()
    }

    records: List[EvalRecord] = []
    best: Optional[float] = None

    try:
        for cand in cands:
            # On repart toujours de l'état d'origine pour chaque candidat
            svc.anchors = {
                n: Anchor(a.name, a.arrivals[:], a.departures[:],
                          a.arrival_weights[:], a.departure_weights[:])
                for n, a in orig.items()
            }

            if is_offset_mode:
                # Horaire de base déjà en place : on applique seulement le décalage global
                shift_service_inplace(svc, cand.offset_min)
            else:
                # loop / line / route / singletrack : génération classique à partir du pattern/phi
                realize_service_schedule(svc, cand, inplace=True)

            s, det = eval_service_objective(svc.name, services, transfers)
            rec = EvalRecord(svc.name, cand, float(s), det)
            records.append(rec)

            if best is None or rec.score < best - tie_tol:
                best = rec.score
    finally:
        # On restaure l'état d'origine du service (anchors de base)
        svc.anchors = orig

    if best is None or not records:
        return None, [], [], []

    # -------------------------
    # 1) Prépare tie-breaker (comme avant)
    # -------------------------
    tb_name, tb_params = None, {}
    for tr in transfers:
        if tr.from_service == svc.name or tr.to_service == svc.name:
            if tr.tie_breaker:
                tb_name = tr.tie_breaker
                tb_params = tr.tie_params or {}
    registry = {**DEFAULT_PLUGIN_TB, **(external_tie_breakers or {})}
    tb_fn = registry.get(tb_name or "none", _tb_identity)

    # Ex æquo sur le score (à tolérance tie_tol près)
    ties = [r for r in records if (r.score == best) or (abs(r.score - best) <= tie_tol)]
    ties_dicts = [_aggregate_for_plugin(r) for r in ties] if ties else []

    # -------------------------
    # 2) Choix gagnant "par score + tie-breaker" (ancien comportement)
    # -------------------------
    if not ties:
        winner = min(records, key=lambda r: r.score)
        chosen = winner
        chosen_dict = _aggregate_for_plugin(winner)
    else:
        chosen_dict = tb_fn(ties_dicts, tb_params)

        def same(rec: EvalRecord, d: dict) -> bool:
            return (
                sorted(rec.candidate.pattern) == list(d.get("pattern", []))
                and rec.candidate.phi == d.get("phi", rec.candidate.phi)
                and rec.candidate.offset_min == d.get("offset", rec.candidate.offset_min)
            )

        chosen = next((r for r in ties if same(r, chosen_dict)), ties[0])

    # -------------------------
    # 3) NEW: validation itérative pour singletrack (best score puis runner-ups)
    # -------------------------
    def _validate_singletrack_exact(rec: EvalRecord) -> Tuple[bool, Dict[str, Any]]:
        """
        Appelle le wrapper singletrack_v2.validate_singletrack_candidate(term_pattern=...)
        en lui passant le TerminusPattern correspondant au st_combo_index.
        """
        if not svc.template or svc.template.topology != "singletrack":
            return True, {"status": "skip", "reason": "not_singletrack"}

        import singletrack_v2 as stv2
        fn = getattr(stv2, "validate_singletrack_candidate", None)
        if not callable(fn):
            # Pas de validateur -> on ne bloque pas (fallback heuristique)
            return True, {"status": "skip", "reason": "no_exact_validator_implemented"}

        tmpl = svc.template

        term_list = getattr(tmpl, "st_term_patterns", None) or []
        if not term_list:
            return False, {"status": "fail", "reason": "st_term_patterns missing on template"}

        idx = int(rec.candidate.st_combo_index or 0)
        if idx < 0 or idx >= len(term_list):
            idx = 0

        term_pattern = term_list[idx]
        if term_pattern is None:
            return False, {"status": "fail", "reason": "term_pattern is None in st_term_patterns"}

        ok, details = fn(
            config_path=str(tmpl.st_config_path),
            term_pattern=term_pattern,
            forbidden_meet_stops=getattr(tmpl, "st_forbidden_meet_stops", None),
            enable_no_overtaking=bool(getattr(tmpl, "st_enable_no_overtaking", True)),
            require_two_meets=True,
        )
        return bool(ok), (details if isinstance(details, dict) else {"status": "unknown", "details": str(details)})

    if t.topology == "singletrack":
        # On teste d'abord le meilleur score, puis runner-ups (et pas "chosen" en premier)
        records_sorted = sorted(
            records,
            key=lambda r: (
                round(float(r.score), 12),
                tuple(sorted(r.candidate.pattern)),
                int(r.candidate.offset_min or 0),
                int(r.candidate.st_combo_index or 0),
            )
        )

        selected: Optional[EvalRecord] = None
        last_details: Dict[str, Any] = {"status": "fail", "reason": "no_try"}

        for k, r in enumerate(records_sorted, start=1):
            ok, details = _validate_singletrack_exact(r)

            print(
                f"[{svc.name}][singletrack-validate] try #{k}: "
                f"score={r.score:.3f} pattern={sorted(r.candidate.pattern)} "
                f"offset={int(r.candidate.offset_min or 0)} st_idx={r.candidate.st_combo_index} "
                f"-> {'OK' if ok else 'FAIL'} ({details})"
            )

            last_details = details

            # "skip" (pas de validateur) => on garde direct le best score
            if details.get("status") == "skip":
                selected = records_sorted[0]
                break

            if ok:
                # <-- ICI: on "fixe" tous les anchors grâce à la reconstruction exacte
                try:
                    fill_singletrack_anchors_from_exact_details(svc, svc.template, details)
                except Exception as e:
                    print(f"[{svc.name}][singletrack-finalize] ⚠ failed to fill anchors: {e}")

                selected = r
                break

        if selected is None:
            print(
                f"[{svc.name}][singletrack-validate] ⚠ aucun candidat validé exactement "
                f"(dernier details={last_details}). On conserve le meilleur score heuristique."
            )
            selected = records_sorted[0]

        chosen = selected

    # Debug / résumé combinatoire
    n_cands = len(cands)
    n_ties = len(ties)
    print(f"\n[{svc.name}] → {n_cands} combinaisons testées | {n_ties} ex æquo (best score={best:.3f})")

    return chosen, ties, ties_dicts, records





# -------------------------
# Optimisation multi-services
# -------------------------
def validate_services_and_transfers(services: Dict[str, Service], transfers: List[Transfer]) -> None:
    for s in services.values():
        if s.template and s.template.topology == "line":
            if not s.template.anchors or len(s.template.anchors) != 2:
                raise ValueError(f"Service '{s.name}' (line) doit avoir 2 anchors.")
    for tr in transfers:
        if tr.from_service not in services: raise ValueError(f"Transfert: service inconnu {tr.from_service}")
        if tr.to_service not in services: raise ValueError(f"Transfert: service inconnu {tr.to_service}")

def _print_ties(service_name: str, ties_dicts: List[dict], chosen_dict: dict, tb_name: str, tb_params: Dict[str, Any]) -> None:
    if not ties_dicts:
        return

    has_phi = any(d.get("phi") is not None for d in ties_dicts)
    has_offset = any(d.get("offset", 0) != 0 for d in ties_dicts)

    cols = ["pattern"]
    if has_phi:
        cols.append("phi")
    if has_offset:
        cols.append("offset")
    cols += ["score", "avg_up_w", "avg_down_w", "max_up", "max_down"]

    print(f"\n=== {service_name} — Ex æquo ({len(ties_dicts)}) | tie_breaker={tb_name or 'none'} params={tb_params or {}}")
    header = "  ".join(f"{c:>12}" for c in cols)
    print(header)
    print("-" * len(header))

    def fmt_pat(d): return " ".join(map(str, d.get("pattern", [])))

    def row(d):
        r = [f"[{fmt_pat(d)}]"]
        if has_phi:
            r.append(str(d.get("phi", "-")))
        if has_offset:
            r.append(str(d.get("offset", 0)))
        r += [
            f"{d['score']:.3f}",
            f"{d['avg_up_w']:.2f}", f"{d['avg_down_w']:.2f}",
            f"{d['max_up']:.2f}", f"{d['max_down']:.2f}",
        ]
        return "  ".join(f"{x:>12}" for x in r)

    # trier d’abord par score, puis pattern, puis phi
    ties_sorted = sorted(
        ties_dicts,
        key=lambda d: (round(d.get("score", float("inf")), 9), tuple(d.get("pattern", [])), d.get("phi", -1))
    )

    for d in ties_sorted[:12]:
        print(row(d))

    suffix = ("" if chosen_dict.get("phi") is None else f", phi={chosen_dict['phi']}")
    print("→ choisi :", f"[{fmt_pat(chosen_dict)}]{suffix}")


def optimize_services(
    services: Dict[str, Service],
    transfers: List[Transfer],
    targets: List[str],
    external_tie_breakers: Optional[Dict[str, TieBreakerFn]] = None,
    csv_path: Optional[str] = None,
    include_ties: bool = True
) -> List[EvalRecord]:
    """
    Optimise les services un par un, en tenant compte des transferts.
    - 'results' : meilleur candidat par service (comme avant)
    - 'all_candidates_global' : tous les candidats testés pour tous les services
      (utile pour export CSV complet).
    """
    validate_services_and_transfers(services, transfers)

    results: List[EvalRecord] = []
    all_candidates_global: List[EvalRecord] = []

    for name in targets:
        svc = services[name]

        # find_best_for_service doit maintenant renvoyer aussi 'records'
        chosen, ties, ties_dicts, records = find_best_for_service(
            svc, services, transfers, external_tie_breakers
        )
        if chosen is None:
            continue

        # On accumule tous les candidats testés pour ce service
        all_candidates_global.extend(records)

        # applique la solution gagnante
        t = svc.template
        is_offset_mode = (
            t is not None
            and t.topology == "scheduled"
            and int(getattr(t, "offset_range_min", 0) or 0) > 0
        )

        if is_offset_mode:
            # on part des horaires de base (tels que lus du JSON / pattern_streams)
            # et on applique simplement le décalage choisi
            shift_service_inplace(svc, chosen.candidate.offset_min)
        else:
            # loop/line/route/singletrack : génération classique
            realize_service_schedule(svc, chosen.candidate, inplace=True)

        results.append(chosen)

        if include_ties and len(ties) > 1:
            # retrouver tie-breaker utilisé
            tb_name, tb_params = None, {}
            for tr in transfers:
                if tr.from_service == svc.name or tr.to_service == svc.name:
                    if tr.tie_breaker:
                        tb_name = tr.tie_breaker
                        tb_params = tr.tie_params or {}

            # dict du choix
            chosen_dict = next(
                (
                    d for d in ties_dicts
                    if sorted(d.get("pattern", [])) == sorted(chosen.candidate.pattern)
                    and d.get("phi", chosen.candidate.phi) == chosen.candidate.phi
                    and d.get("offset", chosen.candidate.offset_min) == chosen.candidate.offset_min
                ),
                ties_dicts[0],
            )

            _print_ties(svc.name, ties_dicts, chosen_dict, tb_name or "none", tb_params)

    if csv_path:
        # CSV "classique" : seulement les gagnants (comportement historique)
        _export_csv(csv_path, results)
        # Nouveau : CSV complet avec tous les candidats testés
        _export_all_candidates_csv(
            csv_path.replace(".csv", "_all.csv"),
            all_candidates_global
        )

    return results


def _export_csv(path: str, chosen: List[EvalRecord]) -> None:
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["service","pattern","phi","score"])
        for r in chosen:
            pat = " ".join(map(str, sorted(r.candidate.pattern)))
            phi = "" if r.candidate.phi is None else r.candidate.phi
            w.writerow([r.service_name, pat, phi, round(r.score, 6)])

def _export_all_candidates_csv(path, records):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["service","pattern","phi","offset","score"])
        for r in records:
            pat = " ".join(map(str, sorted(r.candidate.pattern)))
            phi = "" if r.candidate.phi is None else r.candidate.phi
            off = r.candidate.offset_min
            w.writerow([r.service_name, pat, phi, off, round(r.score, 6)])

#################################################
########### Fixe anchor singletrack #############
#################################################


from typing import Dict, Any, List, Set
import transit_core
from transit_core import Anchor, generate_hourly_times

def _padded_window_from_globals() -> tuple[int, int]:
    return (
        transit_core.WINDOW_START - transit_core.PADDING_MIN,
        transit_core.WINDOW_END   + transit_core.PADDING_MIN,
    )

def fill_singletrack_anchors_from_exact_details(
    svc,          # Service
    tmpl,         # ServiceTemplate
    details: Dict[str, Any],
) -> None:
    """
    Après validation exacte OK, construit arrivals/departures pour CHAQUE stop
    et les écrit dans svc.anchors[stop].
    """
    trains = details.get("trains") or []
    if not trains:
        return  # rien à faire

    period = int(details.get("period", 60))

    # On recharge la config ligne pour obtenir l'ordre des stops
    import singletrack_v2 as stv2
    line = stv2.load_line_config(str(tmpl.st_config_path))
    stops = list(line.stops)

    # Collecte des minutes dans l'heure par stop (arr/dep), en fusionnant UP+DOWN
    arr_minutes: Dict[str, Set[int]] = {s: set() for s in stops}
    dep_minutes: Dict[str, Set[int]] = {s: set() for s in stops}

    A = details.get("A")
    B = details.get("B")

    for tr in trains:
        up_arr = tr.get("up_arr") or {}
        up_dep = tr.get("up_dep") or {}
        dn_arr = tr.get("dn_arr") or {}
        dn_dep = tr.get("dn_dep") or {}

        for s in stops:
            # ----- UP -----
            if s == A:
                # à A: seulement DEP UP
                if s in up_dep: dep_minutes[s].add(int(up_dep[s]) % period)
            elif s == B:
                # à B: ARR + DEP UP (turnaround)
                if s in up_arr: arr_minutes[s].add(int(up_arr[s]) % period)
                if s in up_dep: dep_minutes[s].add(int(up_dep[s]) % period)
            else:
                # intermédiaire: ARR + DEP UP
                if s in up_arr: arr_minutes[s].add(int(up_arr[s]) % period)
                if s in up_dep: dep_minutes[s].add(int(up_dep[s]) % period)

            # ----- DOWN -----
            if s == B:
                # à B: seulement DEP DOWN (départ retour)
                if s in dn_dep: dep_minutes[s].add(int(dn_dep[s]) % period)
            elif s == A:
                # à A: seulement ARR DOWN (arrivée finale)
                if s in dn_arr: arr_minutes[s].add(int(dn_arr[s]) % period)
            else:
                # intermédiaire: ARR + DEP DOWN
                if s in dn_arr: arr_minutes[s].add(int(dn_arr[s]) % period)
                if s in dn_dep: dep_minutes[s].add(int(dn_dep[s]) % period)

    start, end = _padded_window_from_globals()

    # IMPORTANT: on remplace les anchors par ceux du singletrack
    svc.anchors = {}

    for s in stops:
        anc = Anchor(name=s)
        am = sorted(arr_minutes[s])
        dm = sorted(dep_minutes[s])

        anc.arrivals   = generate_hourly_times(am, start, end) if am else []
        anc.departures = generate_hourly_times(dm, start, end) if dm else []

        anc.ensure_weights()
        svc.anchors[s] = anc

    # utile pour ton print_detailed_report
    tmpl.st_line_stops = stops
