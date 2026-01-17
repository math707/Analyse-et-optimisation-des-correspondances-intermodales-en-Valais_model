# engine_runner.py
import json
from typing import Dict, Any, List, Optional

# noyau + optimiseur (déjà dans ton projet)
from transit_core import (
    set_window, set_padding, compute_recommended_padding,
    Service, ServiceTemplate, Anchor, expand_pattern_streams, min_to_hhmm
)
from optimizer import (
    validate_services_and_transfers, optimize_services, load_tie_breakers,
    evaluate_transfer_guarded,  # <-- ajouter ceci
)

# --------- helpers compacts ---------
def _hhmm_to_min(s: str) -> int:
    h, m = s.strip().split(":"); return int(h)*60 + int(m)

def _parse_hhmm_list(xs: List[str]) -> List[int]:
    return sorted(_hhmm_to_min(x) for x in xs)

def _norm_minutes(v) -> List[int]:
    return sorted({int(x) % 60 for x in v})

def _build_services(cfg: Dict[str, Any]) -> Dict[str, Service]:
    services: Dict[str, Service] = {}
    for m in cfg.get("modes", []):
        name, cat = m["name"], m.get("category", "generic")
        schedule, streams, template = m.get("schedule"), m.get("pattern_streams"), m.get("template")

        svc = Service(name=name, category=cat, anchors={}, template=None)


        # 1) schedule absolu (arrivées/départs HH:MM)
        if schedule is not None:
            anc = Anchor("default")
            arr = _parse_hhmm_list(schedule.get("arrivals", []))
            dep = _parse_hhmm_list(schedule.get("departures", []))
            anc.arrivals, anc.departures = arr, dep
            anc.arrival_weights = schedule.get("arrival_weights") or [1.0]*len(arr)
            anc.departure_weights = schedule.get("departure_weights") or [1.0]*len(dep)
            anc.ensure_weights()
            svc.anchors["default"] = anc

        # 2) streams cadencés (on expan- dra après padding)
        if streams:
            svc.anchors["default"] = svc.anchors.get("default") or Anchor("default")
            svc.pattern_streams = []
            for s in streams:
                s2 = dict(s)
                if "depart_minutes" in s2:
                    s2["depart_minutes"] = _norm_minutes(s2["depart_minutes"])
                svc.pattern_streams.append(s2)

        # 3) service à optimiser (loop / line / route / scheduled+offset)
        if template is not None:
            topo = template.get("topology") or ("line" if "anchors" in template else "loop")

            if topo == "scheduled":
                # Cas spécial : service déjà horaire (schedule ou pattern_streams)
                # mais qu'on veut décaler globalement via l'optimizer.
                svc.template = ServiceTemplate(
                    topology="scheduled",
                    base_offset_min=int(template.get("base_offset_min", 0)),
                    offset_range_min=int(template.get("offset_range_min", 0)),
                    offset_step_min=int(template.get("offset_step_min", 1)),
                )
                # anchors et horaires de base sont déjà créés plus haut (schedule ou pattern_streams)

            elif topo == "loop":
                cycle = float(template.get("cycle_time", template.get("travel_time", 0)))
                svc.template = ServiceTemplate(
                    topology="loop",
                    n_per_hour=int(template.get("n_per_hour", 0)),
                    equal_headway=bool(template.get("equal_headway", False)),
                    cycle_time=cycle,
                    turnaround=float(template.get("turnaround", 0)),
                )
                svc.anchors["default"] = svc.anchors.get("default") or Anchor("default")

            elif topo == "line":
                anchors = template.get("anchors") or ["A", "B"]
                svc.template = ServiceTemplate(
                    topology="line",
                    anchors=anchors,
                    n_per_hour=int(template.get("n_per_hour", 0)),
                    equal_headway=bool(template.get("equal_headway", False)),
                    tt_AB=float(template.get("tt_AB", template.get("travel_time", 0))),
                    tt_BA=float(template.get("tt_BA", template.get("travel_time", 0))),
                    turnaround_A=float(template.get("turnaround_A", template.get("turnaround", 0))),
                    turnaround_B=float(template.get("turnaround_B", template.get("turnaround", 0))),
                    phi_minutes=template.get("phi_minutes", "optimize"),
                )
                for a in anchors:
                    svc.anchors[a] = svc.anchors.get(a) or Anchor(a)

            elif topo == "route":  # <-- NEW
                stops = template.get("stops") or []
                legs = template.get("leg_minutes") or []
                if not stops or not legs or len(stops) != len(legs):
                    raise ValueError("route: 'stops' and 'leg_minutes' must be provided and have same length")
                svc.template = ServiceTemplate(
                    topology="route",
                    n_per_hour=int(template.get("n_per_hour", 0)),
                    equal_headway=bool(template.get("equal_headway", False)),
                    stops=stops,
                    leg_minutes=[int(x) for x in legs],
                    dwells={k: int(v) for k, v in (template.get("dwells") or {}).items()},
                )
                # créer une ancre par stop
                for sname in stops:
                    svc.anchors[sname] = svc.anchors.get(sname) or Anchor(sname)





            elif topo == "singletrack":
                # compat : accepter encore l'ancien champ "singletrack_config"
                path = template.get("st_config_path") or template.get("singletrack_config")
                if not path:
                    raise ValueError(
                        "singletrack: 'st_config_path' (ou ancien 'singletrack_config') est manquant dans template."
                    )
                svc.template = ServiceTemplate(
                    topology="singletrack",
                    st_config_path=path,
                    base_offset_min=int(template.get("base_offset_min", 0)),
                    offset_range_min=int(template.get("offset_range_min", 0)),
                    offset_step_min=int(template.get("offset_step_min", 1)),

                    st_n_dep_per_hour=int(template.get("st_n_dep_per_hour", 2)),
                    st_min_headway=float(template.get("st_min_headway", 5.0)),

                    # NEW (filtre faisabilité)
                    st_forbidden_meet_stops=(list(template["st_forbidden_meet_stops"]) if "st_forbidden_meet_stops" in template else None),

                    st_enable_no_overtaking=bool(template.get("st_enable_no_overtaking", True)),
                    st_extra_terminus_slack=float(template.get("st_extra_terminus_slack", 0.0)),
                )



            else:
                raise ValueError(f"Unknown topology: {topo}")

        services[name] = svc
    return services

def _build_transfers(cfg: Dict[str, Any]):
    from transit_core import Transfer
    outs = []
    for t in cfg.get("transfers", []):
        # compat ancien JSON: parent/child → from_service/to_service
        from_service = t.get("from_service") or t.get("parent")
        to_service   = t.get("to_service")   or t.get("child")
        outs.append(Transfer(
            from_service=from_service, from_anchor=t.get("from_anchor", "default"),
            to_service=to_service,     to_anchor=t.get("to_anchor", "default"),
            walk_time=float(t.get("walk_time", 0)),
            min_margin=float(t.get("min_margin", 0)),
            w_up=float(t.get("w_up", 0.5)),
            w_down=float(t.get("w_down", 0.5)),
            tie_breaker=t.get("tie_breaker", None),
            tie_params=t.get("tie_params", {}) or {}
        ))
    return outs

# --------- API publique : 1 fonction à appeler depuis run.py ---------
def run_from_json(config_path: str, targets: Optional[List[str]] = None) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # 1) fenêtre
    win = cfg.get("window", {})
    set_window(win.get("start_hhmm", "06:00"), win.get("end_hhmm", "09:00"))

    from transit_core import set_eval_grace
    set_eval_grace(before_min=0, after_min=0)  # marge de +20 min après la fenêtre
    # marge de +20 min après la fenêtre

    # 2) services + transferts
    services  = _build_services(cfg)
    transfers = _build_transfers(cfg)

    # 3) padding + expand streams, puis padding final
    set_padding(max(compute_recommended_padding(services, []), 120))
    for s in services.values():
        if s.pattern_streams:
            expand_pattern_streams(s)
    set_padding(max(compute_recommended_padding(services, transfers), 120))

    # 4) validation
    validate_services_and_transfers(services, transfers)

    # 5) tie-breakers plugin (optionnel)
    tb_module = ((cfg.get("tie_breakers") or {}).get("module")) or None
    external_tb = load_tie_breakers(tb_module) if tb_module else {}

    # 6) cibles + CSV
    targets = targets or cfg.get("targets") or [s.name for s in services.values() if s.template is not None]
    export_cfg = ((cfg.get("export") or {}).get("csv") or {})
    csv_path = export_cfg.get("path") if export_cfg.get("enabled") else None

    # 7) run
    results = optimize_services(
        services=services,
        transfers=transfers,
        targets=targets,
        external_tie_breakers=external_tb,
        include_ties=True,
        csv_path=csv_path
    )
    print_detailed_report(services, transfers, results)

    # 7bis) Export générique des distributions pour boxplots
    dists = extract_all_transfer_distributions(results, transfers)

    export_box = ((cfg.get("export") or {}).get("boxplots") or {})
    if export_box.get("enabled"):
        out_path = export_box.get("path", "boxplot_payload.json")
        case_name = cfg.get("case_name", config_path)
        payload = {"case": case_name, "transfers": dists}

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[export] boxplot payload written to: {out_path}")



    # 8) résumé succinct (+ petit aperçu pour loop)
    def _fmt(p): return " ".join(map(str, sorted(p)))
    print("\n=== Résultats ===")
    for r in results:
        phi = "" if r.candidate.phi is None else f", phi={r.candidate.phi}"
        off = r.candidate.offset_min if hasattr(r.candidate, "offset_min") else 0
        off_str = "" if off == 0 else f", offset={off} min"
        print(f"- {r.service_name}: [{_fmt(r.candidate.pattern)}]{phi}{off_str}")

    for r in results:
        svc = services[r.service_name]
        if svc.template and svc.template.topology == "loop":
            anc = next(iter(svc.anchors.values()))
            if anc.departures:
                print(f"  Aperçu {svc.name} deps:", [min_to_hhmm(t) for t in anc.departures[:6]])



def print_detailed_report(services: Dict[str, Service],
                          transfers: List["Transfer"],
                          results: List["EvalRecord"]) -> None:
    """
    Rapport lisible, sans doublons :
      - Par service optimisé :
        * pattern, phi, et ancre de référence du pattern
        * patterns (min/heure) des anchors impliqués dans des transferts
        * transferts affichés UNE SEULE FOIS : uniquement pour le service enfant (to_service)
    """
    from transit_core import min_to_hhmm  # pas utilisé pour l'instant, mais dispo si tu veux HH:MM plus tard

    def minutes_pattern(times: List[int]) -> List[int]:
        """Retourne les minutes distinctes dans l'heure (0..59) triées."""
        return sorted({t % 60 for t in times})

    def _fmt_waits(ws):
        """
        Affiche les temps d'attente de manière compacte :
        - on garde uniquement les valeurs DISTINCTES, triées
        - donc [24,17,2,24,17,2,...] devient [2, 17, 24]
        """
        if not ws:
            return "[]"
        ws_int = [int(round(x)) for x in ws]
        uniq = sorted(set(ws_int))
        if len(uniq) <= 6:
            return "[" + ", ".join(str(w) for w in uniq) + "]"
        return "[" + ", ".join(str(w) for w in uniq[:6]) + ", ...]"

    print("\n\n============ RAPPORT DÉTAILLÉ (COMPACT) ============\n")

    for r in results:
        svc = services[r.service_name]
        tmpl = svc.template
        topo = tmpl.topology if tmpl else "scheduled"

        # Identifier l'ancre de référence du pattern
        base_anchor = "default"
        if tmpl:
            if topo == "line" and tmpl.anchors:
                base_anchor = tmpl.anchors[0]
            elif topo == "route" and tmpl.stops:
                base_anchor = tmpl.stops[0]
            elif topo == "loop":
                base_anchor = next(iter(svc.anchors.keys()), "default")

        # Mode offset ?
        is_offset_mode = bool(
            tmpl
            and topo == "scheduled"
            and getattr(tmpl, "offset_range_min", 0) not in (0, None)
        )

        print(f"\n--- Service: {svc.name}  (topology={topo}) ---")

        if is_offset_mode:
            off = getattr(r.candidate, "offset_min", 0)
            sign = "+" if off >= 0 else ""
            print(f"  Offset choisi : {sign}{off} min (horaire de base décalé en bloc)")
            # Optionnel : tu peux afficher un petit résumé du range testé si tu veux plus tard
        else:
            patt_str = " ".join(str(x) for x in sorted(r.candidate.pattern))
            phi_str = "" if r.candidate.phi is None else f", phi={r.candidate.phi}"
            print(f"  Pattern choisi : [{patt_str}]{phi_str}")
            print(f"  Pattern = minutes dans l'heure des départs à l'ancre '{base_anchor}'")

        # Si le candidat a un offset (singletrack, etc.), on l’affiche aussi
        if hasattr(r.candidate, "offset_min") and r.candidate.offset_min != 0:
            off = int(r.candidate.offset_min)
            sign = "+" if off >= 0 else ""
            print(f"  Offset choisi : {sign}{off} min (appliqué au pattern singletrack)")


        print(f"  Score total    : {r.score:.3f}")

        # 1) Anchors impliqués dans des transferts (pour ce service)
        used_anchors = set()
        for tr in transfers:
            if tr.from_service == svc.name:
                used_anchors.add(tr.from_anchor)
            if tr.to_service == svc.name:
                used_anchors.add(tr.to_anchor)

        if used_anchors:
            print("  Anchors impliqués (pattern min/heure) :")
            for aname in sorted(used_anchors):
                anc = svc.anchors.get(aname)
                if not anc:
                    continue
                arr_pat = minutes_pattern(anc.arrivals)
                dep_pat = minutes_pattern(anc.departures)

                def _fmt_min_pattern(x):
                    if not x:
                        return "∅"
                    return "[" + " ".join(str(m) for m in x) + "]"

                print(f"    - {aname}:")
                print(f"        arrivals  : {_fmt_min_pattern(arr_pat)}")
                print(f"        departures: {_fmt_min_pattern(dep_pat)}")
        else:
            print("  (aucun anchor de ce service n'est impliqué dans un transfert)")

        # 2) Transferts : on n'affiche que ceux où ce service est le 'to_service'
        print("  Transferts vers ce service :")
        has_any = False
        for tr in transfers:
            if tr.to_service != svc.name:
                continue
            has_any = True

            sf = services[tr.from_service]
            st_ = services[tr.to_service]
            af = sf.anchors.get(tr.from_anchor) or sf.anchors.get("default")
            at = st_.anchors.get(tr.to_anchor) or st_.anchors.get("default")
            if af is None or at is None:
                continue

            try:
                stats = evaluate_transfer_guarded(af, at, tr)
            except ValueError as e:
                print(f"    • {tr.from_service}:{tr.from_anchor} -> "
                      f"{tr.to_service}:{tr.to_anchor}  ⚠ impossible: {e}")
                continue

            direction = f"{tr.from_service}:{tr.from_anchor} -> {tr.to_service}:{tr.to_anchor}"
            print(f"    • {direction}")
            print(f"        w_up={tr.w_up}, w_down={tr.w_down}")
            print(f"        UP   : n={stats.n_up}, avg={stats.avg_up_w:.2f} min, max={stats.max_up:.1f}")
            print(f"        DOWN : n={stats.n_down}, avg={stats.avg_down_w:.2f} min, max={stats.max_down:.1f}")
            print(f"        waits_up   = {_fmt_waits(stats.waits_up)}")
            print(f"        waits_down = {_fmt_waits(stats.waits_down)}")

        if not has_any:
            print("    (aucun transfert qui arrive sur ce service)")

        # 3) Horaire détaillé pour singletrack (minutes dans l'heure APRÈS offset)
        if topo == "singletrack":
            print("  Horaire détaillé (minute dans l'heure) :")

            # Ordre des arrêts : si on connaît la ligne singletrack, on la respecte
            stop_order = getattr(tmpl, "st_line_stops", None) or sorted(svc.anchors.keys())

            for sname in stop_order:
                anc = svc.anchors.get(sname)
                if not anc:
                    continue
                arr_pat = minutes_pattern(anc.arrivals)
                dep_pat = minutes_pattern(anc.departures)
                print(f"    {sname}:")
                print(f"       arrivals  : {arr_pat}")
                print(f"       departures: {dep_pat}")


    print("\n============ FIN RAPPORT DÉTAILLÉ (COMPACT) ============\n")




def extract_all_transfer_distributions(results, transfers):
    """
    Extraction générique des distributions d'attente pour TOUS les transferts rencontrés.
    Retourne un dict: transfer_key -> dict(...) avec values/weights UP/DOWN/ALL.
    Les poids directionnels w_up/w_down (définis dans le JSON transferts) sont appliqués
    aux poids d'observations (weights_up/weights_down) pour obtenir des poids 'effectifs'
    cohérents avec ton objectif global.
    """
    out = {}

    # map clé transfert -> Transfer (pour récupérer w_up/w_down/walk/margin/etc.)
    tr_by_key = {}
    for tr in transfers:
        k = f"{tr.from_service}:{tr.from_anchor}->{tr.to_service}:{tr.to_anchor}"
        tr_by_key[k] = tr

    for rec in results:
        trs = rec.details.get("transfers", {}) or {}
        for k, d in trs.items():
            tr = tr_by_key.get(k)
            if tr is None:
                # si jamais une clé est différente (rare), on skip
                continue

            waits_up = d.get("waits_up", []) or []
            waits_dn = d.get("waits_down", []) or []
            w_up_obs = d.get("weights_up", []) or []
            w_dn_obs = d.get("weights_down", []) or []

            # Sécurité si mismatch de longueur (au cas où)
            if len(waits_up) != len(w_up_obs):
                w_up_obs = [1.0] * len(waits_up)
            if len(waits_dn) != len(w_dn_obs):
                w_dn_obs = [1.0] * len(waits_dn)

            # Poids effectifs = poids d'observation * poids directionnel (w_up/w_down)
            w_up_eff = [float(tr.w_up) * float(w) for w in w_up_obs]
            w_dn_eff = [float(tr.w_down) * float(w) for w in w_dn_obs]

            out[k] = {
                # meta (utile pour labels / regroupements)
                "from_service": tr.from_service,
                "from_anchor": tr.from_anchor,
                "to_service": tr.to_service,
                "to_anchor": tr.to_anchor,
                "walk_time": tr.walk_time,
                "min_margin": tr.min_margin,
                "w_up": tr.w_up,
                "w_down": tr.w_down,

                # distributions
                "values_up": waits_up,
                "weights_up": w_up_eff,
                "values_down": waits_dn,
                "weights_down": w_dn_eff,

                # distribution mixte (si tu veux un seul boxplot par correspondance)
                "values_all": list(waits_up) + list(waits_dn),
                "weights_all": list(w_up_eff) + list(w_dn_eff),

                # stats rapides utiles (optionnel)
                "n_up": len(waits_up),
                "n_down": len(waits_dn),
            }

    return out
