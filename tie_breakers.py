# tie_breakers.py
# ------------------------------------------------------------
# Registre "plugin" de stratégies de départage (tie-breakers)
# Utilisation côté moteur (une seule fois) :
#   from tie_breakers import TIE_BREAKERS
# Puis dans la config JSON d'un lien :
#   "tie_breaker": "penalty",
#   "tie_params": {"max_wait": 15, "lambda": 2.0}
#
# Chaque stratégie reçoit:
#   - cands: List[dict]  (solutions ex æquo, "brutes", non modifiées)
#     clés disponibles par candidat:
#       pattern, score,
#       avg_up_w, avg_down_w, avg_up, avg_down,
#       n_up, n_down,
#       max_up, max_down, std_up, std_down
#   - params: Dict[str, Any]
# Et DOIT retourner un des dicts de 'cands' (sans le modifier).
# ------------------------------------------------------------

from typing import List, Dict, Any, Callable

# --- Registre global ---
TIE_BREAKERS: Dict[str, Callable[[List[dict], Dict[str, Any]], dict]] = {}

def register_tie_breaker(name: str):
    """Décorateur pour enregistrer automatiquement une stratégie."""
    def decorator(func: Callable[[List[dict], Dict[str, Any]], dict]):
        TIE_BREAKERS[name] = func
        return func
    return decorator

# --- Outils communs (clés stables) ---

def _lexi_stable_key(d: dict):
    """Ordre de repli stable et lisible pour départager strictement à la fin."""
    return (d.get("avg_down_w", float("inf")),
            d.get("avg_up_w", float("inf")),
            d.get("pattern", []))

def _max_wait_of(d: dict) -> float:
    """Pire attente observée, tous sens confondus."""
    return max(d.get("max_up", float("inf")), d.get("max_down", float("inf")))

def _variance_proxy_of(d: dict) -> float:
    """Proxy simple de variance: moyenne des écarts-types up/down."""
    return (d.get("std_up", 0.0) + d.get("std_down", 0.0)) / 2.0

# --- Stratégies de base ---

@register_tie_breaker("none")
def tb_identity(cands: List[dict], params: Dict[str, Any]) -> dict:
    """
    Aucune règle secondaire : renvoie le premier en ordre lexicographique.
    Utile comme fallback stable.
    """
    return sorted(cands, key=lambda d: d.get("pattern", []))[0]

@register_tie_breaker("penalty")
def tb_penalty(cands: List[dict], params: Dict[str, Any]) -> dict:
    """
    Pénalise le dépassement d'un seuil de temps d'attente maximum (tous sens).
    params attendus:
      - max_wait (float, minutes) : seuil "acceptable"
      - lambda   (float) : intensité de la pénalité par minute au-dessus du seuil
    """
    T = params.get("max_wait", None)
    lam = float(params.get("lambda", 0.0))
    if T is None or lam <= 0.0:
        return tb_identity(cands, params)

    def key(d: dict):
        over = _max_wait_of(d) - T
        penalty = lam * max(0.0, over)
        # On ne modifie pas d["score"] (reste brut) : on s'en sert juste pour départager.
        # Tie secondaire lisible : favoriser la descente, puis la montée, puis pattern.
        return (penalty, d.get("avg_down_w", float("inf")), d.get("avg_up_w", float("inf")), d.get("pattern", []))

    return min(cands, key=key)

@register_tie_breaker("min_max")
def tb_min_max(cands: List[dict], params: Dict[str, Any]) -> dict:
    """
    Minimise la pire attente (max montée/descente). Très interprétable.
    """
    return min(
        cands,
        key=lambda d: (_max_wait_of(d),) + _lexi_stable_key(d)
    )

@register_tie_breaker("min_variance")
def tb_min_variance(cands: List[dict], params: Dict[str, Any]) -> dict:
    """
    Minimise la dispersion des attentes (proxy par moyenne des écarts-types).
    """
    return min(
        cands,
        key=lambda d: (_variance_proxy_of(d),) + _lexi_stable_key(d)
    )


@register_tie_breaker("penalty_sum")
def tb_penalty_sum(cands: List[dict], params: Dict[str, Any]) -> dict:
    """
    Pénalise la somme de tous les dépassements individuels (montée + descente).
    params:
      - max_wait (float, minutes) : seuil "acceptable"
      - lambda   (float) : intensité de la pénalité par minute de dépassement
    Nécessite que chaque candidat contienne 'waits_up' et 'waits_down'.
    """
    T = params.get("max_wait", None)
    lam = float(params.get("lambda", 0.0))
    if T is None or lam <= 0.0:
        # si paramètres manquants → repli stable
        return tb_identity(cands, params)

    def key(d: dict):
        wu = d.get("waits_up", []) or []
        wd = d.get("waits_down", []) or []
        penalty = 0.0
        for w in wu:
            if w > T: penalty += lam * (w - T)
        for w in wd:
            if w > T: penalty += lam * (w - T)
        # tie secondaire lisible
        return (penalty, d.get("avg_down_w", float("inf")), d.get("avg_up_w", float("inf")), d.get("pattern", []))

    return min(cands, key=key)


# --- (Optionnel) Chaînage de stratégies ---
# Permet d'appliquer plusieurs règles en cascade sur l'ensemble ex æquo.
# Exemple d’usage:
#   chosen = chain_strategies(cands, [
#       ("penalty", {"max_wait": 15, "lambda": 2.0}),
#       ("min_max", {})
#   ])
def chain_strategies(cands: List[dict], chain: List[tuple]) -> dict:
    """
    Applique plusieurs tie-breakers en cascade.
    'chain' est une liste de tuples (name, params).
    Retourne le meilleur candidat selon la dernière règle appliquée.
    """
    if not cands:
        raise ValueError("chain_strategies: liste vide")
    pool = list(cands)
    for name, params in chain:
        fn = TIE_BREAKERS.get(name)
        if not callable(fn):
            continue
        # On applique la stratégie et on retient le "meilleur"
        chosen = fn(pool, params or {})
        # Si plusieurs sont strictement ex æquo pour cette règle, on peut filtrer :
        # on garde ceux dont la clé est égale à la clé du "chosen"
        def rule_key_map(d):
            if name == "penalty":
                T = params.get("max_wait", None)
                lam = float(params.get("lambda", 0.0))
                if T is None or lam <= 0.0:
                    return _lexi_stable_key(d)
                over = _max_wait_of(d) - T
                pen = lam * max(0.0, over)
                return (pen,) + _lexi_stable_key(d)
            elif name == "min_max":
                return (_max_wait_of(d),) + _lexi_stable_key(d)
            elif name == "min_variance":
                return (_variance_proxy_of(d),) + _lexi_stable_key(d)
            else:
                return _lexi_stable_key(d)

        chosen_key = rule_key_map(chosen)
        pool = [d for d in pool if rule_key_map(d) == chosen_key]
        # Si la pool se réduit à 1, on peut s'arrêter
        if len(pool) == 1:
            return pool[0]

    # Repli stable final si la chaîne n'a pas tranché totalement
    return sorted(pool, key=lambda d: d.get("pattern", []))[0]
