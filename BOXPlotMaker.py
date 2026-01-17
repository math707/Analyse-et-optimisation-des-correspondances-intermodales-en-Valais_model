# BoxPlotMaker.py
from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import matplotlib as mpl

plt.style.use("seaborn-v0_8-whitegrid")

COLOR_VMIN = 0.0
COLOR_VMAX = 65   # ajuste si tes μ montent plus haut (ex 20)
THR1 = 5.0
THR2 = 10
limite_axe = COLOR_VMAX

# -----------------------------
# Weighted boxplot stats
# -----------------------------
@dataclass
class BoxStats:
    label: str
    whislo: float
    q1: float
    med: float
    q3: float
    whishi: float
    mean: float
    n: int


def _clean_vw(values: List[float], weights: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v = v[mask]
    w = w[mask]
    if v.size == 0:
        return v, w
    order = np.argsort(v)
    return v[order], w[order]


def weighted_quantile(values: List[float], weights: List[float], qs: List[float]) -> List[float]:
    v, w = _clean_vw(values, weights)
    if v.size == 0:
        return [float("nan")] * len(qs)

    cw = np.cumsum(w)
    cw = cw / cw[-1]
    # Interpolation on cumulative weights
    return np.interp(qs, cw, v).tolist()


def weighted_mean(values: List[float], weights: List[float]) -> float:
    v, w = _clean_vw(values, weights)
    if v.size == 0:
        return float("nan")
    return float(np.sum(v * w) / np.sum(w))


def make_weighted_boxstats(label: str, values: List[float], weights: List[float]) -> Optional[BoxStats]:
    v, w = _clean_vw(values, weights)
    n = int(v.size)
    if n == 0:
        return None

    q1, med, q3 = weighted_quantile(v.tolist(), w.tolist(), [0.25, 0.50, 0.75])
    mu = weighted_mean(v.tolist(), w.tolist())

    # Weighted IQR whiskers: use the usual 1.5*IQR fences, then clip to existing data
    iqr = q3 - q1
    lo_fence = q1 - 1.5 * iqr
    hi_fence = q3 + 1.5 * iqr

    # whiskers at min/max within fences
    within = v[(v >= lo_fence) & (v <= hi_fence)]
    if within.size == 0:
        whislo = float(v[0])
        whishi = float(v[-1])
    else:
        whislo = float(within[0])
        whishi = float(within[-1])

    return BoxStats(
        label=label,
        whislo=whislo,
        q1=float(q1),
        med=float(med),
        q3=float(q3),
        whishi=whishi,
        mean=float(mu),
        n=n,
    )


# -----------------------------
# Color logic (Good/Moderate/Bad)
# -----------------------------
def _hex_to_rgb01(h: str) -> Tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp_rgb(c1: Tuple[float, float, float], c2: Tuple[float, float, float], t: float) -> Tuple[float, float, float]:
    return (_lerp(c1[0], c2[0], t), _lerp(c1[1], c2[1], t), _lerp(c1[2], c2[2], t))




def make_cmap_and_norm_fixed() -> Tuple[Any, Any]:
    """
    Colormap continue (style 'barre') avec seuils:
    0..2  : vert (clair->foncé)
    2..7  : jaune/orange (clair->foncé)
    >7    : rouge (démarre PLUS FONCÉ que l'orange foncé)
    + échelle fixe commune à tous les plots.
    """
    vmin, vmax = COLOR_VMIN, COLOR_VMAX
    if vmax <= vmin:
        vmax = vmin + 1e-6

    p1 = float(np.clip((THR1 - vmin) / (vmax - vmin), 0.0, 1.0))
    p2 = float(np.clip((THR2 - vmin) / (vmax - vmin), 0.0, 1.0))

    # Couleurs choisies pour garantir:
    # orange_dark < red_start en "foncé"
    green_light  = "#B7E4C7"
    green_dark   = "#1B7F5A"

    orange_light = "#FFE8C7"
    orange_dark  = "#C2410C"

    # IMPORTANT: red_start doit être déjà plus foncé que orange_dark
    red_start    = "#B91C1C"
    red_dark     = "#450A0A"

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "segmented_good_moderate_bad",
        [
            (0.00, green_light),
            (p1,  green_dark),
            (p1,  orange_light),
            (p2,  orange_dark),
            (p2,  red_start),
            (1.00, red_dark),
        ],
    )

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    return cmap, norm


def color_from_mean_bar(m: float, cmap, norm) -> Tuple[float, float, float, float]:
    if not np.isfinite(m):
        return (0.7, 0.7, 0.7, 1.0)
    return cmap(norm(m))



# -----------------------------
# IO: read all JSON payloads
# -----------------------------
def read_payloads(input_dir: str) -> List[Dict[str, Any]]:
    paths = sorted(glob.glob(os.path.join(input_dir, "*.json")))
    payloads = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        # fallback: case name = filename
        if "case" not in data:
            data["case"] = os.path.splitext(os.path.basename(p))[0]
        data["_path"] = p
        payloads.append(data)
    return payloads


def short_transfer_label(k: str) -> str:
    """
    k format: "FromService:FromAnchor->ToService:ToAnchor"
    label compact: "FromAnchor → ToAnchor" (si possible),
    sinon le key brut.
    """
    try:
        left, right = k.split("->")
        from_anchor = left.split(":")[-1]
        to_anchor = right.split(":")[-1]
        return f"{from_anchor} → {to_anchor}"
    except Exception:
        return k


# -----------------------------
# Plotting helpers
# -----------------------------
def plot_bxp(
    title: str,
    boxstats: List[BoxStats],
    out_path: str,
    rotate_x: int = 25,
    mean_fontsize: int = 12,
    axis_label_size: int = 13,
    tick_label_size: int = 12,
    title_size: int = 14,
    top_margin_ratio: float = 0.12,  # headroom en %
) -> None:
    if not boxstats:
        print(f"[plot] no data for: {title}")
        return

    stats_dicts = []
    cmap, norm = make_cmap_and_norm_fixed()
    colors = [color_from_mean_bar(bs.mean, cmap, norm) for bs in boxstats]

    colors = []
    for bs in boxstats:
        if np.isfinite(bs.mean):
            colors.append(cmap(norm(bs.mean)))
        else:
            colors.append((0.7, 0.7, 0.7, 1.0))

    fig, ax = plt.subplots(figsize=(max(8, 0.75 * len(boxstats)), 5.2))

    # --- Colorbar (légende thermique) ---
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # requis par Matplotlib
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.ax.tick_params(labelsize=tick_label_size)

    # --- build stats for matplotlib bxp ---
    for bs in boxstats:
        stats_dicts.append({
            "label": bs.label,
            "whislo": bs.whislo,
            "q1": bs.q1,
            "med": bs.med,
            "q3": bs.q3,
            "whishi": bs.whishi,
            "fliers": [],
        })

    b = ax.bxp(stats_dicts, showfliers=False, patch_artist=True, widths=0.6)

    # Color boxes + lignes un peu plus épaisses
    for patch, c in zip(b["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.95)
        patch.set_linewidth(1.2)

    for line_group in ("whiskers", "caps", "medians"):
        for ln in b[line_group]:
            ln.set_linewidth(1.2)

    # --- headroom auto (marge en haut) ---
    ax.set_ylim(0, limite_axe)

    # --- annotate means (plus grand + joli) ---
    # Place μ légèrement au-dessus de la moustache haute
    for i, bs in enumerate(boxstats, start=1):
        if not np.isfinite(bs.mean):
            continue

        y = bs.whishi
        # petit décalage vertical dépendant de l'échelle
        y0, y1 = ax.get_ylim()
        dy = 0.02 * (y1 - y0)
        ax.text(
            i, y + dy,
            f"μ = {bs.mean:.2f}",
            ha="center", va="bottom",
            fontsize=mean_fontsize,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.85),
            zorder=5
        )

    # --- style axes / labels ---
    #ax.set_title(title, fontsize=title_size, pad=10)
    ax.set_ylabel("Temps d’attente des passagers (min)", fontsize=axis_label_size)
    ax.grid(True, axis="y", alpha=0.25)

    ax.tick_params(axis="both", which="major", labelsize=tick_label_size)

    plt.xticks(rotation=rotate_x, ha="right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=250)  # dpi un peu plus propre
    plt.close(fig)
    print(f"[plot] wrote: {out_path}")



# -----------------------------
# Build box stats per mode
# -----------------------------
def mode_by_transfer(payloads: List[Dict[str, Any]], out_dir: str, use_field: str = "all") -> None:
    """
    1 figure per transfer_key, compare cases.
    use_field: "all" | "up" | "down"
    """
    # collect all transfer keys across cases
    all_keys = set()
    for pl in payloads:
        all_keys.update((pl.get("transfers") or {}).keys())

    for k in sorted(all_keys):
        boxstats = []
        for pl in payloads:
            case = pl.get("case", "case")
            t = (pl.get("transfers") or {}).get(k)
            if not t:
                continue

            if use_field == "up":
                values = t.get("values_up", [])
                weights = t.get("weights_up", [])
            elif use_field == "down":
                values = t.get("values_down", [])
                weights = t.get("weights_down", [])
            else:
                values = t.get("values_all", [])
                weights = t.get("weights_all", [])

            bs = make_weighted_boxstats(label=case, values=values, weights=weights)
            if bs:
                boxstats.append(bs)

        title = f"{short_transfer_label(k)}  ({use_field.upper()})"
        safe_name = "".join(ch if ch.isalnum() or ch in " _-." else "_" for ch in k)[:180]
        out_path = os.path.join(out_dir, f"by_transfer__{use_field}__{safe_name}.png")
        plot_bxp(title, boxstats, out_path)


def mode_by_case(payloads: List[Dict[str, Any]], out_dir: str, use_field: str = "all") -> None:
    """
    1 figure per case, compare transfers inside that case.
    """
    for pl in payloads:
        case = pl.get("case", "case")
        transfers = pl.get("transfers") or {}
        boxstats = []

        for k, t in transfers.items():
            if use_field == "up":
                values = t.get("values_up", [])
                weights = t.get("weights_up", [])
            elif use_field == "down":
                values = t.get("values_down", [])
                weights = t.get("weights_down", [])
            else:
                values = t.get("values_all", [])
                weights = t.get("weights_all", [])

            bs = make_weighted_boxstats(label=short_transfer_label(k), values=values, weights=weights)
            if bs:
                boxstats.append(bs)

        title = f"{case}  ({use_field.upper()})"
        safe_case = "".join(ch if ch.isalnum() or ch in " _-." else "_" for ch in case)[:120]
        out_path = os.path.join(out_dir, f"by_case__{use_field}__{safe_case}.png")
        plot_bxp(title, boxstats, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Folder containing out_*_boxplots.json files")
    ap.add_argument("--output_dir", required=True, help="Where to write figures")
    ap.add_argument("--mode", choices=["by_transfer", "by_case"], default="by_transfer")
    ap.add_argument("--field", choices=["all", "up", "down"], default="all")
    args = ap.parse_args()

    payloads = read_payloads(args.input_dir)
    if not payloads:
        raise SystemExit(f"No JSON files found in {args.input_dir}")

    if args.mode == "by_transfer":
        mode_by_transfer(payloads, args.output_dir, use_field=args.field)
    else:
        mode_by_case(payloads, args.output_dir, use_field=args.field)


def run_boxplots(input_dir: str,
                 output_dir: str,
                 mode: str = "by_transfer",
                 field: str = "all") -> None:
    """
    API simple pour être appelée depuis un runner.
    mode: "by_transfer" ou "by_case"
    field: "all", "up", "down"
    """
    payloads = read_payloads(input_dir)
    if not payloads:
        raise RuntimeError(f"No JSON files found in {input_dir}")

    if mode == "by_transfer":
        mode_by_transfer(payloads, output_dir, use_field=field)
    elif mode == "by_case":
        mode_by_case(payloads, output_dir, use_field=field)
    else:
        raise ValueError("mode must be 'by_transfer' or 'by_case'")



if __name__ == "__main__":
    main()
