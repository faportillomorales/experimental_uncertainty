"""
Gráficos de paridade: α e (∂P/∂z)_t previstos vs experimentais.

Dados embutidos por omissão: escoamento horizontal (AWH00, θ=0°), com Elongated bubble
e Slug; marcadores/cores de padrão vêm de ``plot_tool.get_flow_pattern_symbols``.

Referência visual: linha identidade (y = x), margens ±20 % (α) e ±30 % (dp/dz_t),
grade tracejada, legenda no canto inferior direito. Pontos agrupados na legenda
pelo padrão de escoamento (coluna ``flow_pattern``; omissão ⇒ Slug).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Dados embutidos: escoamento horizontal θ=0° (aba AWH00). Ver ``DEFAULT_ROWS_AW90`` para o caso inclinado 90°.
DEFAULT_ROWS_AWH00 = [
    ("AWH00P01", 0.62, 0.645, 0.1118, 0.056095, "Elongated Bubble"),
    ("AWH00P02", 0.71, 0.651, 0.2448, 0.033601, "Elongated Bubble"),
    ("AWH00P03", 0.78, 0.828, 0.2965, 0.101212, "Elongated Bubble"),
    ("AWH00P04", 0.84, 0.855, 0.4194, 0.532471, "Slug"),
    ("AWH00P05", 0.46, 0.489, 0.1803, 0.101963, "Elongated Bubble"),
    ("AWH00P06", 0.58, 0.729, 0.2312, 0.21121, "Elongated Bubble"),
    ("AWH00P07", 0.71, 0.885, 0.5098, 0.588382, "Elongated Bubble"),
    ("AWH00P08", 0.83, 0.805, 1.1323, 1.019368, "Slug"),
    ("AWH00P09", 0.25, 0.363, 0.268, 0.2794, "Slug"),
    ("AWH00P10", 0.46, 0.594, 0.4247, 0.460616, "Slug"),
    ("AWH00P11", 0.67, 0.812, 1.0918, 1.142187, "Slug"),
    ("AWH00P12", 0.82, 0.897, 2.4586, 2.442363, "Slug"),
    ("AWH00P13", 0.22, 0.212, 0.907, 1.062279, "Slug"),
    ("AWH00P14", 0.34, 0.362, 1.1516, 1.325878, "Slug"),
    ("AWH00P15", 0.57, 0.626, 2.4832, 2.389719, "Slug"),
    ("AWH00P16", 0.7, 0.725, 3.933, 3.44406, "Slug"),
]

# Caso inclinado 90° (AWU90 / AWD90), só Slug — usar ``--preset aw90`` na CLI
DEFAULT_ROWS_AW90 = [
    ("AWU90P01", 0.5, 0.455, 5.91349, 5.288435, "Slug"),
    ("AWU90P02", 0.72, 0.627, 4.09541, 3.608698, "Slug"),
    ("AWU90P03", 0.84, 0.753, 2.22487, 2.447987, "Slug"),
    ("AWU90P04", 0.93, 0.797, 1.6866, 2.034502, "Slug"),
    ("AWU90P05", 0.4, 0.388, 6.52041, 6.010262, "Slug"),
    ("AWU90P06", 0.61, 0.559, 4.62099, 4.374396, "Slug"),
    ("AWU90P07", 0.8, 0.72, 2.76434, 2.834609, "Slug"),
    ("AWU90P08", 0.92, 0.778, 2.99452, 2.280716, "Slug"),
    ("AWD90P09", 0.33, 0.287, 7.31646, 7.218083, "Slug"),
    ("AWD90P10", 0.52, 0.452, 5.41365, 5.573147, "Slug"),
    ("AWD90P11", 0.76, 0.661, 3.59995, 3.606777, "Slug"),
    ("AWD90P12", 0.89, 0.737, 3.63134, 2.940618, "Slug"),
    ("AWD90P13", 0.21, 0.172, 8.93968, 8.885971, "Slug"),
    ("AWD90P14", 0.35, 0.272, 7.99601, 7.942178, "Slug"),
    ("AWD90P15", 0.64, 0.495, 5.99797, 5.944429, "Slug"),
    ("AWD90P16", 0.7, 0.54, 5.8396, 5.582125, "Slug"),
]

DEFAULT_ROWS = DEFAULT_ROWS_AWH00

from plot_tool import (
    NAS_FLOW_PATTERN_MAP,
    _is_blank_excel_value,
    flow_pattern_display_label,
    get_flow_pattern_symbols,
    style_for_flow_pattern_cell,
)


def _normalize_nas_flow_pattern(val):
    """Mapeia códigos curtos NAS (SL, CH, …) para nomes completos, como em ``plot_tool``."""
    if _is_blank_excel_value(val):
        return "Slug"
    if isinstance(val, str) and val.strip() == "":
        return "Slug"
    s = str(val).strip()
    key = s.upper()
    return NAS_FLOW_PATTERN_MAP.get(key, NAS_FLOW_PATTERN_MAP.get(s, val))


def _scatter_style_for_flow_pattern(pattern_key: str, symbols: dict) -> tuple[dict, str]:
    """``marker``/``color`` alinhados a ``style_for_flow_pattern_cell``; legenda = ``flow_pattern_display_label``."""
    pd_ = style_for_flow_pattern_cell(pattern_key, symbols)
    style = {"marker": pd_["symbol"], "color": pd_["color"], "zorder": 3}
    return style, flow_pattern_display_label(pattern_key)


def _parity_axis_limits(x: np.ndarray, y: np.ndarray, margin: float = 0.06) -> tuple[float, float]:
    xy = np.concatenate([x.ravel(), y.ravel()])
    xy = xy[np.isfinite(xy)]
    if xy.size == 0:
        return 0.0, 1.0
    lo = float(np.min(xy))
    hi = float(np.max(xy))
    pad = (hi - lo) * margin + 1e-9
    # Eixos quadrados incluindo a origem quando os dados são positivos
    lo = min(0.0, lo - pad)
    hi = hi + pad
    return lo, hi


def _draw_parity_lines(ax, lo: float, hi: float, upper: float, lower: float, pct_label: str) -> None:
    """Linha y=x (sólida) e y=upper*x, y=lower*x (tracejadas), da origem até hi."""
    xs = np.array([max(0.0, lo), hi], dtype=float)
    if hi <= 0:
        xs = np.array([lo, 0.0], dtype=float)
    ax.plot(xs, xs, "k-", linewidth=1.2, label="y = x")
    ax.plot(xs, upper * xs, "k--", linewidth=1.0, label=f"±{pct_label}")
    ax.plot(xs, lower * xs, "k--", linewidth=1.0, label="_nolegend_")


def plot_parity_alpha_dpdz(
    df: pd.DataFrame | None = None,
    *,
    id_col: str = "ID",
    alpha_exp_col: str = "alpha_exp",
    alpha_pred_col: str = "alpha_pred",
    dpdz_exp_col: str = "dpdz_t_exp",
    dpdz_pred_col: str = "dpdz_t_pred",
    flow_pattern_col: str = "flow_pattern",
    side_by_side: bool = True,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure | tuple[plt.Figure, plt.Figure]:
    """
    Plota paridade para α (±20 %) e para (∂P/∂z)_t (±30 %).

    Parameters
    ----------
    df
        Colunas: ID, alpha experimental/predito, dp/dz total experimental/predito,
        e opcionalmente ``flow_pattern`` (nome completo ou código NAS: SL, CH, …).
        Se a coluna faltar, assume Slug. Estilos de marcador/cor: ``plot_tool.get_flow_pattern_symbols``.
        Se None, usa ``DEFAULT_ROWS`` (por omissão: AWH00 horizontal; ver também ``DEFAULT_ROWS_AW90``).
    flow_pattern_col
        Nome da coluna do padrão de escoamento (legenda por padrão, não por sentido).
    side_by_side
        True: uma figura com dois subplots. False: duas figuras separadas.
    save_path
        Se definido, salva PNG (um arquivo com dois painéis, ou use save separado
        chamando duas vezes com side_by_side=False).
    """
    if df is None:
        df = pd.DataFrame(
            DEFAULT_ROWS,
            columns=[
                id_col,
                alpha_exp_col,
                alpha_pred_col,
                dpdz_exp_col,
                dpdz_pred_col,
                flow_pattern_col,
            ],
        )
    else:
        df = df.copy()
        if flow_pattern_col not in df.columns:
            df[flow_pattern_col] = "Slug"

    symbols = get_flow_pattern_symbols()
    patterns_series = df[flow_pattern_col].map(_normalize_nas_flow_pattern)
    patterns_series = patterns_series.astype(str).str.strip().replace("", "Slug")
    patterns_list = patterns_series.tolist()
    unique_patterns = list(dict.fromkeys(patterns_list))
    def one_panel(ax, x, y, patterns, upper: float, lower: float, pct_label: str, xlab: str, ylab: str) -> None:
        lo, hi = _parity_axis_limits(x, y)
        _draw_parity_lines(ax, lo, hi, upper, lower, pct_label)
        for pname in unique_patterns:
            mask = np.array([p == pname for p in patterns])
            if not np.any(mask):
                continue
            style, leg_label = _scatter_style_for_flow_pattern(pname, symbols)
            ax.scatter(
                x[mask],
                y[mask],
                s=55,
                edgecolors="0.15",
                linewidths=0.6,
                facecolors=style["color"],
                marker=style["marker"],
                label=leg_label,
                zorder=style["zorder"],
            )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(xlab, fontsize=11)
        ax.set_ylabel(ylab, fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.45, color="0.55")
        ax.legend(loc="lower right", fontsize=9, framealpha=0.92)

    x_a = df[alpha_exp_col].to_numpy(dtype=float)
    y_a = df[alpha_pred_col].to_numpy(dtype=float)
    x_d = df[dpdz_exp_col].to_numpy(dtype=float)
    y_d = df[dpdz_pred_col].to_numpy(dtype=float)
    patterns = patterns_list

    if side_by_side:
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.5, 6.0), constrained_layout=True)
        one_panel(
            ax0,
            x_a,
            y_a,
            patterns,
            upper=1.2,
            lower=0.8,
            pct_label="20 %",
            xlab=r"$\alpha$ experimental [-]",
            ylab=r"$\alpha$ predicted [-]",
        )
        one_panel(
            ax1,
            x_d,
            y_d,
            patterns,
            upper=1.3,
            lower=0.7,
            pct_label="30 %",
            xlab=r"$\left(\frac{\partial P}{\partial z}\right)_\mathrm{t,\,exp}$ [kPa/m]",
            ylab=r"$\left(\frac{\partial P}{\partial z}\right)_\mathrm{t,\,pred}$ [kPa/m]",
        )
        if save_path is not None:
            fig.savefig(Path(save_path), dpi=300, bbox_inches="tight", facecolor="white")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig

    fig1, ax_a = plt.subplots(figsize=(7.5, 7.0), constrained_layout=True)
    one_panel(
        ax_a,
        x_a,
        y_a,
        patterns,
        upper=1.2,
        lower=0.8,
        pct_label="20 %",
        xlab=r"$\alpha$ experimental[-]",
        ylab=r"$\alpha$ predicted [-]",
    )
    fig2, ax_d = plt.subplots(figsize=(7.5, 7.0), constrained_layout=True)
    one_panel(
        ax_d,
        x_d,
        y_d,
        patterns,
        upper=1.3,
        lower=0.7,
        pct_label="30 %",
        xlab=r"$\left(\frac{\partial P}{\partial z}\right)_\mathrm{t,\,exp}$ [kPa/m]",
        ylab=r"$\left(\frac{\partial P}{\partial z}\right)_\mathrm{t,\,pred}$ [kPa/m]",
    )
    if save_path is not None:
        p = Path(save_path)
        suf = p.suffix if p.suffix else ".png"
        fig1.savefig(p.with_name(f"{p.stem}_alpha{suf}"), dpi=300, bbox_inches="tight", facecolor="white")
        fig2.savefig(p.with_name(f"{p.stem}_dpdz_t{suf}"), dpi=300, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    else:
        plt.close(fig1)
        plt.close(fig2)
    return fig1, fig2


def main() -> None:
    parser = argparse.ArgumentParser(description="Paridade α e (∂P/∂z)_t: predito vs experimental.")
    parser.add_argument(
        "--preset",
        choices=("awh00", "aw90"),
        default="awh00",
        help="Dados embutidos sem --csv: awh00=horizontal AWH00 (Elongated bubble + Slug); aw90=AWU90/AWD90 só Slug.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="CSV com colunas: ID, alpha_exp, alpha_pred, dpdz_t_exp, dpdz_t_pred; opcional: flow_pattern (omissão ⇒ Slug).",
    )
    parser.add_argument("--save", type=Path, default=None, help="Ficheiro de saída (PNG/PDF). Com --separate, usa stem_alpha e stem_dpdz_t.")
    parser.add_argument("--separate", action="store_true", help="Duas figuras em vez de um painel lado a lado.")
    parser.add_argument("--no-show", action="store_true", help="Não abrir janela interativa (útil em CI).")
    args = parser.parse_args()
    if args.csv is not None:
        df = pd.read_csv(args.csv)
    else:
        rows = DEFAULT_ROWS_AW90 if args.preset == "aw90" else DEFAULT_ROWS_AWH00
        df = pd.DataFrame(
            rows,
            columns=["ID", "alpha_exp", "alpha_pred", "dpdz_t_exp", "dpdz_t_pred", "flow_pattern"],
        )
    plot_parity_alpha_dpdz(
        df,
        side_by_side=not args.separate,
        save_path=args.save,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()