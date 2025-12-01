import itertools
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

TREND_POLY_DEGREE = 2  # ajuste aqui o grau máximo da curva de tendência


def main() -> None:
    dataset: Dict[str, Tuple[float, float, float]] = {
        "P01": (0.52, 0.20, 0.30),
        "P02": (1.27, 0.20, 0.23),
        "P03": (4.06, 0.20, 0.47),
        "P05": (0.49, 0.40, 0.54),
        "P06": (1.26, 0.40, 0.84),
        "P07": (4.02, 0.40, 0.83),
        "P09": (0.51, 0.80, 1.14),
        "P10": (1.36, 0.80, 0.97),
        "P11": (4.12, 0.80, 1.04),
        "P13": (0.50, 1.80, 3.26),
        "P14": (1.02, 1.80, 2.05),
        "P15": (2.04, 1.80, 1.60),
        "P16": (4.18, 1.80, 1.17),
    }

    jg, jl, freq = (np.array(series) for series in _extract_columns(dataset.values()))

    plt.style.use("default")
    _plot_slug_frequency(
        jg=jg,
        jl=jl,
        freq=freq,
        theta_label=r"$\theta = 0°$",
        normalized=False,
    )
    _plot_slug_frequency(
        jg=jg,
        jl=jl,
        freq=freq,
        theta_label=r"$\theta = 0°$",
        normalized=True,
    )


def _plot_slug_frequency(
    jg: np.ndarray,
    jl: np.ndarray,
    freq: np.ndarray,
    theta_label: str,
    normalized: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))

    marker_cycle = itertools.cycle(("o", "s", "D", "^", "v", "P", "X", "*"))
    line_style_cycle = itertools.cycle(("-", "--", "-.", ":"))

    unique_jl = np.unique(jl)
    legend_handles = []

    freq_plot = freq.copy()
    if normalized:
        max_freq = np.max(freq_plot)
        freq_plot = freq_plot / max_freq if max_freq > 0 else freq_plot

    for jl_value in unique_jl:
        mask = jl == jl_value
        jg_subset = jg[mask]
        freq_subset = freq_plot[mask]
        marker = next(marker_cycle)
        line_style = next(line_style_cycle)

        sort_indices = np.argsort(jg_subset)
        jg_sorted = jg_subset[sort_indices]
        freq_sorted = freq_subset[sort_indices]

        ax.plot(
            jg_sorted,
            freq_sorted,
            color="black",
            linestyle=line_style,
            linewidth=1.5,
            alpha=0.7,
            zorder=1,
        )

        ax.scatter(
            jg_subset,
            freq_subset,
            marker=marker,
            facecolors="black",
            edgecolors="black",
            linewidths=1.0,
            s=100,
            zorder=3,
        )
        legend_handles.append(
            Line2D(
                [],
                [],
                color="black",
                linestyle=line_style,
                linewidth=1.5,
                marker=marker,
                markerfacecolor="black",
                markeredgecolor="black",
                markersize=8,
                label=rf"$J_l = {jl_value:.2f}$",
            )
        )

    ax.set_xlabel(rf"Superficial gas velocity, $J_g \, [m/s]$", fontsize=16)
    if normalized:
        ax.set_ylabel(rf"Normalized slug frequency, $f/f_{{\max}}$", fontsize=16)
    else:
        ax.set_ylabel(rf"Slug frequency, $f \, [Hz]$", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.legend(handles=legend_handles, frameon=True, fontsize=14)
    ax.grid(False)
    _, y_max = ax.get_ylim()
    ax.set_ylim(bottom=0, top=y_max)
    fig.text(
        0.4,
        0.85,
        theta_label if not normalized else rf"{theta_label}",
        fontsize=14,
        ha="right",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="black",
            alpha=0.8,
        ),
    )
    fig.tight_layout()
    plt.show()


def _extract_columns(
    rows: Sequence[Tuple[float, float, float]]
) -> Tuple[List[float], List[float], List[float]]:
    gas, liquid, freq = [], [], []
    for jg, jl, f in rows:
        gas.append(jg)
        liquid.append(jl)
        freq.append(f)
    return gas, liquid, freq


if __name__ == "__main__":
    main()

