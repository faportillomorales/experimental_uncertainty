import itertools
import os
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

####################################################################################################################################################
#                                            INPUTS
####################################################################################################################################################
# Dados de exemplo - MODIFIQUE CONFORME NECESSÁRIO
# Formato: {ID: (x_value, group_value, y_value)}
dataset: Dict[str, Tuple[float, float, float]] = {
    "P01": (0.5, 0.2, 0.35),
    "P02": (1.2, 0.2, 0.42),
    "P03": (2.8, 0.2, 0.58),
    "P04": (4.5, 0.2, 0.71),
    "P05": (0.6, 0.4, 0.48),
    "P06": (1.3, 0.4, 0.55),
    "P07": (2.9, 0.4, 0.68),
    "P08": (4.6, 0.4, 0.82),
    "P09": (0.7, 0.6, 0.62),
    "P10": (1.4, 0.6, 0.72),
    "P11": (3.0, 0.6, 0.85),
    "P12": (4.7, 0.6, 0.95),
    "P13": (0.8, 0.8, 0.78),
    "P14": (1.5, 0.8, 0.88),
    "P15": (3.1, 0.8, 1.02),
    "P16": (4.8, 0.8, 1.15),
}

# Configurações do gráfico
x_label = r"Superficial gas velocity, $J_g \, [m/s]$"  # Label do eixo X
y_label = r"Slug frequency, $f \, [Hz]$"  # Label do eixo Y (não normalizado)
y_label_normalized = r"Normalized slug frequency, $f/f_{\max}$"  # Label do eixo Y (normalizado)
group_label_template = r"$J_l = {value:.2f}$"  # Template para label do grupo (use {value} como placeholder)
title_label = r"$\theta = +5°$"  # Label do título/condição

# Configurações de saída
output_dir = "."  # Diretório para salvar os gráficos
save_figures = True  # Se True, salva as figuras; se False, apenas mostra
figure_format = "png"  # Formato da figura: 'png', 'pdf', 'svg', etc.
dpi = 300  # Resolução da figura

# Configurações visuais
figsize = (6, 6)  # Tamanho da figura (largura, altura)
linewidth = 1.5  # Espessura da linha
marker_size = 100  # Tamanho dos marcadores
alpha = 0.7  # Transparência das linhas
fontsize_labels = 16  # Tamanho da fonte dos labels dos eixos
fontsize_ticks = 14  # Tamanho da fonte dos ticks
fontsize_legend = 14  # Tamanho da fonte da legenda
fontsize_title = 14  # Tamanho da fonte do título

####################################################################################################################################################
#                                            END INPUTS
####################################################################################################################################################


def main() -> None:
    """Função principal que processa os dados e gera os gráficos."""
    # Extrai as colunas dos dados
    x_values, group_values, y_values = (np.array(series) for series in _extract_columns(dataset.values()))

    # Cria diretório de saída se necessário
    if save_figures:
        os.makedirs(output_dir, exist_ok=True)

    plt.style.use("default")

    # Gera gráfico não normalizado
    _plot_standard(
        x_values=x_values,
        group_values=group_values,
        y_values=y_values,
        title_label=title_label,
        normalized=False,
    )

    # Gera gráfico normalizado
    _plot_standard(
        x_values=x_values,
        group_values=group_values,
        y_values=y_values,
        title_label=title_label,
        normalized=True,
    )


def _plot_standard(
    x_values: np.ndarray,
    group_values: np.ndarray,
    y_values: np.ndarray,
    title_label: str,
    normalized: bool,
) -> None:
    """
    Plota gráfico padrão agrupado por valores únicos de group_values.
    
    Args:
        x_values: Valores do eixo X
        group_values: Valores para agrupar (cada grupo terá um marcador/linha diferente)
        y_values: Valores do eixo Y
        title_label: Label do título/condição
        normalized: Se True, normaliza os valores de y pelo máximo
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Ciclos para diferentes marcadores e estilos de linha
    marker_cycle = itertools.cycle(("o", "s", "D", "^", "v", "P", "X", "*"))
    line_style_cycle = itertools.cycle(("-", "--", "-.", ":"))

    unique_groups = np.unique(group_values)
    legend_handles = []

    y_plot = y_values.copy()
    if normalized:
        max_y = np.max(y_plot)
        y_plot = y_plot / max_y if max_y > 0 else y_plot

    # Plota cada grupo
    for group_value in unique_groups:
        mask = group_values == group_value
        x_subset = x_values[mask]
        y_subset = y_plot[mask]
        marker = next(marker_cycle)
        line_style = next(line_style_cycle)

        # Ordena por x para plotar linha contínua
        sort_indices = np.argsort(x_subset)
        x_sorted = x_subset[sort_indices]
        y_sorted = y_subset[sort_indices]

        # Plota linha
        ax.plot(
            x_sorted,
            y_sorted,
            color="black",
            linestyle=line_style,
            linewidth=linewidth,
            alpha=alpha,
            zorder=1,
        )

        # Plota marcadores
        ax.scatter(
            x_subset,
            y_subset,
            marker=marker,
            facecolors="black",
            edgecolors="black",
            linewidths=1.0,
            s=marker_size,
            zorder=3,
        )

        # Adiciona handle para legenda
        legend_handles.append(
            Line2D(
                [],
                [],
                color="black",
                linestyle=line_style,
                linewidth=linewidth,
                marker=marker,
                markerfacecolor="black",
                markeredgecolor="black",
                markersize=8,
                label=group_label_template.format(value=group_value),
            )
        )

    # Configura eixos
    ax.set_xlabel(x_label, fontsize=fontsize_labels)
    if normalized:
        ax.set_ylabel(y_label_normalized, fontsize=fontsize_labels)
    else:
        ax.set_ylabel(y_label, fontsize=fontsize_labels)
    ax.tick_params(axis="both", which="major", labelsize=fontsize_ticks)
    ax.legend(handles=legend_handles, frameon=True, fontsize=fontsize_legend)
    ax.grid(False)

    # Ajusta limites do eixo Y
    _, y_max = ax.get_ylim()
    ax.set_ylim(bottom=0, top=y_max)

    # Adiciona label do título/condição
    fig.text(
        0.4,
        0.85,
        title_label,
        fontsize=fontsize_title,
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

    # Salva ou mostra a figura
    if save_figures:
        suffix = "_normalized" if normalized else ""
        output_path = os.path.join(output_dir, f"standard_plot{suffix}.{figure_format}")
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Gráfico salvo em: {output_path}")
        plt.close(fig)
    else:
        plt.show()


def _extract_columns(
    rows: Sequence[Tuple[float, float, float]]
) -> Tuple[List[float], List[float], List[float]]:
    """
    Extrai colunas de uma sequência de tuplas.
    
    Args:
        rows: Sequência de tuplas (x, group, y)
    
    Returns:
        Tupla com três listas: (x_values, group_values, y_values)
    """
    x_list, group_list, y_list = [], [], []
    for x, group, y in rows:
        x_list.append(x)
        group_list.append(group)
        y_list.append(y)
    return x_list, group_list, y_list


if __name__ == "__main__":
    main()

