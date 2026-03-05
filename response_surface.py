"""
Superfície de resposta a partir de pontos dados.
Equivalente em Python ao script f22401.m (DACE/Kriging).
Usa Gaussian Process (sklearn) como substituto do modelo DACE com correlação Gaussiana.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler


# -------------------------------------------------------------------------
# Dados (equivalente à matriz Superf do .m)
# Valores: (J_air, J_water, resposta)
SUPERF: Dict[str, Tuple[float, float, float]] = {
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

# SUPERF: Dict[str, Tuple[float, float, float]] = {
#         "P01": (0.49, 0.20, 0.30),
#         "P02": (1.30, 0.20, 0.30),
#         "P03": (3.89, 0.20, 0.70),
#         # "P04": (10.22, 0.20, 0.13),
#         "P05": (0.51, 0.40, 0.53),
#         "P06": (1.28, 0.40, 0.61),
#         "P07": (3.96, 0.40, 1.01),
#         "P08": (9.64, 0.40, 1.33),
#         "P09": (0.51, 0.80, 1.57),
#         "P10": (1.30, 0.80, 1.17),
#         "P11": (4.19, 0.80, 1.37),
#         "P12": (7.93, 0.80, 0.92),
#         "P13": (0.49, 1.80, 5.64),
#         "P14": (0.95, 1.80, 3.13),
#         "P15": (2.06, 1.80, 2.52),
#         "P16": (3.61, 1.80, 2.05),
#     }


# Entradas (X) e resposta (y)
X_data = np.array([[v[0], v[1]] for v in SUPERF.values()])   # J_air, J_water
y_data = np.array([v[2] for v in SUPERF.values()])           # resposta

# -------------------------------------------------------------------------
# Grau de interpolação (suavidade da superfície)
# 1 = mais suave | 5 = mais interpoladora/ondulada (ajusta mais aos pontos)
# Internamente mapeia para os bounds do length_scale do kernel RBF do GP.
GRAU_INTERPOLACAO: int = 5

# -------------------------------------------------------------------------
# Regularização / ruído (reduz overfitting)
# alpha: nugget no GP; maior = superfície mais suave, menos interpolação exata.
# use_white_kernel: True = adiciona WhiteKernel ao kernel para o modelo aprender
#   o ruído; ajuda a evitar overfitting quando há incerteza nos dados.
ALPHA_GP: float = 1e-1        #1.5e-1
USE_WHITE_KERNEL: bool = False


def gridsamp(range_2n: np.ndarray, q) -> np.ndarray:
    """
    Gera grade n-dimensional no intervalo dado.
    range_2n: shape (2, n) com [lim_inf, lim_sup] por dimensão.
    q: número de pontos por dimensão (escalar) ou vetor (n,).
    """
    low, high = range_2n[0], range_2n[1]
    n = len(low)
    q = np.atleast_1d(q)
    if q.size == 1:
        q = np.full(n, int(np.asarray(q).flat[0]))
    meshes = np.meshgrid(
        *[np.linspace(low[i], high[i], int(q[i])) for i in range(n)],
        indexing="ij"
    )
    return np.column_stack([m.ravel() for m in meshes])


def _length_scale_bounds_from_grau(grau: int) -> Tuple[float, float]:
    """Mapeia GRAU_INTERPOLACAO (1–5) para (min, max) do length_scale do RBF."""
    # grau 1 = mais suave; grau 5 = mais interpoladora (bounds mais conservadores para reduzir overfitting)
    bounds_map = {
        1: (1.2, 30.0),
        2: (0.6, 20.0),
        3: (0.35, 15.0),
        4: (0.2, 10.0),
        5: (0.12, 5.0),
    }
    g = max(1, min(5, int(grau)))
    return bounds_map.get(g, bounds_map[3])


def fit_surface(
    X: np.ndarray,
    y: np.ndarray,
    length_scale_bounds: Optional[Tuple[float, float]] = None,
):
    """
    Ajusta superfície de resposta tipo Kriging (GP com kernel RBF + tendência constante).
    Equivalente a dacefit(..., @regpoly0, @corrgauss, ...).
    Usa alpha e opcionalmente WhiteKernel para reduzir overfitting.
    """
    if length_scale_bounds is None:
        length_scale_bounds = _length_scale_bounds_from_grau(GRAU_INTERPOLACAO)
    # Kernel: constante * RBF (Gaussiano); opcionalmente + WhiteKernel (ruído)
    kernel = ConstantKernel(1.0) * RBF(
        length_scale=[10.0, 10.0],
        length_scale_bounds=(length_scale_bounds[0], length_scale_bounds[1]),
    )
    if USE_WHITE_KERNEL:
        kernel = kernel + WhiteKernel(noise_level_bounds=(1e-6, 1.0))
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=ALPHA_GP,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=0,
    )
    gp.fit(X, y)
    return gp


def predict_surface(gp, X_grid: np.ndarray, return_std: bool = True):
    """Predição e MSE (desvio padrão) na grade."""
    if return_std:
        y_pred, std = gp.predict(X_grid, return_std=True)
        mse = std ** 2
        return y_pred, mse
    return gp.predict(X_grid), None


def plot_response_surface(
    X_data: np.ndarray,
    y_data: np.ndarray,
    X_grid: np.ndarray,
    Y_pred: np.ndarray,
    grid_shape: tuple,
    MSE: Optional[np.ndarray] = None,
    title: str = r"$\theta = 0^\circ$",
    stats_text: Optional[str] = None,
    xlabel: str = r"$J_{water}$ [m/s]",
    ylabel: str = r"$J_{air}$ [m/s]",
    zlabel: str = r"Slug frequency [Hz]",
):
    """Plota pontos e superfície de resposta (estilo f22401.m)."""
    n1, n2 = grid_shape
    X1 = X_grid[:, 0].reshape(n1, n2)
    X2 = X_grid[:, 1].reshape(n1, n2)
    YX = Y_pred.reshape(n1, n2)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    
    # Superfície prevista (superfície contínua colorida)
    surf = ax.plot_surface(
        X2, X1, YX,
        cmap="viridis", edgecolor="none",
        antialiased=True, rstride=1, cstride=1,
    )
    # plt.colorbar(surf, ax=ax, shrink=0.55, pad=0.10)
    # Pontos originais (preto, opacos)
    ax.scatter(
        X_data[:, 1], X_data[:, 0], y_data,
        s=50, c="k", marker="o", label="Dados",
        edgecolors="k", linewidths=1.2, alpha=1.0, depthshade=False
    )

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_zlabel(zlabel, fontsize=11)
    # ax.legend(loc="upper left", fontsize=9)

    try:
        ax.view_init(elev=20, azim=140, roll=0)
    except TypeError:
        ax.view_init(elev=20, azim=140)

    # fig.suptitle(title, fontsize=13, y=0.98)
    # if stats_text:
    #     fig.text(
    #         0.5, 0.02, stats_text,
    #         ha="center", va="bottom", fontsize=9,
    #         family="sans-serif",
    #         bbox=dict(
    #             boxstyle="round,pad=0.4",
    #             facecolor="white",
    #             edgecolor="#374151",
    #             linewidth=1.0,
    #             alpha=0.95,
    #         ),
    #     )
    fig.subplots_adjust(left=0.02, right=0.88, bottom=0.08, top=0.92)
    fig.tight_layout()

    return fig, ax


def plot_response_surface_2d(
    X_grid: np.ndarray,
    Y_pred: np.ndarray,
    grid_shape: tuple,
    X_data: np.ndarray,
    y_data: np.ndarray,
    xlabel: str = r"$j_g$ [m/s]",
    ylabel: str = r"$j_\ell$ [m/s]",
    zlabel: str = r"Slug frequency [Hz]",
):
    """Vista de topo 2D: eixos jg e jl, colorido pela frequência. Figura retangular; mapa (área de desenho) quadrado."""
    n1, n2 = grid_shape
    jg = X_grid[:, 0].reshape(n1, n2)   # J_air
    jl = X_grid[:, 1].reshape(n1, n2)   # J_water
    Z = Y_pred.reshape(n1, n2)

    fig, ax = plt.subplots(figsize=(7, 5))
    cf = ax.contourf(jg, jl, Z, levels=25, cmap="viridis")
    ax.scatter(
        X_data[:, 0], X_data[:, 1],
        c="k", s=40, edgecolors="white", linewidths=0.8, label="Dados", zorder=5
    )
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    # Eixo y: marcadores passo 0.2 de 0.2 até 1.8
    ax.set_yticks(np.arange(0.2, 1.81, 0.2))
    # Mapa quadrado (área de desenho quadrada)
    ax.set_position([0.12, 0.12, 0.68, 0.68])
    fig.colorbar(cf, ax=ax, shrink=0.7, pad=0.02, label=zlabel)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    return fig, ax


def main():
    # Limites do domínio a partir dos dados em SUPERF (com margem de 10%)
    x1_min, x1_max = X_data[:, 0].min(), X_data[:, 0].max()
    x2_min, x2_max = X_data[:, 1].min(), X_data[:, 1].max()
    margin_x1 = max(0.1, (x1_max - x1_min) * 0.1)
    margin_x2 = max(0.05, (x2_max - x2_min) * 0.1)
    range_2d = np.array([
        [x1_min - margin_x1, x2_min - margin_x2],
        [x1_max + margin_x1, x2_max + margin_x2],
    ])
    n_grid = 100  # pontos por eixo (50x50 = 2500 pontos; aumentar para superfície mais suave)

    # Grade para predição
    X_grid = gridsamp(range_2d, n_grid)
    grid_shape = (n_grid, n_grid)

    # Ajuste do modelo (equivalente a dacefit)
    gp = fit_surface(X_data, y_data)
    Y_pred, MSE = predict_surface(gp, X_grid, return_std=True)

    # R² e RMSE (Root Mean Square Error) nos pontos de ajuste
    y_pred_fit, _ = predict_surface(gp, X_data, return_std=False)
    ss_res = np.sum((y_data - y_pred_fit) ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    n_pts = len(y_data)
    rmse = np.sqrt(ss_res / n_pts) if n_pts > 0 else 0.0  # RMSE = sqrt(mean((y - y_pred)²))

    # Frequência nominal, da superfície e erro relativo por ponto
    f_nominal = y_data
    f_superficie = y_pred_fit
    with np.errstate(divide="ignore", invalid="ignore"):
        erro_relativo = np.where(
            np.abs(f_nominal) > 1e-12,
            (f_nominal - f_superficie) / f_nominal,
            np.nan,
        )
    pontos = list(SUPERF.keys())
    print("\n--- Frequência nominal vs superfície de resposta ---")
    print("Ponto   f_nominal [Hz]   f_superf [Hz]   erro_rel")
    for i, pt in enumerate(pontos):
        er = erro_relativo[i]
        er_str = f"{er*100:+.1f}%" if np.isfinite(er) else "  —"
        print(f"  {pt}   {f_nominal[i]:14.4f}   {f_superficie[i]:12.4f}   {er_str}")
    print("\nArrays (f_nominal, f_superficie, erro_relativo):")
    print("f_nominal   =", np.round(f_nominal, 4))
    print("f_superficie=", np.round(f_superficie, 4))
    print("erro_relativo (frac) =", np.round(erro_relativo, 4))
    print("erro_relativo (%)    =", np.round(erro_relativo * 100, 2))

    # Máximo e mínimo na superfície prevista
    YX_mat = Y_pred.reshape(grid_shape)
    max_idx = np.unravel_index(np.argmax(YX_mat), grid_shape)
    min_idx = np.unravel_index(np.argmin(YX_mat), grid_shape)
    X1_mat = X_grid[:, 0].reshape(grid_shape)
    X2_mat = X_grid[:, 1].reshape(grid_shape)
    max_yx, min_yx = YX_mat[max_idx], YX_mat[min_idx]
    max_x1, max_x2 = X1_mat[max_idx], X2_mat[max_idx]
    min_x1, min_x2 = X1_mat[min_idx], X2_mat[min_idx]

    print(f"RMSE (Root Mean Square Error): {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"Máximo: {max_yx:.2f} em J_air={max_x1:.2f}, J_water={max_x2:.2f}")
    print(f"Mínimo: {min_yx:.2f} em J_air={min_x1:.2f}, J_water={min_x2:.2f}")

    # Plot
    stats_lines = [f"R² = {r2:.4f}", f"RMSE = {rmse:.4f}"]
    stats_lines.append(f"Máx = {max_yx:.2f}  (J_air = {max_x1:.2f}, J_water = {max_x2:.2f})")
    stats_lines.append(f"Mín = {min_yx:.2f}  (J_air = {min_x1:.2f}, J_water = {min_x2:.2f})")
    stats_text = "\n".join(stats_lines)
    fig, ax = plot_response_surface(
        X_data, y_data, X_grid, Y_pred, grid_shape, MSE, stats_text=stats_text
    )

    # Vista de topo 2D: jg x jl, colorido pela frequência
    fig2, ax2 = plot_response_surface_2d(
        X_grid, Y_pred, grid_shape, X_data, y_data
    )

    plt.show()

    # Predição em novos pontos (equivalente a predictor(INPUT, dmodel))
    # Exemplo: descomente e defina INPUT para usar
    # INPUT = np.array([[1.0, 0.5], [5.0, 1.0]])
    # YXnew, _ = predict_surface(gp, INPUT, return_std=False)
    # print("Predições em INPUT:", YXnew)

    return gp, X_grid, Y_pred, MSE


if __name__ == "__main__":
    gp, X_grid, Y_pred, MSE = main()
