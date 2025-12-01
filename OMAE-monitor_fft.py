import os
from typing import Tuple, Optional

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy import signal


####################################################################################################################################################
#                                            INPUTS
####################################################################################################################################################
file_path = "data_example/example/OMAE/monitor_nodes/monitor_node_46_cortado.txt"
freq_corte = None #20  # Frequência de corte do filtro passa-baixa em Hz - MODIFIQUE ESTE VALOR CONFORME NECESSÁRIO
freq_min_plot = 0.0  # Frequência mínima em Hz para plotar a FFT - MODIFIQUE ESTE VALOR CONFORME NECESSÁRIO
freq_max_plot = 100.0  # Frequência máxima em Hz para plotar a FFT - MODIFIQUE ESTE VALOR CONFORME NECESSÁRIO
time_start = 2.0  # Tempo inicial da janela em segundos (None = início dos dados) - MODIFIQUE ESTE VALOR CONFORME NECESSÁRIO
time_end = None  # Tempo final da janela em segundos (None = fim dos dados) - MODIFIQUE ESTE VALOR CONFORME NECESSÁRIO
remove_edge_points = 300  # Número de pontos a remover no início e fim após filtragem (para reduzir artefatos) - MODIFIQUE ESTE VALOR CONFORME NECESSÁRIO

####################################################################################################################################################
#                                            END INPUTS
####################################################################################################################################################

LABEL_FONT_SIZE = 16
TICK_FONT_SIZE = 14
LEGEND_FONT_SIZE = 14
LINE_COLOR = "black"
LINEWIDTH = 1.8
FIGSIZE = (6, 6)


def read_monitor_file(file_path: str) -> pd.DataFrame:
    """
    Lê o arquivo de monitoramento e retorna um DataFrame com os dados.
    
    Formato esperado:
    - Primeira linha: cabeçalho com TIME, X, Y, Z, FX, FY, FZ, MX, MY, MZ
    - Dados numéricos separados por tabulação
    """
    try:
        df = pd.read_csv(
            file_path,
            sep="\t",
            skiprows=0,
            decimal=".",
            na_values=[""],
            encoding="utf-8",
        )
        
        # Verifica se as colunas de deslocamento existem
        required_cols = ["TIME", "X", "Y", "Z"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Colunas faltantes no arquivo: {missing_cols}")
        
        return df
    
    except Exception as e:
        print(f"Erro ao ler arquivo: {e}")
        raise


def _base_axes_style(ax, xlabel: str, ylabel: str, hide_ticks: bool = False):
    ax.set_xlabel(xlabel, fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONT_SIZE)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONT_SIZE)
    if hide_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.grid(False)


def _save_figure(fig: plt.Figure, path: str):
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def apply_lowpass_filter(data: np.ndarray, fs: float, fc: Optional[float], order: int = 4, padlen: Optional[int] = None):
    """
    Aplica um filtro passa-baixa Butterworth aos dados.
    
    Args:
        data (np.ndarray): Dados de entrada
        fs (float): Frequência de amostragem em Hz
        fc (float or None): Frequência de corte em Hz (None = sem filtragem)
        order (int): Ordem do filtro (padrão: 4)
        padlen (int, optional): Comprimento do padding para reduzir artefatos nas bordas
        
    Returns:
        np.ndarray: Dados filtrados (ou originais se fc=None)
    """
    # Se fc for None, retorna os dados originais sem filtragem
    if fc is None:
        return data
    
    # Normaliza a frequência de corte
    nyquist = fs / 2
    normalized_cutoff = fc / nyquist
    
    # Projeta o filtro Butterworth
    b, a = signal.butter(order, normalized_cutoff, btype='low', analog=False)
    
    # Se padlen não for especificado, usa um valor padrão baseado no comprimento do sinal
    if padlen is None:
        # Usa 3 vezes o comprimento do filtro ou 10% do sinal, o que for menor
        padlen = min(3 * max(len(b), len(a)), len(data) // 10)
    
    # Aplica o filtro com padding para reduzir artefatos nas bordas
    filtered_data = signal.filtfilt(b, a, data, padlen=padlen)
    
    return filtered_data


def plot_time_series(
    time: np.ndarray,
    x_data: np.ndarray,
    y_data: np.ndarray,
    z_data: np.ndarray,
    x_filtered: np.ndarray,
    y_filtered: np.ndarray,
    z_filtered: np.ndarray,
    freq_corte: float,
    output_path: str,
):
    """Plota a série temporal das variáveis de deslocamento (apenas raw data)."""
    fig, axes = plt.subplots(3, 1, figsize=(6, 12))
    
    # Plota X (apenas raw)
    axes[0].plot(time, x_data, color="black", linestyle="-", linewidth=LINEWIDTH, alpha=0.9)
    _base_axes_style(axes[0], xlabel="", ylabel="X Displacement (m)", hide_ticks=False)
    axes[0].grid(True, alpha=0.3, linestyle="--")
    
    # Plota Y (apenas raw)
    axes[1].plot(time, y_data, color="gray", linestyle="-", linewidth=LINEWIDTH, alpha=0.9)
    _base_axes_style(axes[1], xlabel="", ylabel="Y Displacement (m)", hide_ticks=False)
    axes[1].grid(True, alpha=0.3, linestyle="--")
    
    # Plota Z (apenas raw)
    axes[2].plot(time, z_data, color="darkgray", linestyle="-", linewidth=LINEWIDTH, alpha=0.9)
    _base_axes_style(axes[2], xlabel="Time (s)", ylabel="Z Displacement (m)", hide_ticks=False)
    axes[2].grid(True, alpha=0.3, linestyle="--")
    
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.3)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_y_filtered_time_series(
    time: np.ndarray,
    y_raw: np.ndarray,
    y_filtered: np.ndarray,
    freq_corte: float,
    output_path: str,
):
    """Plota a série temporal do deslocamento Y (apenas raw data) em formato retangular."""
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plota apenas sinal raw em preto
    ax.plot(time, y_raw, color="black", linestyle="-", linewidth=LINEWIDTH, alpha=0.9)
    _base_axes_style(ax, xlabel="Time (s)", ylabel="Y Displacement (m)", hide_ticks=False)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_yticks([])  # Remove valores do eixo Y
    ax.set_xticks([])  # Remove valores do eixo X
    
    # Define limites do eixo Y baseados nos dados com pequena margem
    y_min = np.min(y_raw)
    y_max = np.max(y_raw)
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.2 * y_range, y_max + 0.2 * y_range)
    # Define limite do eixo X entre valores mínimos e máximos da série temporal
    ax.set_xlim(time.min(), time.max())
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fft(
    freqs: np.ndarray,
    magnitude_x_orig: np.ndarray,
    magnitude_y_orig: np.ndarray,
    magnitude_z_orig: np.ndarray,
    magnitude_x_filt: np.ndarray,
    magnitude_y_filt: np.ndarray,
    magnitude_z_filt: np.ndarray,
    freq_corte: Optional[float],
    freq_min_plot: float,
    freq_max_plot: float,
    output_path: str,
):
    """Plota a FFT das variáveis de deslocamento (original e filtrado)."""
    fig, axes = plt.subplots(3, 1, figsize=(6, 12))
    
    # Filtra frequências entre freq_min_plot e freq_max_plot
    mask = (freqs >= freq_min_plot) & (freqs <= freq_max_plot)
    freqs_filtered = freqs[mask]
    mag_x_orig_filt = magnitude_x_orig[mask]
    mag_y_orig_filt = magnitude_y_orig[mask]
    mag_z_orig_filt = magnitude_z_orig[mask]
    mag_x_filt_filt = magnitude_x_filt[mask]
    mag_y_filt_filt = magnitude_y_filt[mask]
    mag_z_filt_filt = magnitude_z_filt[mask]
    
    # Encontra picos de maior energia na FFT filtrada
    if len(freqs_filtered) > 0:
        idx_peak_x = int(np.argmax(mag_x_filt_filt))
        idx_peak_y = int(np.argmax(mag_y_filt_filt))
        idx_peak_z = int(np.argmax(mag_z_filt_filt))
        
        peak_freq_x = float(freqs_filtered[idx_peak_x])
        peak_freq_y = float(freqs_filtered[idx_peak_y])
        peak_freq_z = float(freqs_filtered[idx_peak_z])
        
        peak_mag_x = float(mag_x_filt_filt[idx_peak_x])
        peak_mag_y = float(mag_y_filt_filt[idx_peak_y])
        peak_mag_z = float(mag_z_filt_filt[idx_peak_z])
    
    # Plota X
    axes[0].plot(freqs_filtered, mag_x_orig_filt, color="black", linestyle="--", linewidth=LINEWIDTH, label="Original", alpha=0.5)
    if freq_corte is not None:
        axes[0].plot(freqs_filtered, mag_x_filt_filt, color="black", linestyle="-", linewidth=LINEWIDTH + 0.4, label=f"Filtered (fc={freq_corte} Hz)", alpha=0.9)
        axes[0].axvline(x=freq_corte, color="red", linestyle=":", linewidth=1.5, alpha=0.7, label=f"fc = {freq_corte} Hz")
    else:
        axes[0].plot(freqs_filtered, mag_x_filt_filt, color="black", linestyle="-", linewidth=LINEWIDTH + 0.4, label="Signal (no filter)", alpha=0.9)
    legend_handles_x = []
    if len(freqs_filtered) > 0:
        axes[0].axvline(x=peak_freq_x, color="blue", linestyle=":", linewidth=2.0, alpha=0.7)
        axes[0].scatter(peak_freq_x, peak_mag_x, color="blue", s=100, zorder=5, marker="o", edgecolors="darkblue", linewidths=1.5)
        legend_handles_x.append(
            Line2D([], [], marker="o", color="blue", linestyle="None", markersize=8, 
                   markeredgecolor="darkblue", markeredgewidth=1.5, label=f"Peak: {peak_freq_x:.2f} Hz")
        )
    _base_axes_style(axes[0], xlabel="", ylabel="X Magnitude", hide_ticks=False)
    axes[0].set_xlim(freq_min_plot, freq_max_plot)
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].grid(True, alpha=0.3, linestyle="--")
    # Adiciona handles de legenda existentes mais o pico
    existing_handles, existing_labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles=existing_handles + legend_handles_x, frameon=True, fontsize=LEGEND_FONT_SIZE - 2)
    
    # Plota Y
    axes[1].plot(freqs_filtered, mag_y_orig_filt, color="gray", linestyle="--", linewidth=LINEWIDTH, label="Original", alpha=0.5)
    if freq_corte is not None:
        axes[1].plot(freqs_filtered, mag_y_filt_filt, color="gray", linestyle="-", linewidth=LINEWIDTH + 0.4, label=f"Filtered (fc={freq_corte} Hz)", alpha=0.9)
        axes[1].axvline(x=freq_corte, color="red", linestyle=":", linewidth=1.5, alpha=0.7, label=f"fc = {freq_corte} Hz")
    else:
        axes[1].plot(freqs_filtered, mag_y_filt_filt, color="gray", linestyle="-", linewidth=LINEWIDTH + 0.4, label="Signal (no filter)", alpha=0.9)
    legend_handles_y = []
    if len(freqs_filtered) > 0:
        axes[1].axvline(x=peak_freq_y, color="blue", linestyle=":", linewidth=2.0, alpha=0.7)
        axes[1].scatter(peak_freq_y, peak_mag_y, color="blue", s=100, zorder=5, marker="o", edgecolors="darkblue", linewidths=1.5)
        legend_handles_y.append(
            Line2D([], [], marker="o", color="blue", linestyle="None", markersize=8, 
                   markeredgecolor="darkblue", markeredgewidth=1.5, label=f"Peak: {peak_freq_y:.2f} Hz")
        )
    _base_axes_style(axes[1], xlabel="", ylabel="Y Magnitude", hide_ticks=False)
    axes[1].set_xlim(freq_min_plot, freq_max_plot)
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].grid(True, alpha=0.3, linestyle="--")
    existing_handles, existing_labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles=existing_handles + legend_handles_y, frameon=True, fontsize=LEGEND_FONT_SIZE - 2)
    
    # Plota Z
    axes[2].plot(freqs_filtered, mag_z_orig_filt, color="darkgray", linestyle="--", linewidth=LINEWIDTH, label="Original", alpha=0.5)
    if freq_corte is not None:
        axes[2].plot(freqs_filtered, mag_z_filt_filt, color="darkgray", linestyle="-", linewidth=LINEWIDTH + 0.4, label=f"Filtered (fc={freq_corte} Hz)", alpha=0.9)
        axes[2].axvline(x=freq_corte, color="red", linestyle=":", linewidth=1.5, alpha=0.7, label=f"fc = {freq_corte} Hz")
    else:
        axes[2].plot(freqs_filtered, mag_z_filt_filt, color="darkgray", linestyle="-", linewidth=LINEWIDTH + 0.4, label="Signal (no filter)", alpha=0.9)
    legend_handles_z = []
    if len(freqs_filtered) > 0:
        axes[2].axvline(x=peak_freq_z, color="blue", linestyle=":", linewidth=2.0, alpha=0.7)
        axes[2].scatter(peak_freq_z, peak_mag_z, color="blue", s=100, zorder=5, marker="o", edgecolors="darkblue", linewidths=1.5)
        legend_handles_z.append(
            Line2D([], [], marker="o", color="blue", linestyle="None", markersize=8, 
                   markeredgecolor="darkblue", markeredgewidth=1.5, label=f"Peak: {peak_freq_z:.2f} Hz")
        )
    _base_axes_style(axes[2], xlabel="Frequency (Hz)", ylabel="Z Magnitude", hide_ticks=False)
    axes[2].set_xlim(freq_min_plot, freq_max_plot)
    axes[2].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[2].grid(True, alpha=0.3, linestyle="--")
    existing_handles, existing_labels = axes[2].get_legend_handles_labels()
    axes[2].legend(handles=existing_handles + legend_handles_z, frameon=True, fontsize=LEGEND_FONT_SIZE - 2)
    
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.3)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def apply_fft(
    time: np.ndarray,
    x_data: np.ndarray,
    y_data: np.ndarray,
    z_data: np.ndarray,
    x_filtered: np.ndarray,
    y_filtered: np.ndarray,
    z_filtered: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Aplica FFT nas variáveis de deslocamento (original e filtrado).
    
    Returns:
        Tupla com (frequencies, magnitude_x_orig, magnitude_y_orig, magnitude_z_orig,
                   magnitude_x_filt, magnitude_y_filt, magnitude_z_filt)
    """
    # Calcula intervalo de tempo
    dt = np.mean(np.diff(time))
    fs = 1 / dt  # Frequência de amostragem
    
    n = len(time)
    
    # Aplica FFT nos dados originais
    fft_x_orig = np.fft.fft(x_data)
    fft_y_orig = np.fft.fft(y_data)
    fft_z_orig = np.fft.fft(z_data)
    
    # Aplica FFT nos dados filtrados
    fft_x_filt = np.fft.fft(x_filtered)
    fft_y_filt = np.fft.fft(y_filtered)
    fft_z_filt = np.fft.fft(z_filtered)
    
    # Calcula frequências
    freqs = np.fft.fftfreq(n, dt)
    
    # Calcula magnitudes originais
    magnitude_x_orig = np.abs(fft_x_orig)
    magnitude_y_orig = np.abs(fft_y_orig)
    magnitude_z_orig = np.abs(fft_z_orig)
    
    # Calcula magnitudes filtradas
    magnitude_x_filt = np.abs(fft_x_filt)
    magnitude_y_filt = np.abs(fft_y_filt)
    magnitude_z_filt = np.abs(fft_z_filt)
    
    # Filtra apenas frequências positivas
    positive_freq_mask = freqs > 0
    freqs_pos = freqs[positive_freq_mask]
    mag_x_orig_pos = magnitude_x_orig[positive_freq_mask]
    mag_y_orig_pos = magnitude_y_orig[positive_freq_mask]
    mag_z_orig_pos = magnitude_z_orig[positive_freq_mask]
    mag_x_filt_pos = magnitude_x_filt[positive_freq_mask]
    mag_y_filt_pos = magnitude_y_filt[positive_freq_mask]
    mag_z_filt_pos = magnitude_z_filt[positive_freq_mask]
    
    return freqs_pos, mag_x_orig_pos, mag_y_orig_pos, mag_z_orig_pos, mag_x_filt_pos, mag_y_filt_pos, mag_z_filt_pos


def apply_time_window(
    time: np.ndarray,
    x_data: np.ndarray,
    y_data: np.ndarray,
    z_data: np.ndarray,
    time_start: Optional[float],
    time_end: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Aplica janelamento temporal nos dados.
    
    Args:
        time: Array de tempos
        x_data: Array de dados X
        y_data: Array de dados Y
        z_data: Array de dados Z
        time_start: Tempo inicial da janela (None = início)
        time_end: Tempo final da janela (None = fim)
    
    Returns:
        Tupla com (time_windowed, x_windowed, y_windowed, z_windowed)
    """
    # Cria máscara para janelamento
    window_mask = np.ones(len(time), dtype=bool)
    
    if time_start is not None:
        window_mask = window_mask & (time >= time_start)
    
    if time_end is not None:
        window_mask = window_mask & (time <= time_end)
    
    time_windowed = time[window_mask]
    x_windowed = x_data[window_mask]
    y_windowed = y_data[window_mask]
    z_windowed = z_data[window_mask]
    
    return time_windowed, x_windowed, y_windowed, z_windowed


def process_monitor_data(df: pd.DataFrame, output_dir: str, base_name: str, freq_corte: Optional[float], freq_min_plot: float, freq_max_plot: float, time_start: Optional[float], time_end: Optional[float]):
    """Processa os dados de monitoramento e gera os gráficos."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extrai dados
    time = df["TIME"].values
    x_data = df["X"].values
    y_data = df["Y"].values
    z_data = df["Z"].values
    
    # Remove valores inválidos
    valid_mask = np.isfinite(time) & np.isfinite(x_data) & np.isfinite(y_data) & np.isfinite(z_data)
    time_clean = time[valid_mask]
    x_clean = x_data[valid_mask]
    y_clean = y_data[valid_mask]
    z_clean = z_data[valid_mask]
    
    # Aplica janelamento temporal
    time_windowed, x_windowed, y_windowed, z_windowed = apply_time_window(
        time_clean, x_clean, y_clean, z_clean, time_start, time_end
    )
    
    if len(time_windowed) == 0:
        raise ValueError("Nenhum dado encontrado na janela temporal especificada!")
    
    print(f"Janela temporal aplicada:")
    if time_start is not None:
        print(f"  Tempo inicial: {time_start} s")
    else:
        print(f"  Tempo inicial: {time_windowed[0]:.6f} s (início dos dados)")
    if time_end is not None:
        print(f"  Tempo final: {time_end} s")
    else:
        print(f"  Tempo final: {time_windowed[-1]:.6f} s (fim dos dados)")
    print(f"  Duração: {time_windowed[-1] - time_windowed[0]:.6f} s")
    print(f"  Número de pontos: {len(time_windowed)}")
    
    # Calcula frequência de amostragem
    dt = np.mean(np.diff(time_windowed))
    fs = 1 / dt
    
    # Aplica filtro passa-baixa (ou retorna dados originais se freq_corte=None)
    x_filtered = apply_lowpass_filter(x_windowed, fs, freq_corte)
    y_filtered = apply_lowpass_filter(y_windowed, fs, freq_corte)
    z_filtered = apply_lowpass_filter(z_windowed, fs, freq_corte)
    
    # Remove pontos nas bordas para reduzir artefatos do filtro (apenas se filtro foi aplicado)
    if freq_corte is not None and remove_edge_points > 0 and len(time_windowed) > 2 * remove_edge_points:
        n_remove = min(remove_edge_points, len(time_windowed) // 10)  # Remove no máximo 10% do sinal
        time_filtered = time_windowed[n_remove:-n_remove]
        x_filtered = x_filtered[n_remove:-n_remove]
        y_filtered = y_filtered[n_remove:-n_remove]
        z_filtered = z_filtered[n_remove:-n_remove]
        x_windowed_plot = x_windowed[n_remove:-n_remove]
        y_windowed_plot = y_windowed[n_remove:-n_remove]
        z_windowed_plot = z_windowed[n_remove:-n_remove]
    else:
        time_filtered = time_windowed
        x_windowed_plot = x_windowed
        y_windowed_plot = y_windowed
        z_windowed_plot = z_windowed
    
    # Plota série temporal
    plot_time_series(
        time_filtered,
        x_windowed_plot,
        y_windowed_plot,
        z_windowed_plot,
        x_filtered,
        y_filtered,
        z_filtered,
        freq_corte,
        os.path.join(output_dir, f"time_series_{base_name}.png"),
    )
    
    # Plota série temporal do deslocamento Y filtrado (formato retangular)
    plot_y_filtered_time_series(
        time_filtered,
        y_windowed_plot,
        y_filtered,
        freq_corte,
        os.path.join(output_dir, f"time_series_y_filtered_{base_name}.png"),
    )
    
    # Aplica FFT
    freqs, mag_x_orig, mag_y_orig, mag_z_orig, mag_x_filt, mag_y_filt, mag_z_filt = apply_fft(
        time_filtered, x_windowed_plot, y_windowed_plot, z_windowed_plot, x_filtered, y_filtered, z_filtered
    )
    
    # Plota FFT
    plot_fft(
        freqs,
        mag_x_orig,
        mag_y_orig,
        mag_z_orig,
        mag_x_filt,
        mag_y_filt,
        mag_z_filt,
        freq_corte,
        freq_min_plot,
        freq_max_plot,
        os.path.join(output_dir, f"fft_{base_name}.png"),
    )
    
    # Calcula e imprime frequências dominantes (original e filtrado)
    if len(freqs) > 0:
        idx_dom_x_orig = int(np.argmax(mag_x_orig))
        idx_dom_y_orig = int(np.argmax(mag_y_orig))
        idx_dom_z_orig = int(np.argmax(mag_z_orig))
        idx_dom_x_filt = int(np.argmax(mag_x_filt))
        idx_dom_y_filt = int(np.argmax(mag_y_filt))
        idx_dom_z_filt = int(np.argmax(mag_z_filt))
        
        f_dom_x_orig = float(freqs[idx_dom_x_orig])
        f_dom_y_orig = float(freqs[idx_dom_y_orig])
        f_dom_z_orig = float(freqs[idx_dom_z_orig])
        f_dom_x_filt = float(freqs[idx_dom_x_filt])
        f_dom_y_filt = float(freqs[idx_dom_y_filt])
        f_dom_z_filt = float(freqs[idx_dom_z_filt])
        
        print(f"\nFrequências dominantes (original):")
        print(f"  X: {f_dom_x_orig:.6f} Hz")
        print(f"  Y: {f_dom_y_orig:.6f} Hz")
        print(f"  Z: {f_dom_z_orig:.6f} Hz")
        print(f"\nFrequências dominantes (filtrado):")
        print(f"  X: {f_dom_x_filt:.6f} Hz")
        print(f"  Y: {f_dom_y_filt:.6f} Hz")
        print(f"  Z: {f_dom_z_filt:.6f} Hz")
    
    return freqs, mag_x_orig, mag_y_orig, mag_z_orig, mag_x_filt, mag_y_filt, mag_z_filt


if __name__ == "__main__":
    print(f"\nLendo arquivo: {file_path}")
    
    try:
        df = read_monitor_file(file_path)
        print(f"Dados carregados com sucesso!")
        print(f"Número de pontos: {len(df)}")
        print(f"\nColunas disponíveis: {list(df.columns)}")
        
        # Informações sobre os dados
        dt = np.mean(np.diff(df["TIME"].values))
        fs = 1 / dt
        print(f"\nFrequência de amostragem: {fs:.2f} Hz")
        print(f"Frequência de Nyquist: {fs/2:.2f} Hz")
        print(f"Intervalo de tempo médio: {dt:.6f} s")
        
    except Exception as e:
        print(f"Erro ao ler arquivo: {e}")
        raise SystemExit(1)
    
    output_dir = os.path.dirname(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    print(f"\nProcessando dados...")
    if freq_corte is not None:
        print(f"Frequência de corte do filtro: {freq_corte} Hz")
    else:
        print(f"Nenhum filtro será aplicado (freq_corte = None)")
    print(f"Faixa de frequência para plotar FFT: {freq_min_plot} - {freq_max_plot} Hz")
    
    process_monitor_data(df, output_dir, base_name, freq_corte, freq_min_plot, freq_max_plot, time_start, time_end)
    
    print("\nAnálise concluída!")
    print(f"Gráficos salvos em: {output_dir}")
    print(f"  - Série temporal: time_series_{base_name}.png")
    print(f"  - FFT: fft_{base_name}.png")

