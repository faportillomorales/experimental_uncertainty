import os
from typing import Tuple, Optional

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, MultipleLocator
import numpy as np
import pandas as pd
from scipy import signal
from nptdms import TdmsFile


####################################################################################################################################################
#                                            INPUTS
####################################################################################################################################################
# Arquivo experimental (TDMS)
tdms_file_path = "data_example/example/freq_slug/AOU05P03/AOU05P03_acc.tdms"
tdms_column = "/'Untitled'/'Accel1'"  # Coluna do arquivo TDMS para análise - MODIFIQUE ESTE VALOR CONFORME NECESSÁRIO

# Arquivo de simulação (monitor node)
monitor_file_path = "data_example/example/OMAE/monitor_nodes/monitor_node_58-refined.txt"

# Parâmetros de processamento (mesmos do OMAE-monitor_fft.py)
freq_corte = 1000  # Frequência de corte do filtro passa-baixa em Hz - MODIFIQUE ESTE VALOR CONFORME NECESSÁRIO
freq_min_plot = 0.0  # Frequência mínima em Hz para plotar a FFT - MODIFIQUE ESTE VALOR CONFORME NECESSÁRIO
freq_max_plot = 80.0  # Frequência máxima em Hz para plotar a FFT - MODIFIQUE ESTE VALOR CONFORME NECESSÁRIO
time_start = 2.0  # Tempo inicial da janela em segundos (None = início dos dados) - MODIFIQUE ESTE VALOR CONFORME NECESSÁRIO
time_end = None  # Tempo final da janela em segundos (None = fim dos dados) - MODIFIQUE ESTE VALOR CONFORME NECESSÁRIO
remove_edge_points = 300  # Número de pontos a remover no início e fim após filtragem (para reduzir artefatos) - MODIFIQUE ESTE VALOR CONFORME NECESSÁRIO

####################################################################################################################################################
#                                            END INPUTS
####################################################################################################################################################

LABEL_FONT_SIZE = 20
TICK_FONT_SIZE = 18
LEGEND_FONT_SIZE = 12
LINE_COLOR = "black"
LINEWIDTH = 1.8
FIGSIZE = (6, 6)


def read_tdms_file(file_path: str) -> pd.DataFrame:
    """
    Lê o arquivo TDMS e retorna um DataFrame com os dados.
    Segue o mesmo padrão do fft_analysis.py.
    """
    try:
        # Lê o arquivo TDMS
        tdms_file = TdmsFile.read(file_path)
        
        # Converte para DataFrame
        df = tdms_file.as_dataframe()
        
        # Cria coluna de tempo baseada na coluna de tempo do TDMS (mesmo padrão do fft_analysis.py)
        if "/'Untitled'/'Time'" in df.columns:
            # Converte timestamps para segundos relativos
            time_col = df["/'Untitled'/'Time'"]
            start_time = time_col.iloc[0]
            df['X_Value'] = (time_col - start_time).dt.total_seconds()
        else:
            # Fallback: cria uma coluna de tempo baseada no índice
            df['X_Value'] = np.arange(len(df)) * 0.1  # Assume 10 Hz de frequência de amostragem
        
        return df
        
    except Exception as e:
        print(f"Erro ao ler arquivo TDMS: {e}")
        raise


def read_monitor_file(file_path: str) -> pd.DataFrame:
    """
    Lê o arquivo de monitoramento e retorna um DataFrame com os dados.
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


def apply_lowpass_filter(data: np.ndarray, fs: float, fc: float, order: int = 4, padlen: Optional[int] = None):
    """
    Aplica um filtro passa-baixa Butterworth aos dados.
    Se padlen=None, usa o mesmo padrão do fft_analysis.py (sem padlen no filtfilt).
    """
    # Normaliza a frequência de corte
    nyquist = fs / 2
    normalized_cutoff = fc / nyquist
    
    # Projeta o filtro Butterworth
    b, a = signal.butter(order, normalized_cutoff, btype='low', analog=False)
    
    # Se padlen não for especificado, usa filtfilt sem padlen (padrão do fft_analysis.py)
    if padlen is None:
        filtered_data = signal.filtfilt(b, a, data)
    else:
        # Aplica o filtro com padding para reduzir artefatos nas bordas
        filtered_data = signal.filtfilt(b, a, data, padlen=padlen)
    
    return filtered_data


def apply_time_window(
    time: np.ndarray,
    data: np.ndarray,
    time_start: Optional[float],
    time_end: Optional[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aplica janelamento temporal nos dados.
    """
    # Cria máscara para janelamento
    window_mask = np.ones(len(time), dtype=bool)
    
    if time_start is not None:
        window_mask = window_mask & (time >= time_start)
    
    if time_end is not None:
        window_mask = window_mask & (time <= time_end)
    
    time_windowed = time[window_mask]
    data_windowed = data[window_mask]
    
    return time_windowed, data_windowed


def process_tdms_data(
    df: pd.DataFrame,
    column: str,
    freq_corte: float,
    freq_min_plot: float,
    freq_max_plot: float,
    time_start: Optional[float],
    time_end: Optional[float],
    remove_edge_points: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Processa dados do arquivo TDMS e retorna FFT.
    Segue o mesmo padrão do fft_analysis.py: sem janelamento temporal e sem remoção de pontos.
    
    Returns:
        (freqs, magnitude_orig, magnitude_filt)
    """
    # Extrai dados (usa X_Value como no fft_analysis.py)
    time = df["X_Value"].values
    y_data = df[column].values
    
    # Remove valores inválidos
    valid_mask = np.isfinite(time) & np.isfinite(y_data)
    time_clean = time[valid_mask]
    y_clean = y_data[valid_mask]
    
    # Calcula frequência de amostragem (mesmo padrão do fft_analysis.py)
    dt = np.mean(np.diff(time_clean))
    fs = 1 / dt
    
    # Aplica filtro passa-baixa (sem padlen, como no fft_analysis.py)
    y_filtered = apply_lowpass_filter(y_clean, fs, freq_corte, padlen=None)
    
    # Aplica FFT (mesmo padrão do fft_analysis.py - usa todos os dados)
    n = len(y_clean)
    fft_orig = np.fft.fft(y_clean)
    fft_filt = np.fft.fft(y_filtered)
    freqs = np.fft.fftfreq(n, dt)
    
    magnitude_orig = np.abs(fft_orig)
    magnitude_filt = np.abs(fft_filt)
    
    # Filtra apenas frequências positivas
    positive_freq_mask = freqs > 0
    freqs_pos = freqs[positive_freq_mask]
    mag_orig_pos = magnitude_orig[positive_freq_mask]
    mag_filt_pos = magnitude_filt[positive_freq_mask]
    
    return freqs_pos, mag_orig_pos, mag_filt_pos


def process_monitor_data(
    df: pd.DataFrame,
    freq_corte: float,
    freq_min_plot: float,
    freq_max_plot: float,
    time_start: Optional[float],
    time_end: Optional[float],
    remove_edge_points: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Processa dados do monitor node e retorna FFT da soma dos componentes X, Y e Z.
    
    Returns:
        (freqs, magnitude_orig, magnitude_filt)
    """
    # Extrai dados e soma X, Y e Z
    time = df["TIME"].values
    x_data = df["X"].values
    y_data = df["Y"].values
    z_data = df["Z"].values
    
    # Soma dos deslocamentos
    total_data = x_data + y_data + z_data
    
    # Remove valores inválidos
    valid_mask = np.isfinite(time) & np.isfinite(x_data) & np.isfinite(y_data) & np.isfinite(z_data)
    time_clean = time[valid_mask]
    total_clean = total_data[valid_mask]
    
    # Aplica janelamento temporal
    time_windowed, total_windowed = apply_time_window(
        time_clean, total_clean, time_start, time_end
    )
    
    if len(time_windowed) == 0:
        raise ValueError("Nenhum dado encontrado na janela temporal especificada!")
    
    # Calcula frequência de amostragem
    dt = np.mean(np.diff(time_windowed))
    fs = 1 / dt
    
    # Aplica filtro passa-baixa
    total_filtered = apply_lowpass_filter(total_windowed, fs, freq_corte)
    
    # Remove pontos nas bordas para reduzir artefatos do filtro
    if remove_edge_points > 0 and len(time_windowed) > 2 * remove_edge_points:
        n_remove = min(remove_edge_points, len(time_windowed) // 10)
        time_filtered = time_windowed[n_remove:-n_remove]
        total_filtered = total_filtered[n_remove:-n_remove]
        total_windowed_plot = total_windowed[n_remove:-n_remove]
    else:
        time_filtered = time_windowed
        total_windowed_plot = total_windowed
    
    # Aplica FFT
    n = len(time_filtered)
    dt_fft = np.mean(np.diff(time_filtered))
    
    fft_orig = np.fft.fft(total_windowed_plot)
    fft_filt = np.fft.fft(total_filtered)
    freqs = np.fft.fftfreq(n, dt_fft)
    
    magnitude_orig = np.abs(fft_orig)
    magnitude_filt = np.abs(fft_filt)
    
    # Filtra apenas frequências positivas
    positive_freq_mask = freqs > 0
    freqs_pos = freqs[positive_freq_mask]
    mag_orig_pos = magnitude_orig[positive_freq_mask]
    mag_filt_pos = magnitude_filt[positive_freq_mask]
    
    return freqs_pos, mag_orig_pos, mag_filt_pos


def plot_combined_fft(
    freqs_tdms: np.ndarray,
    magnitude_tdms_orig: np.ndarray,
    magnitude_tdms_filt: np.ndarray,
    freqs_monitor: np.ndarray,
    magnitude_monitor_orig: np.ndarray,
    magnitude_monitor_filt: np.ndarray,
    freq_corte: float,
    freq_min_plot: float,
    freq_max_plot: float,
    output_path: str,
):
    """Plota FFT combinado: experimental (em cima) e simulação (embaixo)."""
    n_panels = 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(6, 5.5 * n_panels))
    
    # Filtra frequências entre freq_min_plot e freq_max_plot
    mask_tdms = (freqs_tdms >= freq_min_plot) & (freqs_tdms <= freq_max_plot)
    mask_monitor = (freqs_monitor >= freq_min_plot) & (freqs_monitor <= freq_max_plot)
    
    freqs_tdms_filt = freqs_tdms[mask_tdms]
    mag_tdms_orig_filt = magnitude_tdms_orig[mask_tdms]
    mag_tdms_filt_filt = magnitude_tdms_filt[mask_tdms]
    
    freqs_monitor_filt = freqs_monitor[mask_monitor]
    mag_monitor_orig_filt = magnitude_monitor_orig[mask_monitor]
    mag_monitor_filt_filt = magnitude_monitor_filt[mask_monitor]
    
    # Encontra picos de maior energia na FFT filtrada
    peak_freq_tdms = None
    peak_mag_tdms = None
    peak_freq_monitor = None
    peak_mag_monitor = None
    
    if len(freqs_tdms_filt) > 0:
        idx_peak_tdms = int(np.argmax(mag_tdms_filt_filt))
        peak_freq_tdms = float(freqs_tdms_filt[idx_peak_tdms])
        peak_mag_tdms = float(mag_tdms_filt_filt[idx_peak_tdms])
    
    if len(freqs_monitor_filt) > 0:
        idx_peak_monitor = int(np.argmax(mag_monitor_filt_filt))
        peak_freq_monitor = float(freqs_monitor_filt[idx_peak_monitor])
        peak_mag_monitor = float(mag_monitor_filt_filt[idx_peak_monitor])
    
    # Plota experimental (em cima) - subplot (a)
    axes[0].plot(freqs_tdms_filt, mag_tdms_filt_filt, color="black", linestyle="-", linewidth=LINEWIDTH + 0.4, alpha=0.9)
    axes[0].axvline(x=freq_corte, color="red", linestyle=":", linewidth=1.5, alpha=0.7)
    # Linha da frequência de slug
    slug_freq = 0.7
    axes[0].axvline(x=slug_freq, color="red", linestyle="--", linewidth=2.5, alpha=0.7)
    legend_handles_tdms = []
    legend_handles_tdms.append(
        Line2D([], [], color="red", linestyle="--", linewidth=2.5, label=f"Slug frequency: {slug_freq} Hz")
    )
    if peak_freq_tdms is not None:
        axes[0].scatter(peak_freq_tdms, peak_mag_tdms, color="blue", s=100, zorder=5, marker="o", edgecolors="darkblue", linewidths=1.5)
        legend_handles_tdms.append(
            Line2D([], [], marker="o", color="blue", linestyle="None", markersize=8, 
                   markeredgecolor="darkblue", markeredgewidth=1.5, label=f"Peak: {peak_freq_tdms:.2f} Hz")
        )
    _base_axes_style(axes[0], xlabel="Frequency (Hz)", ylabel="Magnitude", hide_ticks=False)
    axes[0].set_xlim(freq_min_plot, freq_max_plot)
    axes[0].xaxis.set_major_locator(MultipleLocator(10))
    axes[0].set_yticks([])  # Remove valores do eixo Y
    axes[0].grid(True, alpha=0.3, linestyle="--")
    if len(legend_handles_tdms) > 0:
        axes[0].legend(handles=legend_handles_tdms, frameon=True, fontsize=LEGEND_FONT_SIZE - 2)
    # Adiciona índice (a) abaixo do subplot (centralizado)
    axes[0].text(0.5, -0.2, "(a)", transform=axes[0].transAxes, fontsize=LABEL_FONT_SIZE, ha='center', va='top')
    
    # Plota simulação (embaixo) - subplot (b)
    axes[1].plot(freqs_monitor_filt, mag_monitor_filt_filt, color="darkgray", linestyle="-", linewidth=LINEWIDTH + 0.4, alpha=0.9)
    axes[1].axvline(x=freq_corte, color="red", linestyle=":", linewidth=1.5, alpha=0.7)
    # Linha da frequência de slug
    slug_freq = 0.7
    axes[1].axvline(x=slug_freq, color="red", linestyle="--", linewidth=2.5, alpha=0.7)
    legend_handles_monitor = []
    legend_handles_monitor.append(
        Line2D([], [], color="red", linestyle="--", linewidth=2.5, label=f"Slug frequency: {slug_freq} Hz")
    )
    if peak_freq_monitor is not None:
        axes[1].scatter(peak_freq_monitor, peak_mag_monitor, color="blue", s=100, zorder=5, marker="o", edgecolors="darkblue", linewidths=1.5)
        legend_handles_monitor.append(
            Line2D([], [], marker="o", color="blue", linestyle="None", markersize=8, 
                   markeredgecolor="darkblue", markeredgewidth=1.5, label=f"Peak: {peak_freq_monitor:.2f} Hz")
        )
    _base_axes_style(axes[1], xlabel="Frequency (Hz)", ylabel="Magnitude", hide_ticks=False)
    axes[1].set_xlim(freq_min_plot, freq_max_plot)
    axes[1].xaxis.set_major_locator(MultipleLocator(10))
    axes[1].set_yticks([])  # Remove valores do eixo Y
    axes[1].grid(True, alpha=0.3, linestyle="--")
    if len(legend_handles_monitor) > 0:
        axes[1].legend(handles=legend_handles_monitor, frameon=True, fontsize=LEGEND_FONT_SIZE - 2)
    # Adiciona índice (b) abaixo do subplot (centralizado)
    axes[1].text(0.5, -0.2, "(b)", transform=axes[1].transAxes, fontsize=LABEL_FONT_SIZE, ha='center', va='top')
    
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.4, bottom=0.2)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    print(f"\nLendo arquivo experimental: {tdms_file_path}")
    
    try:
        df_tdms = read_tdms_file(tdms_file_path)
        print(f"Dados TDMS carregados com sucesso!")
        print(f"Número de pontos: {len(df_tdms)}")
        print(f"Colunas disponíveis: {list(df_tdms.columns)}")
        
        if tdms_column not in df_tdms.columns:
            raise ValueError(f"Coluna '{tdms_column}' não encontrada no arquivo TDMS!")
        
    except Exception as e:
        print(f"Erro ao ler arquivo TDMS: {e}")
        raise SystemExit(1)
    
    print(f"\nLendo arquivo de simulação: {monitor_file_path}")
    
    try:
        df_monitor = read_monitor_file(monitor_file_path)
        print(f"Dados de monitoramento carregados com sucesso!")
        print(f"Número de pontos: {len(df_monitor)}")
        
    except Exception as e:
        print(f"Erro ao ler arquivo de monitoramento: {e}")
        raise SystemExit(1)
    
    # Processa dados TDMS
    print(f"\nProcessando dados experimentais...")
    print(f"Coluna: {tdms_column}")
    print(f"Frequência de corte do filtro: {freq_corte} Hz")
    print(f"Faixa de frequência para plotar FFT: {freq_min_plot} - {freq_max_plot} Hz")
    
    freqs_tdms, mag_tdms_orig, mag_tdms_filt = process_tdms_data(
        df_tdms, tdms_column, freq_corte, freq_min_plot, freq_max_plot,
        time_start, time_end, remove_edge_points
    )
    
    # Processa dados do monitor
    print(f"\nProcessando dados de simulação...")
    print(f"Componente: Soma de X + Y + Z (deslocamento)")
    
    freqs_monitor, mag_monitor_orig, mag_monitor_filt = process_monitor_data(
        df_monitor, freq_corte, freq_min_plot, freq_max_plot,
        time_start, time_end, remove_edge_points
    )
    
    # Cria diretório de saída (pasta de execução)
    output_dir = os.getcwd()
    base_name_tdms = os.path.splitext(os.path.basename(tdms_file_path))[0]
    base_name_monitor = os.path.splitext(os.path.basename(monitor_file_path))[0]
    output_path = os.path.join(output_dir, f"combined_fft_{base_name_tdms}_{base_name_monitor}.png")
    
    # Plota FFT combinado
    print(f"\nGerando gráfico combinado...")
    plot_combined_fft(
        freqs_tdms, mag_tdms_orig, mag_tdms_filt,
        freqs_monitor, mag_monitor_orig, mag_monitor_filt,
        freq_corte, freq_min_plot, freq_max_plot,
        output_path
    )
    
    print("\nAnálise concluída!")
    print(f"Gráfico combinado salvo em: {output_path}")

