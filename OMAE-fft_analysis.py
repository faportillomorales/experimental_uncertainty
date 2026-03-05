import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from nptdms import TdmsFile
from scipy import signal


####################################################################################################################################################
#                                            INPUTS
####################################################################################################################################################
file_path = "H:/Meu Drive/LEMI/uncertainties/data_example/example/freq_slug/AOU05P03/AOU05P03_acc.tdms"
freq_corte = 45.0  # Frequência de corte do filtro passa-baixa em Hz - MODIFIQUE ESTE VALOR CONFORME NECESSÁRIO

### Colunas de interesse -> Insira o nome das colunas a plotar e avaliar do arquivo TDMS
# Lista de colunas para análise: [nome_coluna, apelido, unidade]
colunas_analise = [
    ["/'Untitled'/'Accel1'", r"Accel1", r"[g]"],
    ["/'Untitled'/'Accel2'", r"Accel2", r"[g]"],
    ["/'Untitled'/'Accel3'", r"Accel3", r"[g]"],
    ["/'Untitled'/'Accel4'", r"Accel4", r"[g]"],
    ["/'Untitled'/'DP_Validyne'", r"DP\ Validyne", r"[Pa]"],
    ["/'Untitled'/'PIT-M-301'", r"PIT-M-301", r"[Bar]"],
]

####################################################################################################################################################
#                                            END INPUTS
####################################################################################################################################################

LABEL_FONT_SIZE = 16
TICK_FONT_SIZE = 14
LEGEND_FONT_SIZE = 14
LINE_COLOR = "black"
LINEWIDTH = 1.8
FIGSIZE = (6, 6)


def _get_signal_ylabel(coluna: str) -> str:
    lower_name = coluna.lower()
    if "validyne" in lower_name:
        return "Pressure drop"
    if "accel" in lower_name:
        return "Acceleration"
    return coluna.replace("/'Untitled'/", "").replace("'", "").replace("/", "_")


def read_file(file_path: str):
    """
    Lê o arquivo TDMS e retorna um DataFrame do pandas.
    """
    try:
        tdms_file = TdmsFile.read(file_path)
        df = tdms_file.as_dataframe()

        data_teste = None
        try:
            metadata = tdms_file.read_metadata()
            if hasattr(metadata, "description"):
                data_teste = metadata.description
        except Exception:
            pass

        if "/'Untitled'/'Time'" in df.columns:
            time_col = df["/'Untitled'/'Time'"]
            start_time = time_col.iloc[0]
            df["X_Value"] = (time_col - start_time).dt.total_seconds()
        else:
            df["X_Value"] = np.arange(len(df)) * 0.1

        if "J Ar" in df.columns:
            df["J Ar corrigido"] = df["J Ar"] * (1 - 0.06675)
        if "J Agua" in df.columns:
            df["J Agua corrigido"] = df["J Agua"] * (1 - 0.06675)

        return df, data_teste

    except Exception as e:
        print(f"Erro ao ler arquivo TDMS: {e}")
        print("Tentando ler como arquivo de texto...")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="latin-1") as f:
                lines = f.readlines()

        header_count = 0
        header_end_idx = 0
        for i, line in enumerate(lines):
            if "***End_of_Header***" in line:
                header_count += 1
                if header_count == 2:
                    header_end_idx = i + 1
                    break
        if header_end_idx == 0:
            header_end_idx = 5

        column_names = [name.strip() for name in lines[header_end_idx].strip().split("\t")]
        df = pd.read_csv(
            file_path,
            sep="\t",
            skiprows=header_end_idx + 1,
            decimal=",",
            na_values=[""],
            encoding="utf-8",
            names=column_names,
        )

        if "X_Value" not in df.columns:
            df["X_Value"] = np.arange(len(df)) * 0.1

        if "J Ar" in df.columns:
            df["J Ar corrigido"] = df["J Ar"] * (1 - 0.06675)
        if "J Agua" in df.columns:
            df["J Agua corrigido"] = df["J Agua"] * (1 - 0.06675)

        return df, None


def apply_lowpass_filter(data: np.ndarray, fs: float, fc: float, order: int = 4):
    nyquist = fs / 2
    normalized_cutoff = fc / nyquist
    b, a = signal.butter(order, normalized_cutoff, btype="low", analog=False)
    return signal.filtfilt(b, a, data)


def check_required_columns(df: pd.DataFrame, colunas_analise: list):
    colunas_faltantes = []
    colunas_calculadas = ["Alpha", "rho_g", "J Ar corrigido"]

    for coluna_info in colunas_analise:
        nome_coluna = coluna_info[0]
        if nome_coluna in colunas_calculadas:
            continue
        if nome_coluna not in df.columns:
            colunas_faltantes.append(nome_coluna)

    if colunas_faltantes:
        print("ERRO: As seguintes colunas não foram encontradas no arquivo:")
        for coluna in colunas_faltantes:
            print(f"- {coluna}")
        print("Por favor, verifique se o arquivo de entrada está correto.")
        return False

    return True


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


def _plot_time_series(time, original, filtered, coluna_label: str, output_path: str):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(
        time,
        original,
        color="black",
        linestyle="--",
        linewidth=LINEWIDTH,
        alpha=0.5,
        label="Original signal",
    )
    ax.plot(
        time,
        filtered,
        color="black",
        linestyle="-",
        linewidth=LINEWIDTH + 0.4,
        alpha=0.9,
        label="Filtered signal",
    )
    legend_handles = [
        Line2D([], [], color="black", linestyle="--", linewidth=LINEWIDTH, alpha=0.5, label="Original signal"),
        Line2D([], [], color="black", linestyle="-", linewidth=LINEWIDTH + 0.4, alpha=0.9, label="Filtered signal"),
    ]
    ax.legend(handles=legend_handles, frameon=True, fontsize=LEGEND_FONT_SIZE)
    _base_axes_style(ax, xlabel="Time (s)", ylabel=coluna_label, hide_ticks=True)
    # fig.text(
    #     0.5,
    #     0.92,
    #     "Time series",
    #     fontsize=14,
    #     ha="center",
    #     va="center",
    #     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.8),
    # )
    _save_figure(fig, output_path)


def _plot_zoom_series(time, original, filtered, coluna_label: str, output_path: str):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(
        time,
        original,
        color="black",
        linestyle="--",
        linewidth=LINEWIDTH,
        alpha=0.5,
        label="Original",
    )
    ax.plot(
        time,
        filtered,
        color="black",
        linestyle="-",
        linewidth=LINEWIDTH + 0.4,
        alpha=0.9,
        label="Filtered signal",
    )
    legend_handles = [
        Line2D([], [], color="black", linestyle="--", linewidth=LINEWIDTH, alpha=0.5, label="Original"),
        Line2D([], [], color="black", linestyle="-", linewidth=LINEWIDTH + 0.4, alpha=0.9, label="Filtered"),
    ]
    ax.legend(handles=legend_handles, frameon=True, fontsize=LEGEND_FONT_SIZE)
    _base_axes_style(ax, xlabel="Time (s)", ylabel=coluna_label, hide_ticks=True)
    # fig.text(
    #     0.5,
    #     0.23,
    #     "Zoom - first 3 seconds",
    #     fontsize=14,
    #     ha="center",
    #     va="center",
    #     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.8),
    # )
    _save_figure(fig, output_path)


def _plot_filtered_spectrum(freqs, magnitude, freq_corte: float, output_path: str):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    spectrum_line, = ax.plot(freqs, magnitude, color="black", linestyle="-", linewidth=LINEWIDTH)
    ax.axvline(x=freq_corte, color="black", linestyle="--", linewidth=LINEWIDTH - 0.6, alpha=0.8)
    _base_axes_style(ax, xlabel="Frequency (Hz)", ylabel="Magnitude", hide_ticks=True)
    ax.set_xlim(0, freq_corte)
    if len(freqs) > 0:
        peak_idx = int(np.argmax(magnitude))
        peak_freq = float(freqs[peak_idx])
        peak_mag = float(magnitude[peak_idx])
        slug_marker = ax.scatter(
            peak_freq,
            peak_mag,
            color="red",
            s=60,
            zorder=5,
        )
        legend_handles = [
            Line2D([], [], color=spectrum_line.get_color(), linestyle="-", linewidth=LINEWIDTH, label="Filtered spectrum"),
            Line2D([], [], marker="o", color="red", linestyle="None", markersize=8, label="Nominal slug frequency"),
        ]
        ax.legend(handles=legend_handles, frameon=True, fontsize=LEGEND_FONT_SIZE, loc="upper right")
    
    _save_figure(fig, output_path)


def _plot_mosaic_panels(
    time: np.ndarray,
    original: np.ndarray,
    filtered: np.ndarray,
    zoom_time: np.ndarray,
    zoom_original: np.ndarray,
    zoom_filtered: np.ndarray,
    freqs: np.ndarray,
    magnitude: np.ndarray,
    freq_corte: float,
    ylabel: str,
    output_path: str,
    accel1_raw: np.ndarray = None,
) -> None:
    n_panels = 3 if accel1_raw is not None else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(6, 3 * n_panels))
    
    if n_panels == 2:
        axes = [axes[0], axes[1]]
    else:
        axes = [axes[0], axes[1], axes[2]]

    legend_series = [
        Line2D([], [], color="black", linestyle="--", linewidth=LINEWIDTH, alpha=0.5, label="Original signal"),
        Line2D([], [], color="black", linestyle="-", linewidth=LINEWIDTH + 0.4, alpha=0.9, label="Filtered signal"),
    ]
    
    panel_idx = 0
    
    # Plot Accel1 raw data as first panel if available
    if accel1_raw is not None:
        axes[panel_idx].plot(time, accel1_raw, color="black", linestyle="-", linewidth=LINEWIDTH, alpha=0.8)
        _base_axes_style(axes[panel_idx], xlabel="Time (s)", ylabel="Acceleration", hide_ticks=True)
        axes[panel_idx].text(0.5, -0.2, "(a)", transform=axes[panel_idx].transAxes, ha="center", va="center", fontsize=14)
        panel_idx += 1

    axes[panel_idx].plot(time, original, color="black", linestyle="--", linewidth=LINEWIDTH, alpha=0.5)
    axes[panel_idx].plot(time, filtered, color="black", linestyle="-", linewidth=LINEWIDTH + 0.4, alpha=0.9)
    axes[panel_idx].legend(handles=legend_series, frameon=True, fontsize=LEGEND_FONT_SIZE)
    _base_axes_style(axes[panel_idx], xlabel="Time (s)", ylabel=ylabel, hide_ticks=True)
    label_b = "(b)" if accel1_raw is not None else "(a)"
    axes[panel_idx].text(0.5, -0.2, label_b, transform=axes[panel_idx].transAxes, ha="center", va="center", fontsize=14)
    panel_idx += 1

    spectrum_line, = axes[panel_idx].plot(freqs, magnitude, color="black", linestyle="-", linewidth=LINEWIDTH)
    axes[panel_idx].axvline(x=freq_corte, color="black", linestyle="--", linewidth=LINEWIDTH - 0.6, alpha=0.8)
    _base_axes_style(axes[panel_idx], xlabel="Frequency (Hz)", ylabel="Magnitude", hide_ticks=True)
    axes[panel_idx].set_xlim(0, freq_corte)
    if len(freqs) > 0:
        peak_idx = int(np.argmax(magnitude))
        axes[panel_idx].scatter(float(freqs[peak_idx]), float(magnitude[peak_idx]), color="red", s=60, zorder=5)
        legend_handles = [
            Line2D([], [], color=spectrum_line.get_color(), linestyle="-", linewidth=LINEWIDTH, label="Filtered spectrum"),
            Line2D([], [], marker="o", color="red", linestyle="None", markersize=8, label="Nominal slug frequency"),
        ]
        axes[panel_idx].legend(handles=legend_handles, frameon=True, fontsize=LEGEND_FONT_SIZE, loc="upper right")
    label_c = "(c)" if accel1_raw is not None else "(b)"
    axes[panel_idx].text(0.5, -0.2, label_c, transform=axes[panel_idx].transAxes, ha="center", va="center", fontsize=14)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.4)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def apply_fft(df: pd.DataFrame, coluna: str, output_dir: str, base_name: str, freq_corte: float):
    y_data = df[coluna].values
    time = df["X_Value"].values

    dt = np.mean(np.diff(time))
    fs = 1 / dt

    y_filtered = apply_lowpass_filter(y_data, fs, freq_corte)

    n = len(y_data)
    print(f"Número de amostras: {n}")
    fft_result_original = np.fft.fft(y_data)
    freqs = np.fft.fftfreq(n, dt)
    magnitude_original = np.abs(fft_result_original)

    fft_result_filtered = np.fft.fft(y_filtered)
    magnitude_filtered = np.abs(fft_result_filtered)

    positive_freq_mask = freqs > 0
    freqs_pos = freqs[positive_freq_mask]
    mag_orig_pos = magnitude_original[positive_freq_mask]
    mag_filt_pos = magnitude_filtered[positive_freq_mask]

    os.makedirs(output_dir, exist_ok=True)
    coluna_simples = coluna.replace("/'Untitled'/", "").replace("'", "").replace("/", "_")

    signal_ylabel = _get_signal_ylabel(coluna)

    _plot_time_series(
        time,
        y_data,
        y_filtered,
        coluna_label=signal_ylabel,
        output_path=os.path.join(output_dir, f"time_series_{coluna_simples}_{base_name}.png"),
    )

    mask_zoom = time <= 3.0
    _plot_zoom_series(
        time[mask_zoom],
        y_data[mask_zoom],
        y_filtered[mask_zoom],
        coluna_label=signal_ylabel,
        output_path=os.path.join(output_dir, f"time_series_zoom_{coluna_simples}_{base_name}.png"),
    )

    _plot_filtered_spectrum(
        freqs_pos,
        mag_filt_pos,
        freq_corte=freq_corte,
        output_path=os.path.join(output_dir, f"filtered_spectrum_{coluna_simples}_{base_name}.png"),
    )

    # Tenta obter dados brutos do Accel1 se disponível
    accel1_raw = None
    accel1_col = "/'Untitled'/'Accel1'"
    if accel1_col in df.columns and accel1_col != coluna:
        accel1_raw = df[accel1_col].values
    
    _plot_mosaic_panels(
        time=time,
        original=y_data,
        filtered=y_filtered,
        zoom_time=time[mask_zoom],
        zoom_original=y_data[mask_zoom],
        zoom_filtered=y_filtered[mask_zoom],
        freqs=freqs_pos,
        magnitude=mag_filt_pos,
        freq_corte=freq_corte,
        ylabel=signal_ylabel,
        output_path=os.path.join(output_dir, f"mosaic_{coluna_simples}_{base_name}.png"),
        accel1_raw=accel1_raw,
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(freqs_pos, mag_orig_pos, "b-", linewidth=0.8)
    axes[0].axvline(x=freq_corte, color="r", linestyle="--", alpha=0.7, label=f"fc = {freq_corte} Hz")
    axes[0].set_title("Original spectrum")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Magnitude")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_xlim(0, freq_corte)

    axes[1].plot(freqs_pos, mag_orig_pos, "b-", alpha=0.7, label="Original", linewidth=0.8)
    axes[1].plot(freqs_pos, mag_filt_pos, "r-", label="Filtered", linewidth=0.8)
    axes[1].axvline(x=freq_corte, color="g", linestyle="--", alpha=0.7, label=f"fc = {freq_corte} Hz")
    axes[1].set_title("Spectra comparison")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Magnitude")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_xlim(0, freq_corte)

    nyquist = fs / 2
    normalized_cutoff = freq_corte / nyquist
    b, a = signal.butter(4, normalized_cutoff, btype="low", analog=False)
    w, h = signal.freqz(b, a, worN=8000)
    axes[2].plot(0.5 * fs * w / np.pi, np.abs(h), "g-", linewidth=2, label="Filter response")
    axes[2].axvline(x=freq_corte, color="r", linestyle="--", alpha=0.7, label=f"fc = {freq_corte} Hz")
    axes[2].set_title("Butterworth filter response")
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_ylabel("Magnitude")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    axes[2].set_xlim(0, freq_corte)

    if len(freqs_pos) > 0:
        idx_dom_orig = int(np.argmax(mag_orig_pos))
        idx_dom_filt = int(np.argmax(mag_filt_pos))
        f_dom_orig = float(freqs_pos[idx_dom_orig])
        f_dom_filt = float(freqs_pos[idx_dom_filt])

        axes[0].scatter(
            [f_dom_orig],
            [mag_orig_pos[idx_dom_orig]],
            color="black",
            s=50,
            zorder=5,
            label=None,
        )
        axes[1].scatter(
            [f_dom_orig],
            [mag_orig_pos[idx_dom_orig]],
            color="black",
            s=50,
            zorder=5,
            label=None,
        )
        axes[1].scatter(
            [f_dom_filt],
            [mag_filt_pos[idx_dom_filt]],
            color="gray",
            s=50,
            zorder=5,
            label=None,
        )

        print(f"\nFrequência dominante (maior energia) - original: {f_dom_orig:.6f} Hz")
        print(f"Frequência dominante (maior energia) - filtrado: {f_dom_filt:.6f} Hz")

    fig.tight_layout()
    summary_fig_path = os.path.join(output_dir, f"summary_spectra_{coluna_simples}_{base_name}.png")
    fig.savefig(summary_fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fft_data_original = pd.DataFrame(
        {
            "Frequência (Hz)": freqs_pos,
            "Magnitude_Original": mag_orig_pos,
        }
    )
    fft_data_filtered = pd.DataFrame(
        {
            "Frequência (Hz)": freqs_pos,
            "Magnitude_Filtrado": mag_filt_pos,
        }
    )
    fft_data_combined = pd.DataFrame(
        {
            "Frequência (Hz)": freqs_pos,
            "Magnitude_Original": mag_orig_pos,
            "Magnitude_Filtrado": mag_filt_pos,
            "Diferença": mag_orig_pos - mag_filt_pos,
        }
    )

    output_file_original = os.path.join(output_dir, f"fft_original_{coluna_simples}_{base_name}.txt")
    fft_data_original.to_csv(output_file_original, sep="\t", index=False)

    output_file_filtered = os.path.join(
        output_dir, f"fft_filtered_{coluna_simples}_{base_name}_fc{freq_corte}Hz.txt"
    )
    fft_data_filtered.to_csv(output_file_filtered, sep="\t", index=False)

    output_file_combined = os.path.join(
        output_dir, f"fft_comparison_{coluna_simples}_{base_name}_fc{freq_corte}Hz.txt"
    )
    fft_data_combined.to_csv(output_file_combined, sep="\t", index=False)

    return fft_data_combined


def extract_info_from_filename(filename: str) -> Tuple[str, str, str, int, str, bool]:
    fluid_map = {
        "A": "Air",
        "W": "Water",
        "O": "Oil",
        "S": "SF6",
        "D": "Dense Fluid",
    }
    direction_map = {
        "H": "Horizontal",
        "U": "Upward",
        "D": "Downward",
    }
    base_name = os.path.splitext(os.path.basename(filename))[0]

    offset = 1 if base_name[0] == "V" else 0
    is_validation = base_name[0] == "V"
    fluid_1 = fluid_map.get(base_name[0 + offset], "Unknown")
    fluid_2 = fluid_map.get(base_name[1 + offset], "Unknown")
    direction = direction_map.get(base_name[2 + offset], "Unknown")
    theta = int(base_name[3 + offset : 5 + offset])
    ID = base_name[5 + offset :]

    return fluid_1, fluid_2, direction, theta, ID, is_validation


if __name__ == "__main__":
    df, data_teste = read_file(file_path)

    if not check_required_columns(df, colunas_analise):
        raise SystemExit(1)

    output_dir = os.path.dirname(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    fluid_1, fluid_2, direction, theta, ID, is_validation = extract_info_from_filename(file_path)
    print(f"\nInformações extraídas do nome do arquivo:")
    print(f"Fluido 1: {fluid_1}")
    print(f"Fluido 2: {fluid_2}")
    print(f"Direção: {direction}")
    print(f"Inclinação (theta): {theta}°")
    print(f"ID do ponto: {ID}")
    print(f"Ponto de validação: {'Sim' if is_validation else 'Não'}")

    print("\nColunas disponíveis para análise:")
    for i, col in enumerate(colunas_analise, 1):
        print(f"{i}. {col[0]} ({col[1]})")

    while True:
        try:
            escolha = int(input("\nEscolha o número da coluna para análise FFT: "))
            if 1 <= escolha <= len(colunas_analise):
                coluna_escolhida = colunas_analise[escolha - 1][0]
                print(f"\nColuna escolhida: {coluna_escolhida}")
                break
            else:
                print(f"Por favor, escolha um número entre 1 e {len(colunas_analise)}")
        except ValueError:
            print("Entrada inválida. Digite um número válido.")

    print(f"\nFiltro passa-baixa configurado:")
    print(f"Frequência de corte: {freq_corte} Hz")

    dt = np.mean(np.diff(df["X_Value"]))
    fs = 1 / dt
    print(f"Frequência de amostragem: {fs:.2f} Hz")
    print(f"Frequência de Nyquist: {fs/2:.2f} Hz")

    fft_data = apply_fft(df, coluna_escolhida, output_dir, base_name, freq_corte)

    print("\nAnálise FFT com filtro passa-baixa concluída!")
    coluna_simples = coluna_escolhida.replace("/'Untitled'/", "").replace("'", "").replace("/", "_")
    print(f"Gráfico time series salvo em: {os.path.join(output_dir, f'time_series_{coluna_simples}_{base_name}.png')}")
    print(
        f"Gráfico zoom salvo em: {os.path.join(output_dir, f'time_series_zoom_{coluna_simples}_{base_name}.png')}"
    )
    print(
        f"Gráfico filtered spectrum salvo em: {os.path.join(output_dir, f'filtered_spectrum_{coluna_simples}_{base_name}.png')}"
    )
    print(f"Figura mosaico salva em: {os.path.join(output_dir, f'mosaic_{coluna_simples}_{base_name}.png')}")
    print(f"Figura resumo salva em: {os.path.join(output_dir, f'summary_spectra_{coluna_simples}_{base_name}.png')}")
    print(f"Dados originais salvos em: {os.path.join(output_dir, f'fft_original_{coluna_simples}_{base_name}.txt')}")
    print(
        f"Dados filtrados salvos em: {os.path.join(output_dir, f'fft_filtered_{coluna_simples}_{base_name}_fc{freq_corte}Hz.txt')}"
    )
    print(
        f"Dados comparativos salvos em: {os.path.join(output_dir, f'fft_comparison_{coluna_simples}_{base_name}_fc{freq_corte}Hz.txt')}"
    )

