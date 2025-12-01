import os
from typing import Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal


####################################################################################################################################################
#                                            INPUTS
####################################################################################################################################################
# Lista de arquivos para processar - adicione quantos arquivos desejar
file_paths = [
    "G:\Meu Drive/LEMI/uncertainties/data_example/example/OMAE/FRF/FRF_LP_Air_FY_S4Y.txt",
    "G:\Meu Drive/LEMI/uncertainties/data_example/example/OMAE/FRF/FRF_LP_Oil_FY_S4Y.txt",
    "G:\Meu Drive/LEMI/uncertainties/data_example/example/OMAE/FRF/FRF_LP_Water_FY_S4Y.txt",
    # "caminho/para/arquivo2.txt",
    # "caminho/para/arquivo3.txt",
    ]
freq_min = 20.0  # Frequência mínima em Hz - MODIFIQUE ESTE VALOR CONFORME NECESSÁRIO
freq_max = 150.0  # Frequência máxima em Hz - MODIFIQUE ESTE VALOR CONFORME NECESSÁRIO

####################################################################################################################################################
#                                            END INPUTS
####################################################################################################################################################

LABEL_FONT_SIZE = 16
TICK_FONT_SIZE = 14
LEGEND_FONT_SIZE = 14
LINE_COLOR = "black"
LINEWIDTH = 1.8
FIGSIZE = (6, 6)


def read_frf_file(file_path: str) -> Tuple[pd.DataFrame, Optional[dict]]:
    """
    Lê o arquivo FRF do Simcenter Testlab e retorna um DataFrame com os dados.
    
    Formato esperado:
    - Cabeçalho extenso com metadados
    - Linha com cabeçalho: "Hz	g/N	°	Hz	g/N	°"
    - Linha com tipo: "Linear	Log	Phase	Linear	Log	Phase"
    - Dados numéricos a partir da próxima linha
    """
    try:
        # Tenta ler com encoding UTF-8 primeiro
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        # Se falhar, tenta com latin-1
        with open(file_path, "r", encoding="latin-1") as f:
            lines = f.readlines()
    
    # Procura pela linha de cabeçalho dos dados
    header_idx = None
    data_start_idx = None
    
    for i, line in enumerate(lines):
        # Procura pelo padrão do cabeçalho: "Hz	g/N	°" ou variações
        if "Hz" in line and ("g/N" in line or "g/N" in line) and "°" in line:
            # Verifica se tem o formato tabular correto
            if line.count('\t') >= 2:
                header_idx = i
                # A próxima linha tem os tipos (Linear, Log, Phase)
                # E a linha seguinte começa os dados
                if i + 2 < len(lines):
                    data_start_idx = i + 2
                break
    
    if data_start_idx is None:
        raise ValueError("Não foi possível encontrar o início dos dados no arquivo. Verifique se o arquivo está no formato correto.")
    
    # Extrai metadados do cabeçalho (opcional, para referência futura)
    metadata = {}
    for i in range(header_idx):
        line = lines[i].strip()
        if "\t" in line:
            parts = line.split("\t")
            if len(parts) >= 2:
                key = parts[0].strip()
                value = parts[1].strip() if len(parts) > 1 else ""
                if key and value:
                    metadata[key] = value
    
    # Lê os dados numéricos
    data_lines = []
    for i in range(data_start_idx, len(lines)):
        line = lines[i].strip()
        if not line:
            continue
        # Verifica se a linha começa com um número (pode ter sinal + ou -)
        if line[0] in ['+', '-'] or (line[0].isdigit()):
            data_lines.append(line)
    
    # Processa as linhas de dados
    freq1_list = []
    mag1_list = []
    phase1_list = []
    freq2_list = []
    mag2_list = []
    phase2_list = []
    
    for line in data_lines:
        # Substitui vírgula por ponto para conversão numérica
        line_clean = line.replace(',', '.')
        parts = line_clean.split('\t')
        
        if len(parts) >= 6:
            try:
                freq1 = float(parts[0])
                mag1 = float(parts[1])
                phase1 = float(parts[2])
                freq2 = float(parts[3])
                mag2 = float(parts[4])
                phase2 = float(parts[5])
                
                freq1_list.append(freq1)
                mag1_list.append(mag1)
                phase1_list.append(phase1)
                freq2_list.append(freq2)
                mag2_list.append(mag2)
                phase2_list.append(phase2)
            except (ValueError, IndexError):
                continue
    
    # Cria DataFrame
    df = pd.DataFrame({
        'Frequency_1': freq1_list,
        'Magnitude_1': mag1_list,
        'Phase_1': phase1_list,
        'Frequency_2': freq2_list,
        'Magnitude_2': mag2_list,
        'Phase_2': phase2_list,
    })
    
    return df, metadata if metadata else None


def extract_fluid_from_filename(file_path: str) -> str:
    """Extrai o nome do fluido do nome do arquivo."""
    filename = os.path.basename(file_path)
    # Procura por padrões como 'Air', 'Oil', 'Water' no nome do arquivo
    fluids = ['Air', 'Oil', 'Water', 'Gas', 'Liquid']
    for fluid in fluids:
        if fluid in filename:
            return fluid
    # Se não encontrar, retorna "Unknown"
    return "Unknown"


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


def plot_frf_single_axis(
    ax,
    freq1: np.ndarray,
    mag1: np.ndarray,
    freq2: np.ndarray,
    mag2: np.ndarray,
    freq_min: float,
    freq_max: float,
    fluid_label: str = "",
    show_peaks: bool = True,
    show_legend: bool = True,
    show_x_ticks: bool = True,
):
    """Plota as magnitudes das curvas 1 e 2 superpostas em um eixo específico."""
    # Filtra dados entre freq_min e freq_max
    mask1 = (freq1 >= freq_min) & (freq1 <= freq_max)
    mask2 = (freq2 >= freq_min) & (freq2 <= freq_max)
    
    freq1_filtered = freq1[mask1]
    mag1_filtered = mag1[mask1]
    freq2_filtered = freq2[mask2]
    mag2_filtered = mag2[mask2]
    
    # Plota as duas curvas
    ax.semilogy(freq1_filtered, mag1_filtered, color="black", linestyle="-", linewidth=LINEWIDTH, label="Smoothed", alpha=0.8)
    ax.semilogy(freq2_filtered, mag2_filtered, color="gray", linestyle="--", linewidth=LINEWIDTH, label="True FRF", alpha=0.8)
    
    # Detecta e marca picos na Curve 1
    if show_peaks and len(freq1_filtered) > 0:
        max_mag = np.max(mag1_filtered)
        min_height = max_mag * 0.1  # 10% do máximo
        min_distance = max(5, len(freq1_filtered) // 100)  # Pelo menos 5 pontos ou 1% dos dados
        
        peaks_idx, peaks_properties = signal.find_peaks(
            mag1_filtered,
            height=min_height,
            distance=min_distance,
            prominence=max_mag * 0.05,  # Proeminência mínima de 5% do máximo
        )
        
        if len(peaks_idx) > 0:
            peak_freqs = freq1_filtered[peaks_idx]
            peak_mags = mag1_filtered[peaks_idx]
            
            # Adiciona linhas verticais (cursors) nos picos - aumentada a grossura
            for peak_freq, peak_mag in zip(peak_freqs, peak_mags):
                ax.axvline(x=peak_freq, color="red", linestyle=":", linewidth=2.0, alpha=0.7)
    
    _base_axes_style(ax, xlabel="Frequency (Hz)", ylabel="Magnitude (g/N)", hide_ticks=not show_x_ticks)
    ax.set_xlim(freq_min, freq_max)
    ax.grid(True, alpha=0.3, linestyle="--")
    if show_legend:
        ax.legend(frameon=True, fontsize=LEGEND_FONT_SIZE - 2)
    
    # Adiciona label do fluido na parte inferior com folga
    if fluid_label:
        ax.text(0.05, 0.85, f"{fluid_label}", transform=ax.transAxes, 
                ha="left", va="bottom", fontsize=LABEL_FONT_SIZE - 2,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=0.8))


def plot_frf_magnitude_superposed(
    freq1: np.ndarray,
    mag1: np.ndarray,
    freq2: np.ndarray,
    mag2: np.ndarray,
    freq_min: float,
    freq_max: float,
    output_path: str,
    fluid_label: str = "",
):
    """Plota as magnitudes das curvas 1 e 2 superpostas em escala logarítmica com picos marcados."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    plot_frf_single_axis(ax, freq1, mag1, freq2, mag2, freq_min, freq_max, fluid_label=fluid_label, show_peaks=True, show_legend=True, show_x_ticks=True)
    _save_figure(fig, output_path)


def plot_frf_mosaic(
    data_list: list,
    freq_min: float,
    freq_max: float,
    output_path: str,
):
    """
    Cria um mosaico vertical com múltiplos gráficos FRF (um encima do outro).
    
    Args:
        data_list: Lista de tuplas (freq1, mag1, freq2, mag2, title, fluid_label) para cada subplot
        freq_min: Frequência mínima
        freq_max: Frequência máxima
        output_path: Caminho para salvar a figura
    """
    n_files = len(data_list)
    
    # Layout vertical: um subplot encima do outro
    n_rows = n_files
    n_cols = 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6, 4 * n_rows))
    
    # Garante que axes seja sempre um array 1D
    if n_files == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Labels para os subplots: (a), (b), (c), etc.
    labels = [f"({chr(97 + i)})" for i in range(n_files)]  # 97 = 'a' em ASCII
    
    # Plota cada arquivo em um subplot
    for idx, (freq1, mag1, freq2, mag2, title, fluid_label) in enumerate(data_list):
        ax = axes[idx]
        
        # Mostra legenda apenas no último subplot
        show_legend = (idx == n_files - 1)
        plot_frf_single_axis(ax, freq1, mag1, freq2, mag2, freq_min, freq_max, fluid_label=fluid_label, show_peaks=True, show_legend=show_legend, show_x_ticks=True)
        
        # Adiciona label (a), (b), etc. abaixo do subplot
        ax.text(0.5, -0.25, labels[idx], transform=ax.transAxes, ha="center", va="center", fontsize=14)
    
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.4)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def process_frf_data(df: pd.DataFrame, freq_max: float):
    """Processa os dados FRF e retorna arrays limpos."""
    # Extrai dados da Curve 1
    freq1 = df['Frequency_1'].values
    mag1 = df['Magnitude_1'].values
    
    # Extrai dados da Curve 2
    freq2 = df['Frequency_2'].values
    mag2 = df['Magnitude_2'].values
    
    # Remove valores zero ou inválidos
    valid_mask1 = (freq1 > 0) & (mag1 > 0) & np.isfinite(mag1)
    valid_mask2 = (freq2 > 0) & (mag2 > 0) & np.isfinite(mag2)
    
    freq1_clean = freq1[valid_mask1]
    mag1_clean = mag1[valid_mask1]
    
    freq2_clean = freq2[valid_mask2]
    mag2_clean = mag2[valid_mask2]
    
    return freq1_clean, mag1_clean, freq2_clean, mag2_clean


if __name__ == "__main__":
    # Garante que file_paths seja uma lista
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    
    print(f"\nProcessando {len(file_paths)} arquivo(s)...")
    print(f"Faixa de frequência: {freq_min} - {freq_max} Hz")
    
    data_list = []
    output_dir = None
    
    # Processa cada arquivo
    for file_path in file_paths:
        print(f"\nLendo arquivo: {file_path}")
        
        try:
            df, metadata = read_frf_file(file_path)
            print(f"  Dados carregados com sucesso! ({len(df)} pontos)")
            
            # Processa os dados
            freq1, mag1, freq2, mag2 = process_frf_data(df, freq_max)
            
            # Extrai título do nome do arquivo
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            title = base_name
            
            # Extrai fluido do nome do arquivo
            fluid_label = extract_fluid_from_filename(file_path)
            
            data_list.append((freq1, mag1, freq2, mag2, title, fluid_label))
            
            # Define diretório de saída (usa o primeiro arquivo como referência)
            if output_dir is None:
                output_dir = os.path.dirname(file_path)
            
            # Detecta e imprime picos
            if len(freq1) > 0:
                max_mag = np.max(mag1)
                min_height = max_mag * 0.1
                min_distance = max(5, len(freq1) // 100)
                
                peaks_idx, _ = signal.find_peaks(
                    mag1,
                    height=min_height,
                    distance=min_distance,
                    prominence=max_mag * 0.05,
                )
                
                if len(peaks_idx) > 0:
                    peak_freqs = freq1[peaks_idx]
                    peak_mags = mag1[peaks_idx]
                    print(f"  Picos detectados na Curve 1 ({len(peak_freqs)} picos):")
                    for i, (peak_freq, peak_mag) in enumerate(zip(peak_freqs, peak_mags), 1):
                        print(f"    Pico {i}: {peak_freq:.2f} Hz (Magnitude: {peak_mag:.6e} g/N)")
            
        except Exception as e:
            print(f"  ERRO ao ler arquivo {file_path}: {e}")
            continue
    
    if len(data_list) == 0:
        print("\nNenhum arquivo foi processado com sucesso!")
        raise SystemExit(1)
    
    # Cria mosaico
    if output_dir is None:
        output_dir = os.getcwd()
    
    mosaic_path = os.path.join(output_dir, f"frf_mosaic_{len(data_list)}files.png")
    print(f"\nCriando mosaico com {len(data_list)} arquivo(s)...")
    plot_frf_mosaic(data_list, freq_min, freq_max, mosaic_path)
    
    print("\nAnálise FRF concluída!")
    print(f"Mosaico salvo em: {mosaic_path}")
    
    # Se houver apenas um arquivo, também salva gráfico individual
    if len(data_list) == 1:
        freq1, mag1, freq2, mag2, title, fluid_label = data_list[0]
        individual_path = os.path.join(output_dir, f"frf_magnitude_superposed_{title}.png")
        plot_frf_magnitude_superposed(freq1, mag1, freq2, mag2, freq_min, freq_max, individual_path, fluid_label=fluid_label)
        print(f"Gráfico individual salvo em: {individual_path}")

