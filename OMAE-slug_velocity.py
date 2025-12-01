import os
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from nptdms import TdmsFile
from scipy import signal


####################################################################################################################################################
#                                            INPUTS
####################################################################################################################################################
file_path = "G:/Meu Drive/LEMI/uncertainties/data_example/example/freq_slug/AOH00P02/AOH00P02_acc.tdms"

### Colunas de interesse -> Insira o nome das colunas a plotar e avaliar do arquivo TDMS
# Lista de colunas para análise: [nome_coluna, apelido, unidade]
colunas_analise = [
    ["/'Untitled'/'Accel1'", r"Accel1", r"[g]"],
    ["/'Untitled'/'Accel2'", r"Accel2", r"[g]"],
    ["/'Untitled'/'Accel3'", r"Accel3", r"[g]"],
    ["/'Untitled'/'Accel4'", r"Accel4", r"[g]"],
]

# Parâmetros de filtragem
freq_corte = None  # Frequência de corte do filtro passa-baixa em Hz - MODIFIQUE ESTE VALOR CONFORME NECESSÁRIO

# Parâmetros de janelamento temporal
time_min = 7  # Tempo mínimo da janela em segundos (None = início dos dados) - MODIFIQUE ESTE VALOR CONFORME NECESSÁRIO
time_max = 22  # Tempo máximo da janela em segundos (None = fim dos dados) - MODIFIQUE ESTE VALOR CONFORME NECESSÁRIO

# Parâmetros de detecção de picos
threshold_factor = 2.5  # Fator de threshold relativo ao desvio padrão do sinal (apenas picos acima deste valor serão considerados)
prominence_factor = 0.8  # Fator de proeminência relativa ao desvio padrão do sinal
distance_min_samples = None  # Distância mínima entre picos em amostras (None = automático)
ignore_initial_seconds = 1.0  # Ignora picos detectados nos primeiros N segundos (para evitar transientes do filtro) - MODIFIQUE ESTE VALOR CONFORME NECESSÁRIO

# Parâmetros de validação de velocidades
min_delta_t_seconds = 0.04  # Limite mínimo de delta_t em segundos (intervalos menores que este valor serão rejeitados) - MODIFIQUE ESTE VALOR CONFORME NECESSÁRIO

####################################################################################################################################################
#                                            END INPUTS
####################################################################################################################################################

LABEL_FONT_SIZE = 16
TICK_FONT_SIZE = 14
LEGEND_FONT_SIZE = 14
LINE_COLOR = "black"
LINEWIDTH = 1.8
FIGSIZE = (6, 6)


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
    """
    Aplica filtro passa-baixa Butterworth nos dados.
    Se fc for None, retorna os dados originais sem filtrar.
    """
    if fc is None:
        return data
    
    nyquist = fs / 2
    normalized_cutoff = fc / nyquist
    b, a = signal.butter(order, normalized_cutoff, btype="low", analog=False)
    return signal.filtfilt(b, a, data)


def apply_time_window(
    time: np.ndarray,
    signal1: np.ndarray,
    signal2: np.ndarray,
    time_min: Optional[float],
    time_max: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aplica janelamento temporal nos dados.
    
    Args:
        time: Array de tempos
        signal1: Array do primeiro sinal
        signal2: Array do segundo sinal
        time_min: Tempo mínimo da janela (None = início)
        time_max: Tempo máximo da janela (None = fim)
    
    Returns:
        Tupla com (time_windowed, signal1_windowed, signal2_windowed)
    """
    # Cria máscara para janelamento
    window_mask = np.ones(len(time), dtype=bool)
    
    if time_min is not None:
        window_mask = window_mask & (time >= time_min)
    
    if time_max is not None:
        window_mask = window_mask & (time <= time_max)
    
    time_windowed = time[window_mask]
    signal1_windowed = signal1[window_mask]
    signal2_windowed = signal2[window_mask]
    
    return time_windowed, signal1_windowed, signal2_windowed


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


def plot_raw_signals(
    time: np.ndarray,
    signal1: np.ndarray,
    signal2: np.ndarray,
    coluna1: str,
    coluna2: str,
    output_path: str,
):
    """
    Plota os sinais brutos dos dois acelerômetros logo após a leitura.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Simplifica nomes das colunas
    label1 = coluna1.replace("/'Untitled'/", "").replace("'", "").replace("/", "_")
    label2 = coluna2.replace("/'Untitled'/", "").replace("'", "").replace("/", "_")
    
    # Plota primeiro acelerômetro
    axes[0].plot(time, signal1, color="black", linestyle="-", linewidth=LINEWIDTH, alpha=0.8)
    _base_axes_style(axes[0], xlabel="Time (s)", ylabel="Acceleration", hide_ticks=True)
    axes[0].text(0.02, 0.95, label1, transform=axes[0].transAxes, fontsize=12, 
                verticalalignment="top", bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                edgecolor="black", alpha=0.8))
    
    # Plota segundo acelerômetro
    axes[1].plot(time, signal2, color="gray", linestyle="-", linewidth=LINEWIDTH, alpha=0.8)
    _base_axes_style(axes[1], xlabel="Time (s)", ylabel="Acceleration", hide_ticks=True)
    axes[1].text(0.02, 0.95, label2, transform=axes[1].transAxes, fontsize=12, 
                verticalalignment="top", bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                edgecolor="black", alpha=0.8))
    
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.3)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    # plt.show()
    plt.close(fig)


def detect_peaks_in_signal(
    time: np.ndarray, 
    signal_data: np.ndarray, 
    threshold_factor: float = 2.0,
    prominence_factor: float = 0.3, 
    distance_min: int = None,
    ignore_initial_seconds: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detecta picos no sinal usando scipy.signal.find_peaks com threshold.
    
    Args:
        time: Array de tempo
        signal_data: Array de dados do sinal
        threshold_factor: Fator de threshold relativo ao desvio padrão (apenas picos acima deste valor)
        prominence_factor: Fator de proeminência relativa ao desvio padrão
        distance_min: Distância mínima entre picos em amostras (None = automático)
        ignore_initial_seconds: Ignora picos nos primeiros N segundos (para evitar transientes)
    
    Returns:
        Tuple de (índices dos picos, tempos dos picos)
    """
    # Remove NaNs
    valid_mask = ~np.isnan(signal_data)
    if not np.any(valid_mask):
        return np.array([]), np.array([])
    
    signal_clean = signal_data[valid_mask]
    time_clean = time[valid_mask]
    
    # Calcula threshold baseado no desvio padrão
    std_signal = np.std(signal_clean)
    mean_signal = np.mean(signal_clean)
    # Threshold absoluto: média + threshold_factor * desvio padrão
    threshold_height = mean_signal + threshold_factor * std_signal
    
    # Calcula proeminência baseada no desvio padrão
    prominence = prominence_factor * std_signal
    
    # Calcula distância mínima se não fornecida (baseada na frequência de amostragem)
    if distance_min is None:
        dt = np.mean(np.diff(time_clean))
        fs = 1.0 / dt
        # Distância mínima de ~0.5 segundos
        distance_min = int(0.5 * fs)
    
    # Detecta picos com threshold e proeminência
    peak_indices, properties = signal.find_peaks(
        signal_clean, 
        height=threshold_height,
        prominence=prominence, 
        distance=distance_min
    )
    
    # Converte índices do sinal limpo para índices do sinal original
    valid_indices = np.where(valid_mask)[0]
    original_peak_indices = valid_indices[peak_indices]
    peak_times = time[original_peak_indices]
    
    # Filtra picos muito próximos do início (transientes do filtro)
    if ignore_initial_seconds > 0 and len(peak_times) > 0:
        mask_valid_time = peak_times >= ignore_initial_seconds
        original_peak_indices = original_peak_indices[mask_valid_time]
        peak_times = peak_times[mask_valid_time]
    
    return original_peak_indices, peak_times


def match_peaks_between_signals(
    time1: np.ndarray, peaks1: np.ndarray, time2: np.ndarray, peaks2: np.ndarray
) -> List[Tuple[int, int, float]]:
    """
    Faz correspondência entre picos de dois sinais baseado na ordem de aparição sequencial.
    O primeiro pico de acc1 corresponde ao primeiro pico de acc2, segundo com segundo, etc.
    
    Args:
        time1: Array de tempo do primeiro sinal
        peaks1: Índices dos picos no primeiro sinal
        time2: Array de tempo do segundo sinal
        peaks2: Índices dos picos no segundo sinal
    
    Returns:
        Lista de tuplas (índice_pico1, índice_pico2, delta_tempo)
    """
    matches = []
    
    if len(peaks1) == 0 or len(peaks2) == 0:
        return matches
    
    # Obtém tempos dos picos
    peak_times1 = time1[peaks1]
    peak_times2 = time2[peaks2]
    
    # Ordena índices por tempo (retorna índices que ordenam os arrays)
    sorted_order1 = np.argsort(peak_times1)
    sorted_order2 = np.argsort(peak_times2)
    
    # Picos ordenados por tempo (usando índices ordenados)
    sorted_peaks1 = peaks1[sorted_order1]
    sorted_peaks2 = peaks2[sorted_order2]
    sorted_times1 = peak_times1[sorted_order1]
    sorted_times2 = peak_times2[sorted_order2]
    
    # Faz correspondência sequencial: primeiro com primeiro, segundo com segundo, etc.
    # Usa o menor número de picos entre os dois sinais
    n_matches = min(len(sorted_peaks1), len(sorted_peaks2))
    
    for i in range(n_matches):
        # Índices originais dos picos (peaks1 e peaks2 já contêm os índices no array completo)
        # sorted_peaks1[i] é o índice no array de tempo original para o i-ésimo pico ordenado
        idx1_original = sorted_peaks1[i]
        idx2_original = sorted_peaks2[i]
        
        # Tempos dos picos correspondentes
        t1 = sorted_times1[i]
        t2 = sorted_times2[i]
        
        # Calcula intervalo de tempo (pode ser positivo ou negativo dependendo da ordem)
        delta_t = t2 - t1
        
        # Usa valor absoluto para calcular velocidade (independente da ordem dos acelerômetros)
        # Limite máximo razoável: 30 segundos (ajustável conforme necessário)
        # Para velocidades típicas de slugs (0.1-5 m/s) e distâncias de 0.1-1 m, 
        # intervalos de até 10s são esperados, mas usamos 30s para ser mais flexível
        if abs(delta_t) < 10.0:
            # Usa as posições originais nos arrays peaks1 e peaks2
            pos1_in_peaks = np.where(peaks1 == idx1_original)[0][0]
            pos2_in_peaks = np.where(peaks2 == idx2_original)[0][0]
            # Armazena o valor absoluto do delta_t para cálculo de velocidade
            matches.append((pos1_in_peaks, pos2_in_peaks, abs(delta_t)))
    
    return matches


def calculate_velocities(delta_times: np.ndarray, distance: float) -> np.ndarray:
    """
    Calcula velocidades a partir dos intervalos de tempo e distância.
    
    Args:
        delta_times: Array de intervalos de tempo em segundos
        distance: Distância entre acelerômetros em metros
    
    Returns:
        Array de velocidades em m/s
    """
    # Evita divisão por zero
    with np.errstate(divide="ignore", invalid="ignore"):
        velocities = distance / delta_times
    return velocities


def plot_slug_velocity_analysis(
    time: np.ndarray,
    signals: List[np.ndarray],
    signal_names: List[str],
    peaks_list: List[np.ndarray],
    matches: List[Tuple[int, int, float]],
    velocities: np.ndarray,
    distance: float,
    output_path: str,
    ignore_initial_seconds: float = 0.0,
):
    """
    Plota os sinais dos acelerômetros com picos marcados e mostra velocidades calculadas.
    """
    n_signals = len(signals)
    fig, axes = plt.subplots(n_signals, 1, figsize=(10, 5 * n_signals))
    
    if n_signals == 1:
        axes = [axes]
    
    colors = ["black", "gray"]
    
    # Filtra dados para remover os segundos iniciais ignorados
    if ignore_initial_seconds > 0:
        time_mask = time >= ignore_initial_seconds
        time_plot = time[time_mask]
        signals_plot = [sig[time_mask] for sig in signals]
    else:
        time_plot = time
        signals_plot = signals
    
    # Plota sinais e picos
    for i, (ax, signal_data_plot, signal_name, peaks) in enumerate(zip(axes, signals_plot, signal_names, peaks_list)):
        # Plota sinal filtrado
        ax.plot(time_plot, signal_data_plot, color=colors[i % len(colors)], linestyle="-", linewidth=LINEWIDTH, alpha=0.8)
        
        # Marca picos (apenas os que estão no intervalo plotado)
        if len(peaks) > 0:
            peak_times = time[peaks]
            # Usa o sinal original para obter os valores dos picos
            signal_original = signals[i]
            peak_values = signal_original[peaks]
            
            if ignore_initial_seconds > 0:
                # Filtra picos que estão no intervalo plotado
                peak_mask = peak_times >= ignore_initial_seconds
                peak_times_plot = peak_times[peak_mask]
                peak_values_plot = peak_values[peak_mask]
            else:
                peak_times_plot = peak_times
                peak_values_plot = peak_values
            
            if len(peak_times_plot) > 0:
                ax.scatter(peak_times_plot, peak_values_plot, color="red", s=100, zorder=5, marker="o", 
                          edgecolors="black", linewidths=1.5, label="Peaks")
        
        # Estilo do eixo
        signal_label = signal_name.replace("/'Untitled'/", "").replace("'", "").replace("/", "_")
        _base_axes_style(ax, xlabel="Time (s)", ylabel="Acceleration", hide_ticks=True)
        
        if len(peaks) > 0:
            ax.legend(frameon=True, fontsize=LEGEND_FONT_SIZE, loc="upper right")
        
        # Adiciona nome do acelerômetro
        ax.text(0.02, 0.95, signal_label, transform=ax.transAxes, fontsize=12, 
                verticalalignment="top", bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                edgecolor="black", alpha=0.8))
    
    # Conecta picos correspondentes com linhas verticais e mostra informações
    if len(matches) > 0 and len(velocities) > 0 and n_signals >= 2:
        peaks1 = peaks_list[0]
        peaks2 = peaks_list[1]
        
        # Limita número de anotações para não poluir o gráfico
        max_annotations = min(10, len(matches))
        annotation_indices = np.linspace(0, len(matches) - 1, max_annotations, dtype=int)
        
        for match_idx, (idx1, idx2, delta_t) in enumerate(matches):
            if match_idx < len(velocities) and velocities[match_idx] > 0:
                peak1_time = time[peaks1[idx1]]
                peak2_time = time[peaks2[idx2]]
                
                # Apenas desenha linhas e anotações se os picos estão no intervalo plotado
                if ignore_initial_seconds > 0:
                    if peak1_time < ignore_initial_seconds or peak2_time < ignore_initial_seconds:
                        continue
                
                # Desenha linhas verticais nos tempos dos picos
                axes[0].axvline(x=peak1_time, color="blue", linestyle="--", linewidth=1.0, alpha=0.4)
                axes[1].axvline(x=peak2_time, color="blue", linestyle="--", linewidth=1.0, alpha=0.4)
                
                # Adiciona anotação com velocidade (apenas para alguns picos para não poluir)
                if match_idx in annotation_indices:
                    mid_x = (peak1_time + peak2_time) / 2
                    y_min, y_max = axes[1].get_ylim()
                    
                    # Alterna posição vertical: pares na parte superior, ímpares na inferior
                    annotation_position_idx = np.where(annotation_indices == match_idx)[0][0]
                    if annotation_position_idx % 2 == 0:
                        # Posição superior
                        y_annot = y_max * 0.85
                        va_align = "bottom"
                    else:
                        # Posição inferior
                        y_annot = y_min + (y_max - y_min) * 0.15
                        va_align = "top"
                    
                    axes[1].annotate(
                        f"Δt={delta_t:.3f}s\nv={velocities[match_idx]:.2f}m/s",
                        xy=(mid_x, y_annot),
                        xytext=(mid_x, y_annot),
                        fontsize=8,
                        ha="center",
                        va=va_align,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", 
                                 edgecolor="black", alpha=0.7),
                    )
    
    # Adiciona informações de velocidade abaixo do último gráfico
    if len(matches) > 0 and len(velocities) > 0:
        valid_velocities = velocities[velocities > 0]
        if len(valid_velocities) > 0:
            info_text = f"Distance between accelerometers: {distance:.3f} m\n"
            info_text += f"Number of matched peaks: {len(matches)}\n"
            info_text += f"Average velocity: {np.mean(valid_velocities):.3f} m/s\n"
            info_text += f"Std velocity: {np.std(valid_velocities):.3f} m/s\n"
            info_text += f"Min velocity: {np.min(valid_velocities):.3f} m/s\n"
            info_text += f"Max velocity: {np.max(valid_velocities):.3f} m/s"
            
            axes[-1].text(0.5, -0.12, info_text, transform=axes[-1].transAxes, 
                         ha="center", va="top", fontsize=11,
                         bbox=dict(boxstyle="round,pad=0.5", facecolor="white", 
                                  edgecolor="black", alpha=0.9))
    
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.4)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    # Lê o arquivo
    df, data_teste = read_file(file_path)
    
    # Verifica colunas
    if not check_required_columns(df, colunas_analise):
        raise SystemExit(1)
    
    output_dir = os.path.dirname(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Mostra colunas disponíveis
    print("\nAcelerômetros disponíveis para análise:")
    accel_cols = [col for col in colunas_analise if "accel" in col[0].lower()]
    for i, col in enumerate(accel_cols, 1):
        print(f"{i}. {col[0]} ({col[1]})")
    
    if len(accel_cols) < 2:
        print("ERRO: É necessário pelo menos 2 acelerômetros para calcular velocidade.")
        raise SystemExit(1)
    
    # Seleciona dois acelerômetros
    print("\nSelecione dois acelerômetros para análise:")
    while True:
        try:
            escolha1 = int(input(f"Primeiro acelerômetro (1-{len(accel_cols)}): "))
            if 1 <= escolha1 <= len(accel_cols):
                coluna1 = accel_cols[escolha1 - 1][0]
                break
            else:
                print(f"Por favor, escolha um número entre 1 e {len(accel_cols)}")
        except ValueError:
            print("Entrada inválida. Digite um número válido.")
    
    while True:
        try:
            escolha2 = int(input(f"Segundo acelerômetro (1-{len(accel_cols)}): "))
            if 1 <= escolha2 <= len(accel_cols):
                if escolha2 == escolha1:
                    print("Por favor, escolha um acelerômetro diferente do primeiro.")
                    continue
                coluna2 = accel_cols[escolha2 - 1][0]
                break
            else:
                print(f"Por favor, escolha um número entre 1 e {len(accel_cols)}")
        except ValueError:
            print("Entrada inválida. Digite um número válido.")
    
    print(f"\nAcelerômetros selecionados:")
    print(f"1. {coluna1}")
    print(f"2. {coluna2}")
    
    # Solicita distância entre acelerômetros
    while True:
        try:
            distance = float(input("\nDigite a distância entre os acelerômetros (em metros): "))
            if distance > 0:
                break
            else:
                print("A distância deve ser maior que zero.")
        except ValueError:
            print("Entrada inválida. Digite um número válido.")
    
    # Obtém dados
    time = df["X_Value"].values
    signal1 = df[coluna1].values
    signal2 = df[coluna2].values
    
    # Aplica janelamento temporal se especificado
    if time_min is not None or time_max is not None:
        print(f"\nAplicando janela de tempo: {time_min if time_min is not None else 'início'} - {time_max if time_max is not None else 'fim'} s")
        time, signal1, signal2 = apply_time_window(time, signal1, signal2, time_min, time_max)
        print(f"Janela aplicada: {len(time)} pontos (de {time[0]:.2f} s a {time[-1]:.2f} s)")
    else:
        print(f"\nNenhuma janela de tempo especificada. Usando todos os dados: {len(time)} pontos (de {time[0]:.2f} s a {time[-1]:.2f} s)")
    
    # Calcula frequência de amostragem
    dt = np.mean(np.diff(time))
    fs = 1 / dt
    
    print(f"\nFrequência de amostragem: {fs:.2f} Hz")
    print(f"Frequência de corte do filtro: {freq_corte} Hz")
    print(f"Threshold factor: {threshold_factor} (apenas picos acima de média + {threshold_factor}*std serão considerados)")
    
    # Plota sinais brutos logo após a leitura
    print("\nPlotando sinais brutos dos acelerômetros...")
    coluna1_simples = coluna1.replace("/'Untitled'/", "").replace("'", "").replace("/", "_")
    coluna2_simples = coluna2.replace("/'Untitled'/", "").replace("'", "").replace("/", "_")
    raw_signals_path = os.path.join(output_dir, f"raw_signals_{coluna1_simples}_vs_{coluna2_simples}_{base_name}.png")
    plot_raw_signals(time, signal1, signal2, coluna1, coluna2, raw_signals_path)
    print(f"Sinais brutos salvos em: {raw_signals_path}")
    
    # Aplica filtro passa-baixa nos sinais
    if freq_corte is None:
        print(f"\nNenhum filtro será aplicado (freq_corte = None)")
    else:
        print(f"\nAplicando filtro passa-baixa (fc={freq_corte} Hz)...")
    signal1_filtered = apply_lowpass_filter(signal1, fs, freq_corte)
    signal2_filtered = apply_lowpass_filter(signal2, fs, freq_corte)
    
    # Detecta picos nos sinais filtrados
    print(f"Detectando picos nos sinais filtrados (ignorando primeiros {ignore_initial_seconds} segundos)...")
    peaks1_idx, peaks1_times = detect_peaks_in_signal(
        time, signal1_filtered, threshold_factor, prominence_factor, distance_min_samples, ignore_initial_seconds
    )
    peaks2_idx, peaks2_times = detect_peaks_in_signal(
        time, signal2_filtered, threshold_factor, prominence_factor, distance_min_samples, ignore_initial_seconds
    )
    
    print(f"Picos detectados no primeiro acelerômetro: {len(peaks1_idx)}")
    print(f"Picos detectados no segundo acelerômetro: {len(peaks2_idx)}")
    
    # Mostra informações de debug sobre os primeiros picos
    if len(peaks1_idx) > 0 and len(peaks2_idx) > 0:
        peak_times1_sorted = np.sort(time[peaks1_idx])
        peak_times2_sorted = np.sort(time[peaks2_idx])
        print(f"\nPrimeiros 3 picos (ordenados por tempo):")
        print(f"  Accel1: {peak_times1_sorted[:min(3, len(peak_times1_sorted))]}")
        print(f"  Accel2: {peak_times2_sorted[:min(3, len(peak_times2_sorted))]}")
        if len(peak_times1_sorted) > 0 and len(peak_times2_sorted) > 0:
            first_delta = peak_times2_sorted[0] - peak_times1_sorted[0]
            print(f"  Delta t (primeiro par): {first_delta:.3f} s")
    
    # Faz correspondência sequencial entre picos (primeiro com primeiro, segundo com segundo, etc.)
    print("\nFazendo correspondência sequencial de picos...")
    matches = match_peaks_between_signals(time, peaks1_idx, time, peaks2_idx)
    print(f"Picos correspondentes encontrados: {len(matches)} (de {min(len(peaks1_idx), len(peaks2_idx))} possíveis)")
    
    if len(matches) == 0:
        print("\nERRO: Não foi possível encontrar picos correspondentes entre os dois sinais.")
        print("Possíveis causas:")
        print("  1. Intervalos de tempo muito grandes (> 30 s) entre picos correspondentes")
        print("  2. Os dois acelerômetros têm número muito diferente de picos")
        print("  3. Os picos não estão na mesma ordem temporal")
        if len(peaks1_idx) > 0 and len(peaks2_idx) > 0:
            peak_times1_sorted = np.sort(time[peaks1_idx])
            peak_times2_sorted = np.sort(time[peaks2_idx])
            n_check = min(5, len(peak_times1_sorted), len(peak_times2_sorted))
            print(f"\nVerificando primeiros {n_check} pares:")
            for i in range(n_check):
                dt = peak_times2_sorted[i] - peak_times1_sorted[i]
                print(f"  Par {i+1}: t1={peak_times1_sorted[i]:.3f}s, t2={peak_times2_sorted[i]:.3f}s, Δt={dt:.3f}s")
        raise SystemExit(1)
    
    if len(peaks1_idx) != len(peaks2_idx):
        print(f"AVISO: Número diferente de picos detectados!")
        print(f"  Accel1: {len(peaks1_idx)} picos")
        print(f"  Accel2: {len(peaks2_idx)} picos")
        print(f"  Usando apenas os primeiros {len(matches)} pares correspondentes")
    
    # Calcula intervalos de tempo
    delta_times = np.array([match[2] for match in matches])
    
    # Debug: mostra primeiros intervalos de tempo calculados
    print(f"\n=== DEBUG: Intervalos de Tempo Calculados ===")
    print(f"Primeiros 5 intervalos de tempo (delta_t):")
    for i in range(min(5, len(delta_times))):
        idx1, idx2, dt = matches[i]
        t1 = time[peaks1_idx[idx1]]
        t2 = time[peaks2_idx[idx2]]
        print(f"  Par {i+1}: t1={t1:.6f}s, t2={t2:.6f}s, delta_t={dt:.6f}s")
    print(f"Delta_t mínimo: {np.min(delta_times):.6f} s")
    print(f"Delta_t máximo: {np.max(delta_times):.6f} s")
    print(f"Delta_t médio: {np.mean(delta_times):.6f} s")
    
    # Verifica se há intervalos muito pequenos (possível erro)
    very_small_dt = delta_times < min_delta_t_seconds  # Usa parâmetro configurado pelo usuário
    if np.any(very_small_dt):
        n_small = np.sum(very_small_dt)
        print(f"\n⚠️  AVISO: {n_small} pares têm delta_t < {min_delta_t_seconds*1000:.1f}ms (muito pequeno, possível erro de correspondência)")
        print(f"   Isso resultaria em velocidades > {distance / min_delta_t_seconds:.1f} m/s")
    
    # Validação: remove delta_t muito pequenos antes de calcular velocidade
    # Delta_t mínimo físico baseado na distância e velocidade máxima razoável
    # Se a distância é, por exemplo, 0.1m e a velocidade máxima é 10 m/s, delta_t mínimo seria 0.01s
    min_delta_t = distance / 100.0  # Assume velocidade máxima de 100 m/s (muito alta, mas possível)
    if min_delta_t < min_delta_t_seconds:  # Mas não menor que o limite configurado pelo usuário
        min_delta_t = min_delta_t_seconds
    
    print(f"\nValidação de delta_t:")
    print(f"  Delta_t mínimo aceitável: {min_delta_t:.6f} s (baseado em distância={distance:.3f}m e v_max=100 m/s)")
    
    valid_dt_mask = delta_times >= min_delta_t
    n_invalid = np.sum(~valid_dt_mask)
    if n_invalid > 0:
        print(f"  ⚠️  {n_invalid} pares têm delta_t < {min_delta_t:.6f}s e serão REJEITADOS")
        print(f"  Delta_t rejeitados: {delta_times[~valid_dt_mask]}")
    
    # Calcula velocidades apenas com delta_t válidos
    delta_times_valid = delta_times[valid_dt_mask]
    velocities = calculate_velocities(delta_times_valid, distance)
    
    # Remove velocidades inválidas (infinito ou NaN)
    valid_velocity_mask = np.isfinite(velocities) & (velocities > 0) & (velocities < 100.0)  # Limita velocidade máxima
    velocities_valid = velocities[valid_velocity_mask]
    delta_times_final = delta_times_valid[valid_velocity_mask]
    
    # Reconstroi valid_mask completo (incluindo os rejeitados na primeira etapa)
    valid_mask = np.zeros(len(matches), dtype=bool)
    valid_dt_indices = np.where(valid_dt_mask)[0]  # Índices dos matches que passaram validação de delta_t
    valid_mask[valid_dt_indices[valid_velocity_mask]] = True  # Marca apenas os que passaram ambas validações
    
    if len(velocities_valid) == 0:
        print("\nERRO: Não foi possível calcular velocidades válidas.")
        print("Possíveis causas:")
        print("  1. Todos os delta_t são muito pequenos (picos muito próximos)")
        print("  2. Erro na correspondência de picos")
        print("  3. Os picos não são realmente correspondentes")
        raise SystemExit(1)
    
    # Estatísticas
    print(f"\n=== RESULTADOS ===")
    print(f"Distância entre acelerômetros: {distance:.3f} m")
    print(f"Pares rejeitados (delta_t muito pequeno): {n_invalid}")
    print(f"Número de medidas válidas: {len(velocities_valid)}")
    print(f"Velocidade média: {np.mean(velocities_valid):.3f} m/s")
    print(f"Desvio padrão: {np.std(velocities_valid):.3f} m/s")
    print(f"Velocidade mínima: {np.min(velocities_valid):.3f} m/s")
    print(f"Velocidade máxima: {np.max(velocities_valid):.3f} m/s")
    
    # Salva resultados em arquivo
    valid_matches_indices = np.where(valid_mask)[0]
    results_df = pd.DataFrame({
        "Peak1_Index": [matches[i][0] for i in valid_matches_indices],
        "Peak2_Index": [matches[i][1] for i in valid_matches_indices],
        "Delta_Time_s": delta_times_final,
        "Velocity_m_s": velocities_valid,
    })
    
    coluna1_simples = coluna1.replace("/'Untitled'/", "").replace("'", "").replace("/", "_")
    coluna2_simples = coluna2.replace("/'Untitled'/", "").replace("'", "").replace("/", "_")
    
    results_path = os.path.join(output_dir, f"slug_velocity_{coluna1_simples}_vs_{coluna2_simples}_{base_name}.txt")
    results_df.to_csv(results_path, sep="\t", index=False)
    print(f"\nResultados salvos em: {results_path}")
    
    # Plota resultados
    plot_path = os.path.join(output_dir, f"slug_velocity_plot_{coluna1_simples}_vs_{coluna2_simples}_{base_name}.png")
    
    # Cria lista de sinais e picos para plotagem (usa sinais filtrados)
    signals_plot = [signal1_filtered, signal2_filtered]
    signal_names_plot = [coluna1, coluna2]
    peaks_list_plot = [peaks1_idx, peaks2_idx]
    
    # Filtra matches válidos para plotagem
    valid_matches = [matches[i] for i in valid_matches_indices]
    
    plot_slug_velocity_analysis(
        time=time,
        signals=signals_plot,
        signal_names=signal_names_plot,
        peaks_list=peaks_list_plot,
        matches=valid_matches,
        velocities=velocities_valid,
        distance=distance,
        output_path=plot_path,
        ignore_initial_seconds=ignore_initial_seconds,
    )
    
    print(f"Gráfico salvo em: {plot_path}")
    print("\nAnálise de velocidade de slug concluída!")


if __name__ == "__main__":
    main()

