import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from nptdms import TdmsFile


# =============================== CONFIGURAÇÕES ===============================
# Caminho do arquivo TDMS (ajuste se necessário)
file_path = 'G:/Meu Drive/LEMI/uncertainties/data_example/example/AOD45P01/AOD45P01_acc.tdms'

# Colunas a comparar (defina explicitamente para evitar interação)
# Use exatamente os nomes conforme df.columns (veja exemplo abaixo)
coluna_1 = "/'Untitled'/'Accel1'"
coluna_2 = "/'Untitled'/'Accel2'"

# Parâmetros da detecção/match
max_freq_hz = 10.0         # limite superior para análise/match (Hz)
tolerancia_match_hz = 0.05    # tolerância máxima de diferença entre picos (Hz)
prominencia_min = None       # proeminência mínima (None usa heurística)
altura_min = None            # altura mínima (None usa heurística)
distancia_min_hz = 2.0       # distância mínima entre picos (Hz)


# =============================== FUNÇÕES =====================================
def read_tdms_to_dataframe(tdms_path: str) -> pd.DataFrame:
    tdms_file = TdmsFile.read(tdms_path)
    df = tdms_file.as_dataframe()
    # Cria eixo de tempo relativo em segundos
    if "/'Untitled'/'Time'" in df.columns:
        time_col = df["/'Untitled'/'Time'"]
        start_time = time_col.iloc[0]
        df['X_Value'] = (time_col - start_time).dt.total_seconds()
    else:
        # fallback com passo constante (supõe 10 Hz se não houver tempo)
        df['X_Value'] = np.arange(len(df)) * 0.1
    return df


def compute_fft(time_s: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    # remove NaNs
    mask_valid = np.isfinite(time_s) & np.isfinite(y)
    time_s = time_s[mask_valid]
    y = y[mask_valid]
    # remove média para reduzir pico DC
    y = y - np.mean(y)
    # janela para reduzir vazamento espectral
    window = signal.windows.hann(len(y))
    yw = y * window
    dt = float(np.mean(np.diff(time_s)))
    fs = 1.0 / dt
    n = len(yw)
    fft_vals = np.fft.fft(yw)
    freqs = np.fft.fftfreq(n, dt)
    # apenas frequências positivas
    pos = freqs > 0
    freqs = freqs[pos]
    magnitude = np.abs(fft_vals[pos]) * 2.0 / np.sum(window)  # normalização aproximada
    return freqs, magnitude, fs


def hz_to_bins(distance_hz: float, freqs: np.ndarray) -> int:
    # converte distância em Hz para amostras aproximadas no eixo de frequência
    if len(freqs) < 2:
        return 1
    df = freqs[1] - freqs[0]
    bins = max(1, int(round(distance_hz / df)))
    return bins


def detect_peaks(freqs: np.ndarray, magnitude: np.ndarray,
                 max_freq: float,
                 prominencia_min: float | None,
                 altura_min: float | None,
                 distancia_min_hz: float) -> Tuple[np.ndarray, np.ndarray]:
    mask = freqs <= max_freq
    f = freqs[mask]
    m = magnitude[mask]

    # heurísticas se não fornecido
    if prominencia_min is None:
        prominencia_min = 0.1 * np.nanmax(m) if np.nanmax(m) > 0 else 0.0
    if altura_min is None:
        altura_min = 0.05 * np.nanmax(m) if np.nanmax(m) > 0 else 0.0

    distance_bins = hz_to_bins(distancia_min_hz, f)
    idx, _ = signal.find_peaks(m, prominence=prominencia_min, height=altura_min, distance=distance_bins)
    return f[idx], m[idx]


def match_frequencies(freqs_1: np.ndarray, mags_1: np.ndarray,
                      freqs_2: np.ndarray, mags_2: np.ndarray,
                      tol_hz: float) -> pd.DataFrame:
    matches: List[Tuple[float, float, float, float, float]] = []
    j0 = 0
    for i in range(len(freqs_1)):
        f1 = freqs_1[i]
        # avanço linear no segundo vetor (ambos ordenados)
        while j0 < len(freqs_2) and freqs_2[j0] < f1 - tol_hz:
            j0 += 1
        if j0 >= len(freqs_2):
            break
        # verificar candidatos dentro da tolerância
        candidates = []
        j = j0
        while j < len(freqs_2) and freqs_2[j] <= f1 + tol_hz:
            candidates.append(j)
            j += 1
        if not candidates:
            continue
        # escolhe o mais próximo
        j_best = min(candidates, key=lambda k: abs(freqs_2[k] - f1))
        f2 = freqs_2[j_best]
        dfreq = f2 - f1
        matches.append((f1, f2, dfreq, mags_1[i], mags_2[j_best]))

    result = pd.DataFrame(matches, columns=[
        'f1 (Hz)', 'f2 (Hz)', 'Δf (Hz)', 'Mag1', 'Mag2'
    ])
    return result


def main():
    df = read_tdms_to_dataframe(file_path)

    # valida colunas
    for c in (coluna_1, coluna_2):
        if c not in df.columns:
            raise ValueError(f"Coluna não encontrada no arquivo: {c}")

    time = df['X_Value'].values.astype(float)
    y1 = df[coluna_1].values.astype(float)
    y2 = df[coluna_2].values.astype(float)

    # FFTs
    freqs1, mag1, fs1 = compute_fft(time, y1)
    freqs2, mag2, fs2 = compute_fft(time, y2)
    fs = min(fs1, fs2)

    # Detecção de picos
    p_f1, p_m1 = detect_peaks(freqs1, mag1, max_freq_hz, prominencia_min, altura_min, distancia_min_hz)
    p_f2, p_m2 = detect_peaks(freqs2, mag2, max_freq_hz, prominencia_min, altura_min, distancia_min_hz)

    # Matching
    matches_df = match_frequencies(p_f1, p_m1, p_f2, p_m2, tolerancia_match_hz)

    # Saídas
    out_dir = os.path.dirname(file_path)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(file_path))[0]
    c1_simple = coluna_1.replace("/'Untitled'/", "").replace("'", "").replace("/", "_")
    c2_simple = coluna_2.replace("/'Untitled'/", "").replace("'", "").replace("/", "_")

    # CSV com picos e matches
    peaks1_df = pd.DataFrame({'freq (Hz)': p_f1, 'mag': p_m1})
    peaks2_df = pd.DataFrame({'freq (Hz)': p_f2, 'mag': p_m2})
    peaks1_path = os.path.join(out_dir, f"peaks_{c1_simple}_{base}.txt")
    peaks2_path = os.path.join(out_dir, f"peaks_{c2_simple}_{base}.txt")
    match_path = os.path.join(out_dir, f"matches_{c1_simple}_vs_{c2_simple}_{base}_tol{tolerancia_match_hz}Hz.txt")
    peaks1_df.to_csv(peaks1_path, sep='\t', index=False)
    peaks2_df.to_csv(peaks2_path, sep='\t', index=False)
    matches_df.to_csv(match_path, sep='\t', index=False)

    # Gráficos
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Espectros
    pos1 = freqs1 <= max_freq_hz
    pos2 = freqs2 <= max_freq_hz
    axes[0, 0].plot(freqs1[pos1], mag1[pos1], 'b-', lw=0.8, label=c1_simple)
    axes[0, 0].plot(p_f1, p_m1, 'bo', ms=5, alpha=0.7)
    axes[0, 0].set_title(f"Espectro - {c1_simple}")
    axes[0, 0].set_xlabel('Frequência (Hz)')
    axes[0, 0].set_ylabel('Magnitude')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(freqs2[pos2], mag2[pos2], 'r-', lw=0.8, label=c2_simple)
    axes[0, 1].plot(p_f2, p_m2, 'ro', ms=5, alpha=0.7)
    axes[0, 1].set_title(f"Espectro - {c2_simple}")
    axes[0, 1].set_xlabel('Frequência (Hz)')
    axes[0, 1].set_ylabel('Magnitude')
    axes[0, 1].grid(True, alpha=0.3)

    # Matches (stem + tabela resumida)
    axes[1, 0].stem(matches_df['f1 (Hz)'], matches_df['Mag1'], linefmt='b-', markerfmt='bo', basefmt='k-', label=c1_simple)
    axes[1, 0].stem(matches_df['f2 (Hz)'], matches_df['Mag2'], linefmt='r-', markerfmt='ro', basefmt='k-', label=c2_simple)
    axes[1, 0].set_title(f"Picos casados (tolerância = {tolerancia_match_hz} Hz)")
    axes[1, 0].set_xlabel('Frequência (Hz)')
    axes[1, 0].set_ylabel('Magnitude')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    axes[1, 0].set_xlim(0, max_freq_hz)

    # Tabela de matches (texto)
    axes[1, 1].axis('off')
    txt_lines = [
        f"Total de matches: {len(matches_df)}",
        f"Tolerância: {tolerancia_match_hz} Hz",
        "",
    ]
    head = "f1 (Hz)    f2 (Hz)    Δf (Hz)    Mag1       Mag2"
    txt_lines.append(head)
    for _, row in matches_df.head(20).iterrows():
        txt_lines.append(f"{row['f1 (Hz)']:<10.3f}{row['f2 (Hz)']:<11.3f}{row['Δf (Hz)']:<11.3f}{row['Mag1']:<11.3f}{row['Mag2']:<11.3f}")
    axes[1, 1].text(0.01, 0.98, "\n".join(txt_lines), va='top', family='monospace')

    plt.tight_layout()
    out_png = os.path.join(out_dir, f"compare_fft_{c1_simple}_vs_{c2_simple}_{base}.png")
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Peaks 1 salvo em: {peaks1_path}")
    print(f"Peaks 2 salvo em: {peaks2_path}")
    print(f"Matches salvo em: {match_path}")
    print(f"Gráfico salvo em: {out_png}")


if __name__ == '__main__':
    main()


