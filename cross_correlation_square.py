"""
Correlação cruzada entre dois sinais (sensores) a partir de arquivo TDMS ou texto.
Baseado na forma de leitura de OMAE-slug_velocity.py.
Permite aplicar ou não filtro passa-baixa aos sinais.
"""

import os
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nptdms import TdmsFile
from scipy import signal


################################################################################
#                                            INPUTS
################################################################################
file_path = "G:/Meu Drive/LEMI/uncertainties/data_example/example/freq_slug/AOH00P02/AOH00P02_acc.tdms"

# Filtro passa-baixa: None = não aplicar; valor em Hz = frequência de corte
freq_corte = None  # Ex.: 50.0 para filtrar acima de 50 Hz

# Janelamento temporal (None = usar todo o sinal)
time_min = None  # segundos
time_max = None  # segundos

# -------------------------------------------------------------------------
# Estimação do lag (robustez / física)
# Faixa de velocidades plausíveis (usada para restringir o lag pesquisado):
# lag_min = d / v_max ; lag_max = d / v_min
VEL_MIN_MS = 0.05    # m/s  (aumente se ainda pegar picos muito próximos de zero)
VEL_MAX_MS = 20.0   # m/s

# (Opcional) refinamento em torno de uma velocidade de referência
# Se USE_VEL_REF_WINDOW for True, o pico da correlação é buscado
# apenas em uma janela estreita em torno de |d / VEL_REF_MS|.
USE_VEL_REF_WINDOW = False
VEL_REF_MS = 1.0          # m/s (velocidade de referência aproximada)
VEL_REF_TOL_FRAC = 0.3    # fração de tolerância (0.3 => ±30% em torno de |lag_ref|)

# Ignora região de lag muito próxima de zero (em segundos)
IGNORE_LAG_BELOW_S = 0.0

# Se True, procura pico em |corr| (robusto a inversão de polaridade entre sensores)
USE_ABS_CORR_FOR_PEAK = False

# Pré-processamento (não muda a física, mas pode reduzir pico em lag=0 por tendência/offset)
DETREND_SIGNALS = True
Z_SCORE_SIGNALS = True

# Se True, usa os sinais ao quadrado (energia) na correlação cruzada,
# dando mais peso a picos de maior magnitude.
USE_SQUARED_SIGNALS_FOR_CC = True

# Se True, usa apenas a parte positiva, suaviza (envelope) e binariza o sinal
# antes de calcular a correlação cruzada (sinal quadrado).
USE_BINARY_ENVELOPE_FOR_CC = True
ENVELOPE_SMOOTH_WINDOW_S = 0.05  # janela de suavização em segundos

################################################################################
#                                            END INPUTS
################################################################################


def read_file(file_path: str) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Lê o arquivo TDMS ou texto (tab-separado) e retorna um DataFrame.
    Mesma lógica de OMAE-slug_velocity.py.
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


def apply_lowpass_filter(
    data: np.ndarray, fs: float, fc: Optional[float], order: int = 4
) -> np.ndarray:
    """
    Aplica filtro passa-baixa Butterworth. Se fc for None, retorna os dados sem filtrar.
    """
    if fc is None:
        return data
    nyquist = fs / 2
    normalized_cutoff = fc / nyquist
    if normalized_cutoff >= 1.0:
        return data
    b, a = signal.butter(order, normalized_cutoff, btype="low", analog=False)
    return signal.filtfilt(b, a, data)


def apply_time_window(
    time: np.ndarray,
    s1: np.ndarray,
    s2: np.ndarray,
    time_min: Optional[float],
    time_max: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aplica janelamento temporal nos dados."""
    mask = np.ones(len(time), dtype=bool)
    if time_min is not None:
        mask = mask & (time >= time_min)
    if time_max is not None:
        mask = mask & (time <= time_max)
    return time[mask], s1[mask], s2[mask]


def preprocess_signal(x: np.ndarray) -> np.ndarray:
    """Remove NaNs, detrend e normaliza (opcional)."""
    x = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0)
    if DETREND_SIGNALS:
        x = signal.detrend(x, type="constant")
    if Z_SCORE_SIGNALS:
        s = np.std(x)
        if s > 1e-15:
            x = (x - np.mean(x)) / s
        else:
            x = x - np.mean(x)
    return x


def pick_lag_peak(
    lags_time: np.ndarray, corr: np.ndarray, distancia_m: float
) -> int:
    """
    Escolhe o índice do pico de correlação dentro de uma janela física de lag.
    """
    metric = np.abs(corr) if USE_ABS_CORR_FOR_PEAK else corr

    lag_min = distancia_m / VEL_MAX_MS if VEL_MAX_MS > 0 else 0.0
    lag_max = distancia_m / VEL_MIN_MS if VEL_MIN_MS > 0 else np.inf

    abs_lag = np.abs(lags_time)
    mask = (abs_lag >= max(IGNORE_LAG_BELOW_S, lag_min)) & (abs_lag <= lag_max)

    # Se houver velocidade de referência configurada, restringe ainda mais
    if USE_VEL_REF_WINDOW and VEL_REF_MS > 0:
        lag_ref = distancia_m / VEL_REF_MS
        lag_ref_abs = abs(lag_ref)
        if lag_ref_abs > 0:
            tol = VEL_REF_TOL_FRAC * lag_ref_abs
            mask = mask & (np.abs(abs_lag - lag_ref_abs) <= tol)

    if not np.any(mask):
        # fallback: pico global (mas avisa no terminal em main)
        return int(np.argmax(metric))

    idx_local = int(np.argmax(metric[mask]))
    return int(np.where(mask)[0][idx_local])


def build_binary_envelope(
    sig: np.ndarray, fs: float, threshold_frac: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Usa apenas parte positiva, suaviza (envelope) e binariza.
    threshold_frac é a fração do valor máximo do envelope (0–1).

    Retorna:
        sig_pos: sinal retificado (parte positiva)
        env:     envelope suavizado
        binary:  sinal quadrado (0/1) após limiar
    """
    sig_pos = np.maximum(sig, 0.0)
    win_samples = max(1, int(ENVELOPE_SMOOTH_WINDOW_S * fs))
    if win_samples > 1:
        kernel = np.ones(win_samples) / win_samples
        env = np.convolve(sig_pos, kernel, mode="same")
    else:
        env = sig_pos

    max_env = np.max(env) if env.size > 0 else 0.0
    if max_env <= 0:
        return sig_pos, env, np.zeros_like(env)
    thr = threshold_frac * max_env
    binary = (env >= thr).astype(float)
    return sig_pos, env, binary


def cross_correlation_normalized(
    s1: np.ndarray, s2: np.ndarray, mode: str = "full"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Correlação cruzada normalizada (coeficiente em [-1, 1]).
    Retorna (lags em amostras, correlação).
    """
    s1 = np.asarray(s1, dtype=float)
    s2 = np.asarray(s2, dtype=float)
    s1_ = s1 - np.mean(s1)
    s2_ = s2 - np.mean(s2)
    c = signal.correlate(s1_, s2_, mode=mode)
    n = len(s1)
    # Normalização: correlação máxima = n * std(s1)*std(s2)
    norm = np.sqrt(np.sum(s1_**2) * np.sum(s2_**2))
    if norm > 1e-15:
        c = c / norm
    lags = signal.correlation_lags(len(s1), len(s2), mode=mode)
    return lags, c


def main():
    df, _ = read_file(file_path)

    if "X_Value" not in df.columns:
        print("ERRO: Coluna X_Value não encontrada.")
        return

    # Lista colunas disponíveis (exclui tempo; prioriza colunas numéricas)
    exclude = {"X_Value", "/'Untitled'/'Time'"}
    colunas_num = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    colunas = colunas_num if colunas_num else [c for c in df.columns if c not in exclude]

    print("\nColunas disponíveis para correlação cruzada:")
    for i, col in enumerate(colunas, 1):
        print(f"  {i:2d}. {col}")
    n_cols = len(colunas)
    if n_cols < 2:
        print("ERRO: É necessário pelo menos 2 colunas de sinal.")
        return

    # Escolha do primeiro sensor
    while True:
        try:
            escolha1 = int(input(f"\nPrimeiro sensor (1-{n_cols}): "))
            if 1 <= escolha1 <= n_cols:
                coluna_sensor1 = colunas[escolha1 - 1]
                break
            print(f"Digite um número entre 1 e {n_cols}.")
        except ValueError:
            print("Entrada inválida. Digite um número.")

    # Escolha do segundo sensor
    while True:
        try:
            escolha2 = int(input(f"Segundo sensor (1-{n_cols}): "))
            if 1 <= escolha2 <= n_cols:
                if escolha2 == escolha1:
                    print("Escolha uma coluna diferente do primeiro sensor.")
                    continue
                coluna_sensor2 = colunas[escolha2 - 1]
                break
            print(f"Digite um número entre 1 e {n_cols}.")
        except ValueError:
            print("Entrada inválida. Digite um número.")

    print(f"\nSensores selecionados: 1 = {coluna_sensor1}  |  2 = {coluna_sensor2}")

    time = df["X_Value"].values
    s1 = np.nan_to_num(df[coluna_sensor1].values, nan=0.0)
    s2 = np.nan_to_num(df[coluna_sensor2].values, nan=0.0)

    if time_min is not None or time_max is not None:
        time, s1, s2 = apply_time_window(time, s1, s2, time_min, time_max)

    dt = np.mean(np.diff(time))
    fs = 1.0 / dt
    print(f"Frequência de amostragem: {fs:.2f} Hz")
    print(f"Filtro passa-baixa: {'Não aplicado' if freq_corte is None else f'{freq_corte} Hz'}")

    # Distância entre sensores (para cálculo da velocidade e janela física do lag)
    while True:
        try:
            distancia_m = float(input("\nDistância física entre os dois sensores (m): "))
            if distancia_m > 0:
                break
            print("A distância deve ser maior que zero.")
        except ValueError:
            print("Entrada inválida. Digite um número (ex.: 0.5).")

    s1_f = apply_lowpass_filter(s1, fs, freq_corte)
    s2_f = apply_lowpass_filter(s2, fs, freq_corte)

    # Pré-processamento ou envelope binário antes da correlação
    if USE_BINARY_ENVELOPE_FOR_CC:
        while True:
            try:
                txt = input(
                    "\nLimiar para binarização do envelope (fração do máximo, 0–1, vazio = 0.5): "
                ).strip()
                if txt == "":
                    thr_frac = 0.5
                    break
                thr_frac = float(txt)
                if 0.0 < thr_frac < 1.0:
                    break
                print("Digite um valor entre 0 e 1 (ex.: 0.4).")
            except ValueError:
                print("Entrada inválida. Digite um número.")

        # Para esta análise, usa-se o sinal elevado ao quadrado antes do envelope/binário
        s1_sq = s1_f**2
        s2_sq = s2_f**2
        s1_pos, s1_env, s1_bin = build_binary_envelope(s1_sq, fs, thr_frac)
        s2_pos, s2_env, s2_bin = build_binary_envelope(s2_sq, fs, thr_frac)

        # Para a correlação, usamos o envelope contínuo (não binário)
        s1_cc = s1_env
        s2_cc = s2_env
    else:
        s1_cc = preprocess_signal(s1_f)
        s2_cc = preprocess_signal(s2_f)

        # Opcional: correlação dos quadrados (energia), para dar mais peso a picos
        if USE_SQUARED_SIGNALS_FOR_CC:
            s1_cc = s1_cc**2
            s2_cc = s2_cc**2

    lags_samp, corr = cross_correlation_normalized(s1_cc, s2_cc, mode="full")
    lags_time = lags_samp * dt

    imax = pick_lag_peak(lags_time, corr, distancia_m)
    lag_max_s = lags_time[imax]
    lag_max_samp = lags_samp[imax]
    print(f"\nCorrelação cruzada (normalizada):")
    print(f"  Lag de máxima correlação: {lag_max_s:.4f} s ({lag_max_samp} amostras)")
    print(f"  Valor da correlação no máximo: {corr[imax]:.4f}")

    lag_min = distancia_m / VEL_MAX_MS if VEL_MAX_MS > 0 else 0.0
    lag_max = distancia_m / VEL_MIN_MS if VEL_MIN_MS > 0 else np.inf
    print(
        f"  Janela física aplicada: |lag| ∈ [{max(IGNORE_LAG_BELOW_S, lag_min):.4f}, {lag_max:.4f}] s "
        f"(v ∈ [{VEL_MIN_MS}, {VEL_MAX_MS}] m/s)"
    )

    # Velocidade = distância / |lag|; lag > 0 => propagação do sensor 1 para o 2
    lag_abs_s = abs(lag_max_s)
    if lag_abs_s < 1e-9:
        print("  Aviso: lag muito próximo de zero, velocidade não calculada.")
        velocidade_ms = float("nan")
    else:
        velocidade_ms = distancia_m / lag_abs_s
    velocidade_kmh = velocidade_ms * 3.6
    print(f"\nVelocidade estimada (correlação cruzada):")
    print(f"  {velocidade_ms:.2f} m/s  ({velocidade_kmh:.2f} km/h)")
    if lag_max_s > 0:
        print(f"  Sentido: do primeiro ao segundo sensor (lag > 0)")
    else:
        print(f"  Sentido: do segundo ao primeiro sensor (lag < 0)")

    # Plot: correlação vs lag
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=False)
    axes[0].plot(lags_time, corr, color="black", linewidth=1.2)
    axes[0].axvline(lag_max_s, color="red", linestyle="--", linewidth=1, label=f"Lag máx = {lag_max_s:.3f} s")
    axes[0].axhline(0, color="gray", linestyle="-", linewidth=0.5)
    axes[0].set_ylabel("Correlação normalizada")
    axes[0].set_xlabel("Lag [s]")
    axes[0].set_xlim(-6.0, 6.0)
    titulo_cc = "Correlação cruzada entre os dois sensores"
    if not np.isnan(velocidade_ms):
        titulo_cc += f"  |  v = {velocidade_ms:.2f} m/s (d = {distancia_m} m)"
    axes[0].set_title(titulo_cc)
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Plot: sinais (filtrados ou não) em janela, com transparência para sobreposição
    axes[1].plot(time, s1_f, color="C0", linewidth=0.8, alpha=0.5, label=coluna_sensor1.split("/")[-1].strip("'"))
    axes[1].plot(time, s2_f, color="C1", linewidth=0.8, alpha=0.5, label=coluna_sensor2.split("/")[-1].strip("'"))

    # Se envelope binário foi usado na correlação, plota também os sinais quadrados (0/1)
    if USE_BINARY_ENVELOPE_FOR_CC:
        axes[1].step(time, s1_bin, where="mid", color="C0", alpha=0.9, linewidth=1.0, linestyle="--", label="Binário 1")
        axes[1].step(time, s2_bin, where="mid", color="C1", alpha=0.9, linewidth=1.0, linestyle="--", label="Binário 2")

        # Novo plot dedicado: base positiva, envelope suavizado e sinal quadrado
        fig_env, axes_env = plt.subplots(2, 1, figsize=(9, 5), sharex=True)

        label1 = coluna_sensor1.split("/")[-1].strip("'")
        label2 = coluna_sensor2.split("/")[-1].strip("'")

        # Sensor 1: eixo esquerdo = sinal base positivo; eixo direito = envelope e sinal quadrado (0–1)
        ax1_left = axes_env[0]
        ax1_left.plot(time, s1_pos, color="C0", alpha=0.5, linewidth=0.9, label=f"{label1} (positivo)")
        ax1_left.set_ylabel("Amplitude do sinal")
        ax1_left.set_title("Sensor 1: base positiva, envelope e sinal quadrado")
        ax1_left.grid(True, alpha=0.3)

        ax1_right = ax1_left.twinx()
        max_env1 = np.max(s1_env) if s1_env.size > 0 else 1.0
        env1_norm = s1_env / max_env1 if max_env1 > 0 else s1_env
        ax1_right.plot(time, env1_norm, color="C2", alpha=0.9, linewidth=1.2, label="Envelope suavizado 1")
        ax1_right.step(time, s1_bin, where="mid", color="k", alpha=0.9, linewidth=1.0, linestyle="--", label="Sinal quadrado 1")
        ax1_right.set_ylabel("Envelope / binário (0–1)")
        ax1_right.set_ylim(-0.1, 1.1)

        # Sensor 2
        ax2_left = axes_env[1]
        ax2_left.plot(time, s2_pos, color="C1", alpha=0.5, linewidth=0.9, label=f"{label2} (positivo)")
        ax2_left.set_ylabel("Amplitude do sinal")
        ax2_left.set_xlabel("Tempo [s]")
        ax2_left.set_title("Sensor 2: base positiva, envelope e sinal quadrado")
        ax2_left.grid(True, alpha=0.3)

        ax2_right = ax2_left.twinx()
        max_env2 = np.max(s2_env) if s2_env.size > 0 else 1.0
        env2_norm = s2_env / max_env2 if max_env2 > 0 else s2_env
        ax2_right.plot(time, env2_norm, color="C3", alpha=0.9, linewidth=1.2, label="Envelope suavizado 2")
        ax2_right.step(time, s2_bin, where="mid", color="k", alpha=0.9, linewidth=1.0, linestyle="--", label="Sinal quadrado 2")
        ax2_right.set_ylabel("Envelope / binário (0–1)")
        ax2_right.set_ylim(-0.1, 1.1)

        plt.tight_layout()
    axes[1].set_ylabel("Amplitude")
    axes[1].set_xlabel("Tempo [s]")
    axes[1].set_title("Sinais utilizados (filtrados)" if freq_corte else "Sinais utilizados (sem filtro)")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_dir = os.path.dirname(file_path)
    base = os.path.splitext(os.path.basename(file_path))[0]
    name1 = coluna_sensor1.replace("/", "_").replace("'", "")
    name2 = coluna_sensor2.replace("/", "_").replace("'", "")
    out_path = os.path.join(out_dir, f"cross_corr_{name1}_vs_{name2}_{base}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nFigura salva: {out_path}")


if __name__ == "__main__":
    main()
