import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nptdms import TdmsFile
from scipy import signal
import os

# Configuração do arquivo
file_path = 'G:/Meu Drive/LEMI/uncertainties/data_example/example/AOD45P01/AOD45P01_acc.tdms'

def apply_lowpass_filter(data: np.ndarray, fs: float, fc: float, order: int = 4):
    """
    Aplica um filtro passa-baixa Butterworth aos dados.
    """
    nyquist = fs / 2
    normalized_cutoff = fc / nyquist
    b, a = signal.butter(order, normalized_cutoff, btype='low', analog=False)
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

def compare_filters():
    """
    Compara diferentes frequências de corte do filtro passa-baixa
    """
    try:
        # Lê o arquivo TDMS
        tdms_file = TdmsFile.read(file_path)
        df = tdms_file.as_dataframe()
        
        # Cria coluna de tempo
        if "/'Untitled'/'Time'" in df.columns:
            time_col = df["/'Untitled'/'Time'"]
            start_time = time_col.iloc[0]
            df['X_Value'] = (time_col - start_time).dt.total_seconds()
        else:
            df['X_Value'] = np.arange(len(df)) * 0.1
        
        # Escolhe uma coluna para análise (Accel1)
        coluna = "/'Untitled'/'Accel1'"
        y_data = df[coluna].values
        time = df['X_Value'].values
        
        # Calcula frequência de amostragem
        dt = np.mean(np.diff(time))
        fs = 1/dt
        
        # Diferentes frequências de corte para comparação
        freq_cortes = [10, 20, 50, 100]
        
        # Cria figura para comparação
        fig, axes = plt.subplots(len(freq_cortes) + 1, 2, figsize=(20, 4*(len(freq_cortes) + 1)))
        
        # Plota sinal original
        axes[0, 0].plot(time[:int(2/dt)], y_data[:int(2/dt)], 'b-', linewidth=0.8, label='Sinal Original')
        axes[0, 0].set_xlabel('Tempo (s)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].set_title('Sinal Original (Primeiros 2 segundos)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # FFT do sinal original
        n = len(y_data)
        fft_result = np.fft.fft(y_data)
        freqs = np.fft.fftfreq(n, dt)
        magnitude = np.abs(fft_result)
        positive_freq_mask = freqs > 0
        
        axes[0, 1].plot(freqs[positive_freq_mask], magnitude[positive_freq_mask], 'b-', linewidth=0.8, label='Original')
        axes[0, 1].set_xlabel('Frequência (Hz)')
        axes[0, 1].set_ylabel('Magnitude')
        axes[0, 1].set_title('Espectro Original')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        axes[0, 1].set_xlim(0, 2000)  # Limita para melhor visualização
        
        # Aplica diferentes filtros e plota resultados
        for i, fc in enumerate(freq_cortes):
            # Aplica filtro
            y_filtered = apply_lowpass_filter(y_data, fs, fc)
            
            # Plota série temporal filtrada
            axes[i+1, 0].plot(time[:int(2/dt)], y_data[:int(2/dt)], 'b-', alpha=0.5, linewidth=0.5, label='Original')
            axes[i+1, 0].plot(time[:int(2/dt)], y_filtered[:int(2/dt)], 'r-', linewidth=1.2, label=f'Filtrado (fc={fc} Hz)')
            axes[i+1, 0].set_xlabel('Tempo (s)')
            axes[i+1, 0].set_ylabel('Amplitude')
            axes[i+1, 0].set_title(f'Sinal Filtrado - fc = {fc} Hz')
            axes[i+1, 0].grid(True, alpha=0.3)
            axes[i+1, 0].legend()
            
            # FFT do sinal filtrado
            fft_filtered = np.fft.fft(y_filtered)
            magnitude_filtered = np.abs(fft_filtered)
            
            axes[i+1, 1].plot(freqs[positive_freq_mask], magnitude[positive_freq_mask], 'b-', alpha=0.5, linewidth=0.5, label='Original')
            axes[i+1, 1].plot(freqs[positive_freq_mask], magnitude_filtered[positive_freq_mask], 'r-', linewidth=0.8, label=f'Filtrado (fc={fc} Hz)')
            axes[i+1, 1].axvline(x=fc, color='g', linestyle='--', alpha=0.7, label=f'Fc = {fc} Hz')
            axes[i+1, 1].set_xlabel('Frequência (Hz)')
            axes[i+1, 1].set_ylabel('Magnitude')
            axes[i+1, 1].set_title(f'Espectro Filtrado - fc = {fc} Hz')
            axes[i+1, 1].grid(True, alpha=0.3)
            axes[i+1, 1].legend()
            axes[i+1, 1].set_xlim(0, 2000)
        
        plt.tight_layout()
        
        # Salva o gráfico
        output_dir = os.path.dirname(file_path)
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = os.path.join(output_dir, f"filter_comparison_{base_name}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Comparação de filtros salva em: {output_path}")
        
        # Calcula estatísticas para cada filtro
        print("\nEstatísticas dos filtros:")
        print("="*60)
        print(f"{'Fc (Hz)':<10} {'RMS Original':<15} {'RMS Filtrado':<15} {'Redução (%)':<15}")
        print("-"*60)
        
        for fc in freq_cortes:
            y_filtered = apply_lowpass_filter(y_data, fs, fc)
            rms_original = np.sqrt(np.mean(y_data**2))
            rms_filtered = np.sqrt(np.mean(y_filtered**2))
            reducao = (1 - rms_filtered/rms_original) * 100
            print(f"{fc:<10} {rms_original:<15.6f} {rms_filtered:<15.6f} {reducao:<15.2f}")
        
        return df
        
    except Exception as e:
        print(f"Erro ao processar arquivo: {e}")
        return None

if __name__ == "__main__":
    df = compare_filters()
