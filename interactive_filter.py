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

def interactive_filter_test():
    """
    Permite ao usuário testar diferentes frequências de corte interativamente
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
        
        # Colunas disponíveis
        data_columns = [col for col in df.columns if col != "/'Untitled'/'Time'" and col != 'X_Value']
        
        print("Colunas disponíveis para análise:")
        for i, col in enumerate(data_columns, 1):
            col_name = col.split("/")[-1].replace("'", "").replace("_", " ")
            print(f"{i}. {col_name}")
        
        # Escolhe coluna
        while True:
            try:
                escolha = int(input("\nEscolha o número da coluna para análise: "))
                if 1 <= escolha <= len(data_columns):
                    coluna = data_columns[escolha-1]
                    col_name = coluna.split("/")[-1].replace("'", "").replace("_", " ")
                    print(f"Coluna escolhida: {col_name}")
                    break
                else:
                    print(f"Por favor, escolha um número entre 1 e {len(data_columns)}")
            except ValueError:
                print("Entrada inválida. Digite um número válido.")
        
        # Dados da coluna escolhida
        y_data = df[coluna].values
        time = df['X_Value'].values
        
        # Calcula frequência de amostragem
        dt = np.mean(np.diff(time))
        fs = 1/dt
        
        print(f"\nInformações do sinal:")
        print(f"Frequência de amostragem: {fs:.2f} Hz")
        print(f"Frequência de Nyquist: {fs/2:.2f} Hz")
        print(f"Duração total: {time[-1]:.2f} s")
        print(f"Número de pontos: {len(y_data)}")
        
        # Loop para testar diferentes frequências
        while True:
            try:
                fc_input = input(f"\nDigite a frequência de corte em Hz (0 para sair, max {fs/2:.0f}): ")
                fc = float(fc_input)
                
                if fc == 0:
                    print("Encerrando análise...")
                    break
                
                if fc <= 0 or fc >= fs/2:
                    print(f"Frequência deve estar entre 0 e {fs/2:.0f} Hz")
                    continue
                
                # Aplica filtro
                y_filtered = apply_lowpass_filter(y_data, fs, fc)
                
                # Cria gráfico
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                
                # Série temporal completa
                axes[0, 0].plot(time, y_data, 'b-', alpha=0.7, label='Original', linewidth=0.5)
                axes[0, 0].plot(time, y_filtered, 'r-', label=f'Filtrado (fc={fc} Hz)', linewidth=0.8)
                axes[0, 0].set_xlabel('Tempo (s)')
                axes[0, 0].set_ylabel('Amplitude')
                axes[0, 0].set_title(f'Série Temporal - {col_name}')
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 0].legend()
                
                # Zoom da série temporal (primeiros 2 segundos)
                mask_zoom = time <= 2.0
                axes[0, 1].plot(time[mask_zoom], y_data[mask_zoom], 'b-', alpha=0.7, label='Original', linewidth=0.8)
                axes[0, 1].plot(time[mask_zoom], y_filtered[mask_zoom], 'r-', label=f'Filtrado (fc={fc} Hz)', linewidth=1.2)
                axes[0, 1].set_xlabel('Tempo (s)')
                axes[0, 1].set_ylabel('Amplitude')
                axes[0, 1].set_title('Zoom - Primeiros 2 segundos')
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].legend()
                
                # Espectros de frequência
                n = len(y_data)
                fft_original = np.fft.fft(y_data)
                fft_filtered = np.fft.fft(y_filtered)
                freqs = np.fft.fftfreq(n, dt)
                positive_freq_mask = freqs > 0
                
                axes[1, 0].plot(freqs[positive_freq_mask], np.abs(fft_original[positive_freq_mask]), 'b-', alpha=0.7, label='Original', linewidth=0.8)
                axes[1, 0].plot(freqs[positive_freq_mask], np.abs(fft_filtered[positive_freq_mask]), 'r-', label='Filtrado', linewidth=0.8)
                axes[1, 0].axvline(x=fc, color='g', linestyle='--', alpha=0.7, label=f'Fc = {fc} Hz')
                axes[1, 0].set_xlabel('Frequência (Hz)')
                axes[1, 0].set_ylabel('Magnitude')
                axes[1, 0].set_title('Espectros de Frequência')
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].legend()
                axes[1, 0].set_xlim(0, min(fs/2, fc*3))
                
                # Resposta do filtro
                nyquist = fs / 2
                normalized_cutoff = fc / nyquist
                b, a = signal.butter(4, normalized_cutoff, btype='low', analog=False)
                w, h = signal.freqz(b, a, worN=8000)
                axes[1, 1].plot(0.5*fs*w/np.pi, np.abs(h), 'g-', linewidth=2, label='Resposta do Filtro')
                axes[1, 1].axvline(x=fc, color='r', linestyle='--', alpha=0.7, label=f'Fc = {fc} Hz')
                axes[1, 1].set_xlabel('Frequência (Hz)')
                axes[1, 1].set_ylabel('Magnitude')
                axes[1, 1].set_title('Resposta do Filtro Butterworth')
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].legend()
                axes[1, 1].set_xlim(0, min(fs/2, fc*3))
                
                plt.tight_layout()
                plt.show()
                
                # Calcula estatísticas
                rms_original = np.sqrt(np.mean(y_data**2))
                rms_filtered = np.sqrt(np.mean(y_filtered**2))
                reducao = (1 - rms_filtered/rms_original) * 100
                
                print(f"\nEstatísticas para fc = {fc} Hz:")
                print(f"RMS Original: {rms_original:.6f}")
                print(f"RMS Filtrado: {rms_filtered:.6f}")
                print(f"Redução de energia: {reducao:.2f}%")
                
                # Pergunta se quer salvar
                salvar = input("\nDeseja salvar este resultado? (s/n): ").lower()
                if salvar == 's':
                    output_dir = os.path.dirname(file_path)
                    os.makedirs(output_dir, exist_ok=True)
                    base_name = os.path.splitext(os.path.basename(file_path))[0]
                    coluna_simples = coluna.replace("/'Untitled'/", "").replace("'", "").replace("/", "_")
                    
                    output_path = os.path.join(output_dir, f"interactive_filter_{coluna_simples}_{base_name}_fc{fc}Hz.png")
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')
                    print(f"Gráfico salvo em: {output_path}")
                
            except ValueError:
                print("Entrada inválida. Digite um número válido.")
            except KeyboardInterrupt:
                print("\nAnálise interrompida pelo usuário.")
                break
        
    except Exception as e:
        print(f"Erro ao processar arquivo: {e}")

if __name__ == "__main__":
    interactive_filter_test()
