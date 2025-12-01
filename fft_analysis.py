import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import sys
from nptdms import TdmsFile
from scipy import signal

####################################################################################################################################################
#                                            INPUTS
####################################################################################################################################################
file_path = 'G:/Meu Drive/LEMI/uncertainties/data_example/example/freq_slug/AOU05P03/AOU05P03_acc.tdms' #Insira o caminho do arquivo a ser analisado NOTE: USE SEMPRE A BARRA NORMAL '/', SE ESTIVER INVERTIDA, MODIFIQUE-A
freq_corte = 1000.0  # Frequência de corte do filtro passa-baixa em Hz (None = sem filtragem) - MODIFIQUE ESTE VALOR CONFORME NECESSÁRIO
freq_min_plot = 0.0  # Frequência mínima para o plot da FFT em Hz - MODIFIQUE ESTE VALOR CONFORME NECESSÁRIO
freq_max_plot = 100.0  # Frequência máxima para o plot da FFT em Hz - MODIFIQUE ESTE VALOR CONFORME NECESSÁRIO

### Colunas de interesse -> Insira o nome das colunas a plotar e avaliar do arquivo TDMS
# Lista de colunas para análise: [nome_coluna, apelido, unidade]
colunas_analise = [
    ["/'Untitled'/'Accel1'", r'Accel1', r'[g]'],
    ["/'Untitled'/'Accel2'", r'Accel2', r'[g]'],
    ["/'Untitled'/'Accel3'", r'Accel3', r'[g]'],
    ["/'Untitled'/'Accel4'", r'Accel4', r'[g]'],
    ["/'Untitled'/'DP_Validyne'", r'DP\ Validyne', r'[Pa]'],
    ["/'Untitled'/'PIT-M-301'", r'PIT-M-301', r'[Bar]']
]

####################################################################################################################################################
#                                            END INPUTS
####################################################################################################################################################

def read_file(file_path: str):
    """
    Lê o arquivo TDMS e retorna um DataFrame do pandas.
    
    Args:
        file_path (str): Caminho para o arquivo TDMS
        
    Returns:
        tuple: (DataFrame com os dados, data do teste experimental)
    """
    try:
        # Lê o arquivo TDMS
        tdms_file = TdmsFile.read(file_path)
        
        # Converte para DataFrame
        df = tdms_file.as_dataframe()
        
        # Extrai informações do arquivo
        data_teste = None
        try:
            metadata = tdms_file.read_metadata()
            if hasattr(metadata, 'description'):
                data_teste = metadata.description
        except:
            pass
        
        # Cria coluna de tempo baseada na coluna de tempo do TDMS
        if "/'Untitled'/'Time'" in df.columns:
            # Converte timestamps para segundos relativos
            time_col = df["/'Untitled'/'Time'"]
            start_time = time_col.iloc[0]
            df['X_Value'] = (time_col - start_time).dt.total_seconds()
        else:
            # Fallback: cria uma coluna de tempo baseada no índice
            df['X_Value'] = np.arange(len(df)) * 0.1  # Assume 10 Hz de frequência de amostragem
        
        # Calcula colunas corrigidas se as colunas originais existirem
        if 'J Ar' in df.columns:
            df['J Ar corrigido'] = df['J Ar'] * (1 - 0.06675)
        if 'J Agua' in df.columns:
            df['J Agua corrigido'] = df['J Agua'] * (1 - 0.06675)
        
        return df, data_teste
        
    except Exception as e:
        print(f"Erro ao ler arquivo TDMS: {e}")
        print("Tentando ler como arquivo de texto...")
        
        # Fallback para arquivo de texto
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as f:
                lines = f.readlines()
        
        # Procura por marcadores de cabeçalho
        header_count = 0
        header_end_idx = 0
        for i, line in enumerate(lines):
            if '***End_of_Header***' in line:
                header_count += 1
                if header_count == 2:
                    header_end_idx = i + 1
                    break
        
        if header_end_idx == 0:
            # Se não encontrar marcadores, assume que os dados começam após algumas linhas
            header_end_idx = 5
        
        column_names = [name.strip() for name in lines[header_end_idx].strip().split('\t')]
        
        df = pd.read_csv(file_path, 
                         sep='\t',
                         skiprows=header_end_idx+1,
                         decimal=',',
                         na_values=[''],
                         encoding='utf-8',
                         names=column_names)
        
        # Adiciona coluna de tempo se não existir
        if 'X_Value' not in df.columns:
            df['X_Value'] = np.arange(len(df)) * 0.1
        
        # Calcula colunas corrigidas
        if 'J Ar' in df.columns:
            df['J Ar corrigido'] = df['J Ar'] * (1 - 0.06675)
        if 'J Agua' in df.columns:
            df['J Agua corrigido'] = df['J Agua'] * (1 - 0.06675)
        
        return df, None

def apply_lowpass_filter(data: np.ndarray, fs: float, fc: float, order: int = 4):
    """
    Aplica um filtro passa-baixa Butterworth aos dados.
    
    Args:
        data (np.ndarray): Dados de entrada
        fs (float): Frequência de amostragem em Hz
        fc (float): Frequência de corte em Hz (None = sem filtragem)
        order (int): Ordem do filtro (padrão: 4)
        
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
    
    # Aplica o filtro
    filtered_data = signal.filtfilt(b, a, data)
    
    return filtered_data

def check_required_columns(df: pd.DataFrame, colunas_analise: list):
    """
    Verifica se as colunas necessárias existem no DataFrame.
    Retorna True se todas existirem, False caso contrário.
    """
    colunas_faltantes = []
    colunas_calculadas = ['Alpha', 'rho_g', 'J Ar corrigido']
    
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

def apply_fft(df: pd.DataFrame, coluna: str, output_dir: str, base_name: str, freq_corte: float, freq_min_plot: float, freq_max_plot: float):
    """
    Aplica a FFT em uma coluna específica do DataFrame e plota os resultados.
    Inclui aplicação de filtro passa-baixa (se freq_corte não for None).
    
    Args:
        df (pd.DataFrame): DataFrame com os dados
        coluna (str): Nome da coluna para análise
        output_dir (str): Diretório para salvar os resultados
        base_name (str): Nome base para os arquivos de saída
        freq_corte (float or None): Frequência de corte do filtro passa-baixa em Hz (None = sem filtragem)
        freq_min_plot (float): Frequência mínima para o plot da FFT em Hz
        freq_max_plot (float): Frequência máxima para o plot da FFT em Hz
    """
    # Obtém os dados da coluna
    y_data = df[coluna].values
    time = df['X_Value'].values
    
    # Calcula o intervalo de tempo entre amostras e frequência de amostragem
    dt = np.mean(np.diff(time))
    fs = 1/dt
    
    # Aplica filtro passa-baixa
    y_filtered = apply_lowpass_filter(y_data, fs, freq_corte)
    
    # Aplica a FFT no sinal original
    n = len(y_data)
    fft_result_original = np.fft.fft(y_data)
    freqs = np.fft.fftfreq(n, dt)
    magnitude_original = np.abs(fft_result_original)
    
    # Aplica a FFT no sinal filtrado
    fft_result_filtered = np.fft.fft(y_filtered)
    magnitude_filtered = np.abs(fft_result_filtered)
    
    # Filtra frequências para o plot
    positive_freq_mask = freqs > 0
    freq_plot_mask = (freqs >= freq_min_plot) & (freqs <= freq_max_plot) & positive_freq_mask
    
    # Plota os resultados
    plt.figure(figsize=(20, 12))
    
    # Time series - original vs filtered
    plt.subplot(3, 2, 1)
    plt.plot(time, y_data, 'b-', alpha=0.7, label='Original signal', linewidth=0.8)
    if freq_corte is not None:
        plt.plot(time, y_filtered, 'r-', label=f'Filtered signal (fc={freq_corte} Hz)', linewidth=1.2)
    else:
        plt.plot(time, y_filtered, 'r-', label='Signal (no filter)', linewidth=1.2)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Time Series - {coluna.split("/")[-1].replace("_", " ")}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Time series zoom (first 2 seconds)
    plt.subplot(3, 2, 2)
    mask_zoom = time <= 2.0
    plt.plot(time[mask_zoom], y_data[mask_zoom], 'b-', alpha=0.7, label='Original', linewidth=0.8)
    if freq_corte is not None:
        plt.plot(time[mask_zoom], y_filtered[mask_zoom], 'r-', label=f'Filtered (fc={freq_corte} Hz)', linewidth=1.2)
    else:
        plt.plot(time[mask_zoom], y_filtered[mask_zoom], 'r-', label='Signal (no filter)', linewidth=1.2)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Zoom - First 2 seconds')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Frequency spectrum - original
    plt.subplot(3, 2, 3)
    plt.plot(freqs[freq_plot_mask], magnitude_original[freq_plot_mask], 'b-', linewidth=0.8)
    if freq_corte is not None:
        plt.axvline(x=freq_corte, color='r', linestyle='--', alpha=0.7, label=f'Cutoff frequency ({freq_corte} Hz)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Original Spectrum')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(freq_min_plot, freq_max_plot)
    
    # Frequency spectrum - filtered
    plt.subplot(3, 2, 4)
    plt.plot(freqs[freq_plot_mask], magnitude_filtered[freq_plot_mask], 'r-', linewidth=0.8)
    if freq_corte is not None:
        plt.axvline(x=freq_corte, color='r', linestyle='--', alpha=0.7, label=f'Cutoff frequency ({freq_corte} Hz)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    if freq_corte is not None:
        plt.title('Filtered Spectrum')
    else:
        plt.title('Spectrum (no filter)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(freq_min_plot, freq_max_plot)
    
    # Spectra comparison
    plt.subplot(3, 2, 5)
    plt.plot(freqs[freq_plot_mask], magnitude_original[freq_plot_mask], 'b-', alpha=0.7, label='Original', linewidth=0.8)
    plt.plot(freqs[freq_plot_mask], magnitude_filtered[freq_plot_mask], 'r-', label='Filtered' if freq_corte is not None else 'Signal', linewidth=0.8)
    if freq_corte is not None:
        plt.axvline(x=freq_corte, color='g', linestyle='--', alpha=0.7, label=f'fc = {freq_corte} Hz')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Spectra Comparison')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(freq_min_plot, freq_max_plot)

    # Frequências dominantes (maior energia)
    freqs_pos = freqs[freq_plot_mask]
    mag_orig_pos = magnitude_original[freq_plot_mask]
    mag_filt_pos = magnitude_filtered[freq_plot_mask]
    if len(freqs_pos) > 0:
        idx_dom_orig = int(np.argmax(mag_orig_pos))
        idx_dom_filt = int(np.argmax(mag_filt_pos))
        f_dom_orig = float(freqs_pos[idx_dom_orig])
        f_dom_filt = float(freqs_pos[idx_dom_filt])
        # imprime no terminal para indicar frequência de passagem (pistonado)
        print(f"\nFrequência dominante (maior energia) - original: {f_dom_orig:.6f} Hz")
        print(f"Frequência dominante (maior energia) - filtrado: {f_dom_filt:.6f} Hz")
        # destaca nos gráficos
        plt.subplot(3, 2, 3)
        plt.plot([f_dom_orig], [mag_orig_pos[idx_dom_orig]], 'ko', label=rf'Slug freq. = {f_dom_orig:.2f} Hz')
        plt.legend()
        plt.subplot(3, 2, 4)
        plt.plot([f_dom_filt], [mag_filt_pos[idx_dom_filt]], 'ko', label=rf'Slug freq. = {f_dom_filt:.2f} Hz')
        plt.legend()
    
    # Filter response
    plt.subplot(3, 2, 6)
    if freq_corte is not None:
        nyquist = fs / 2
        normalized_cutoff = freq_corte / nyquist
        b, a = signal.butter(4, normalized_cutoff, btype='low', analog=False)
        w, h = signal.freqz(b, a, worN=8000)
        freq_response = 0.5*fs*w/np.pi
        mask_response = (freq_response >= freq_min_plot) & (freq_response <= freq_max_plot)
        plt.plot(freq_response[mask_response], np.abs(h)[mask_response], 'g-', linewidth=2, label='Filter response')
        plt.axvline(x=freq_corte, color='r', linestyle='--', alpha=0.7, label=f'fc = {freq_corte} Hz')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.title('Butterworth Filter Response')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xlim(freq_min_plot, freq_max_plot)
    else:
        plt.text(0.5, 0.5, 'No filter applied', ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.title('Filter Response')
        plt.grid(True, alpha=0.3)
        plt.xlim(freq_min_plot, freq_max_plot)
    
    plt.tight_layout()
    
    # Cria diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Simplifica o nome da coluna para o arquivo
    coluna_simples = coluna.replace("/'Untitled'/", "").replace("'", "").replace("/", "_")
    
    # Salva o gráfico
    if freq_corte is not None:
        output_path = os.path.join(output_dir, f"fft_filtered_{coluna_simples}_{base_name}_fc{freq_corte}Hz.png")
    else:
        output_path = os.path.join(output_dir, f"fft_{coluna_simples}_{base_name}_nofilter.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Salva os dados da FFT em arquivos separados
    fft_data_original = pd.DataFrame({
        'Frequência (Hz)': freqs[freq_plot_mask],
        'Magnitude_Original': magnitude_original[freq_plot_mask]
    })
    
    fft_data_filtered = pd.DataFrame({
        'Frequência (Hz)': freqs[freq_plot_mask],
        'Magnitude_Filtrado': magnitude_filtered[freq_plot_mask]
    })
    
    # Salva dados originais
    output_file_original = os.path.join(output_dir, f"fft_original_{coluna_simples}_{base_name}.txt")
    fft_data_original.to_csv(output_file_original, sep='\t', index=False)
    
    # Salva dados filtrados
    if freq_corte is not None:
        output_file_filtered = os.path.join(output_dir, f"fft_filtered_{coluna_simples}_{base_name}_fc{freq_corte}Hz.txt")
    else:
        output_file_filtered = os.path.join(output_dir, f"fft_{coluna_simples}_{base_name}_nofilter.txt")
    fft_data_filtered.to_csv(output_file_filtered, sep='\t', index=False)
    
    # Salva dados combinados
    fft_data_combined = pd.DataFrame({
        'Frequência (Hz)': freqs[freq_plot_mask],
        'Magnitude_Original': magnitude_original[freq_plot_mask],
        'Magnitude_Filtrado': magnitude_filtered[freq_plot_mask],
        'Diferença': magnitude_original[freq_plot_mask] - magnitude_filtered[freq_plot_mask]
    })
    
    if freq_corte is not None:
        output_file_combined = os.path.join(output_dir, f"fft_comparison_{coluna_simples}_{base_name}_fc{freq_corte}Hz.txt")
    else:
        output_file_combined = os.path.join(output_dir, f"fft_comparison_{coluna_simples}_{base_name}_nofilter.txt")
    fft_data_combined.to_csv(output_file_combined, sep='\t', index=False)
    
    return fft_data_combined

def extract_info_from_filename(filename: str):
    """
    Extrai informações do nome do arquivo experimental.
    Se começar com 'V', desloca a leitura em uma casa (ponto de validação).
    Formato esperado: [V]XXX##ID## onde:
    - X: letra indicando o fluido (A:Air, W:Water, O:Oil, S:SF6)
    - #: número indicando a inclinação em graus
    - ID: identificador do ponto experimental
    """
    fluid_map = {
        'A': 'Air',
        'W': 'Water',
        'O': 'Oil',
        'S': 'SF6',
        'D': 'Dense Fluid'
    }
    direction_map = {
        'H': 'Horizontal',
        'U': 'Upward',
        'D': 'Downward'
    }
    base_name = os.path.splitext(os.path.basename(filename))[0]
    
    offset = 1 if base_name[0] == 'V' else 0
    is_validation = base_name[0] == 'V'
    fluid_1 = fluid_map.get(base_name[0+offset], 'Unknown')
    fluid_2 = fluid_map.get(base_name[1+offset], 'Unknown')
    direction = direction_map.get(base_name[2+offset], 'Unknown')
    theta = int(base_name[3+offset:5+offset])
    ID = base_name[5+offset:]
    
    return fluid_1, fluid_2, direction, theta, ID, is_validation

if __name__ == "__main__":
    # Lê o arquivo
    df, data_teste = read_file(file_path)
    
    # Verifica se as colunas necessárias existem
    if not check_required_columns(df, colunas_analise):
        sys.exit(1)
    
    # Obtém o diretório e nome base do arquivo original
    output_dir = os.path.dirname(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    # Extrai informações do nome do arquivo
    fluid_1, fluid_2, direction, theta, ID, is_validation = extract_info_from_filename(file_path)
    print(f"\nInformações extraídas do nome do arquivo:")
    print(f"Fluido 1: {fluid_1}")
    print(f"Fluido 2: {fluid_2}")
    print(f"Direção: {direction}")
    print(f"Inclinação (theta): {theta}°")
    print(f"ID do ponto: {ID}")
    print(f"Ponto de validação: {'Sim' if is_validation else 'Não'}")
    
    # Mostra as colunas disponíveis
    print("\nColunas disponíveis para análise:")
    for i, col in enumerate(colunas_analise, 1):
        print(f"{i}. {col[0]} ({col[1]})")
    
    # Pergunta qual coluna analisar
    while True:
        try:
            escolha = int(input("\nEscolha o número da coluna para análise FFT: "))
            if 1 <= escolha <= len(colunas_analise):
                coluna_escolhida = colunas_analise[escolha-1][0]
                print(f"\nColuna escolhida: {coluna_escolhida}")
                break
            else:
                print(f"Por favor, escolha um número entre 1 e {len(colunas_analise)}")
        except ValueError:
            print("Entrada inválida. Digite um número válido.")
    
    # Mostra informações sobre o filtro
    if freq_corte is not None:
        print(f"\nFiltro passa-baixa configurado:")
        print(f"Frequência de corte: {freq_corte} Hz")
    else:
        print(f"\nNenhum filtro será aplicado (freq_corte = None)")
    
    print(f"Faixa de frequência para plotar FFT: {freq_min_plot} - {freq_max_plot} Hz")
    
    # Calcula frequência de amostragem para mostrar informações
    dt = np.mean(np.diff(df['X_Value']))
    fs = 1/dt
    print(f"Frequência de amostragem: {fs:.2f} Hz")
    print(f"Frequência de Nyquist: {fs/2:.2f} Hz")
    
    # Aplica a FFT com filtro
    fft_data = apply_fft(df, coluna_escolhida, output_dir, base_name, freq_corte, freq_min_plot, freq_max_plot)
    
    print("\nAnálise FFT concluída!")
    coluna_simples = coluna_escolhida.replace("/'Untitled'/", "").replace("'", "").replace("/", "_")
    if freq_corte is not None:
        print(f"Gráfico salvo em: {os.path.join(output_dir, f'fft_filtered_{coluna_simples}_{base_name}_fc{freq_corte}Hz.png')}")
        print(f"Dados filtrados salvos em: {os.path.join(output_dir, f'fft_filtered_{coluna_simples}_{base_name}_fc{freq_corte}Hz.txt')}")
        print(f"Dados comparativos salvos em: {os.path.join(output_dir, f'fft_comparison_{coluna_simples}_{base_name}_fc{freq_corte}Hz.txt')}")
    else:
        print(f"Gráfico salvo em: {os.path.join(output_dir, f'fft_{coluna_simples}_{base_name}_nofilter.png')}")
        print(f"Dados salvos em: {os.path.join(output_dir, f'fft_{coluna_simples}_{base_name}_nofilter.txt')}")
        print(f"Dados comparativos salvos em: {os.path.join(output_dir, f'fft_comparison_{coluna_simples}_{base_name}_nofilter.txt')}")
    print(f"Dados originais salvos em: {os.path.join(output_dir, f'fft_original_{coluna_simples}_{base_name}.txt')}") 