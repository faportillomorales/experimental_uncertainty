import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import sys

####################################################################################################################################################
#                                            INPUTS
####################################################################################################################################################
file_path = 'C:/Users/User/Documents/LEMI/FSC2_pc/2. Air-Water Tests/2. Air-Water Tests/AWH00/AWH00P02/Data/AWH00P02' #Insira o caminho do arquivo a ser analisado NOTE: USE SEMPRE A BARRA NORMAL '/', SE ESTIVER INVERTIDA, MODIFIQUE-A

### Colunas de interesse -> Insira o nome das colunas a plotar e avaliar do arquivo .dat
# Lista de colunas para análise: [nome_coluna, apelido, unidade]
colunas_analise = [
    ['PDT-M-0101C-3kPa', r'\Delta P_{30\,kPa} / L', r'[Pa/m]'],
    ['PDT-M-0101-40kPa', r'\Delta P_{40\,kPa} / L', r'[Pa/m]'],
    ['Alpha', r'\alpha', r''],
    ['J Agua corrigido', r'J_{water}', r'[m/s]'],
    ['J Ar corrigido', r'J_{air}', r'[m/s]'],
    ['FT-A-0302', r'Q_{air}', r'[m³/h]'],
    ['PIT-M-0101', r'Gauge\ Pressure', r'[Bar]'],
    ['TIT-M-0101', r'Temperature', r'[°C]'],
    ['rho_g', r'\rho_{air}', r'[kg/m³]']
]

####################################################################################################################################################
#                                            END INPUTS
####################################################################################################################################################

def read_file(file_path: str):
    """
    Lê o arquivo e retorna um DataFrame do pandas.
    Os dados são lidos a partir do segundo ***End_of_Header***.
    Os nomes das colunas são lidos da linha após o segundo ***End_of_Header***.
    
    Args:
        file_path (str): Caminho para o arquivo
        
    Returns:
        tuple: (DataFrame com os dados, data do teste experimental)
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    data_teste = None
    primeiro_header_end = False
    
    for line in lines:
        if '***End_of_Header***' in line:
            primeiro_header_end = True
            continue
            
        if not primeiro_header_end:
            if 'Date' in line:
                try:
                    data = line.strip().split('Date')[1].strip()
                    partes = data.split('/')
                    if len(partes) == 3:
                        data_teste = f"{partes[0]}/{partes[1]}/{partes[2]}"
                except:
                    pass
    
    header_count = 0
    header_end_idx = 0
    for i, line in enumerate(lines):
        if '***End_of_Header***' in line:
            header_count += 1
            if header_count == 2:
                header_end_idx = i + 1
                break
            
    column_names = [name.strip() for name in lines[header_end_idx].strip().split('\t')]

    df = pd.read_csv(file_path, 
                     sep='\t',
                     skiprows=header_end_idx+1,
                     decimal=',',
                     na_values=[''],
                     encoding='utf-8',
                     names=column_names)
    
    df['J Ar corrigido'] = df['J Ar'] * (1 - 0.06675) 
    df['J Agua corrigido'] = df['J Agua'] * (1 - 0.06675) 
    
    return df, data_teste

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

def apply_fft(df: pd.DataFrame, coluna: str, output_dir: str, base_name: str):
    """
    Aplica a FFT em uma coluna específica do DataFrame e plota os resultados.
    
    Args:
        df (pd.DataFrame): DataFrame com os dados
        coluna (str): Nome da coluna para análise
        output_dir (str): Diretório para salvar os resultados
        base_name (str): Nome base para os arquivos de saída
    """
    # Obtém os dados da coluna
    y_data = df[coluna].values
    time = df['X_Value'].values
    
    # Calcula o intervalo de tempo entre amostras
    dt = np.mean(np.diff(time))
    
    # Aplica a FFT
    n = len(y_data)
    fft_result = np.fft.fft(y_data)
    freqs = np.fft.fftfreq(n, dt)
    
    # Calcula a magnitude do espectro
    magnitude = np.abs(fft_result)
    
    # Plota o sinal original
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.plot(time, y_data, 'b-', label='Sinal Original')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Sinal Original - {coluna}')
    plt.grid(True)
    plt.legend()
    
    # Plota o espectro de frequência
    plt.subplot(2, 1, 2)
    # Plota apenas a parte positiva do espectro
    positive_freq_mask = freqs > 0
    plt.plot(freqs[positive_freq_mask], magnitude[positive_freq_mask], 'r-')
    plt.xlabel('Frequência (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Espectro de Frequência')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Salva o gráfico
    output_path = os.path.join(output_dir, f"fft_analysis_{coluna}_{base_name}.png")
    plt.savefig(output_path)
    plt.close()
    
    # Salva os dados da FFT em um arquivo
    fft_data = pd.DataFrame({
        'Frequência (Hz)': freqs[positive_freq_mask],
        'Magnitude': magnitude[positive_freq_mask]
    })
    
    output_file = os.path.join(output_dir, f"fft_data_{coluna}_{base_name}.txt")
    fft_data.to_csv(output_file, sep='\t', index=False)
    
    return fft_data

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
    
    # Aplica a FFT
    fft_data = apply_fft(df, coluna_escolhida, output_dir, base_name)
    
    print("\nAnálise FFT concluída!")
    print(f"Gráfico salvo em: {os.path.join(output_dir, f'fft_analysis_{coluna_escolhida}_{base_name}.png')}")
    print(f"Dados da FFT salvos em: {os.path.join(output_dir, f'fft_data_{coluna_escolhida}_{base_name}.txt')}") 