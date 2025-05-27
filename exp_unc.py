import uncertainties as unc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from CoolProp.CoolProp import PropsSI

####################################################################################################################################################
#                                            INPUTS
####################################################################################################################################################
file_path = 'example/AWD45/AWD45ID01/AWD45ID01' #Insira o caminho do arquivo a ser analisado
variavel_criterio = 'J Ar'          # Escolha a variável de critério para realizar o janelamento - Digite o nome exato da coluna para análise

theta = 45      # Inclinação da plataforma
L = 1.7         # m comprimento entre as tomadas de diferencial de pressão
g = 9.81        # m/s² 

sensor_Yokogawa = 'PDT-M-0101D-30Kpa_mA'
rho_s = 962     # kg/m³
direction ='Downward'
### Colunas de interesse -> Insira o nome das colunas a plotar e avaliar do arquivo .dat
# Lista de colunas para análise: [nome_coluna, apelido, unidade]
colunas_analise = [
    
    ['PDT-M-0101-40kPa_mA', r'|\delta P_{40\,kPa}|', r'[Pa]'],
    ['PDT-M-0101D-30Kpa_mA', r'|\delta P_{30\,kPa}|', r'[Pa]'],
    ['Alpha', r'\alpha', r''],
    ['J Agua', r'J_{water}', r'[m/s]'],
    ['J Ar', r'J_{air}', r'[m/s]'],
    ['FT-A-0303', r'Q_{air}', r'[m³/h]'],
    ['PIT-M-0101', r'Gauge\ Pressure', r'[Bar]'],
    ['TIT-M-0101', r'Temperature', r'[°C]'],
    ['rho_g', r'\rho_{air}', r'[kg/m³]']
]

### Cálculo de propagação de incerteza ###
# Valores de calibração do densitômetro
I_g = 253892                     # Insira a intensidade padrão para o gás (Calibração do densitômetro)
I_f = 151174                      # Insira a intensidade padrão para o líquido (Calibração do densitômetro)
#
p_PIT_M_0101 = 0.2                  # Insira a precisão de medição do sensor de pressão
P_PDT_M_0101_40kPa = 0.1            # Insira a precisão de medição do diferencial de pressão de 40kPa
p_PDT_M_0101D_30Kpa = 0.055         # Insira a precisão de medição do diferencial de pressão de 30kPa  0.055% do span
span_yokogawa = 30E3 #Pa
p_PDT_M_0101B_10kPa = 0.1           # Insira a precisão de medição do diferencial de pressão de 10kPa
p_PDT_M_0101C_3kPa = 0.1            # Insira a precisão de medição do diferencial de pressão de 3kPa
p_TIT_M_0101 = 0.1                  # Insira a precisão de medição do sensor de temperatura
E_Densitometro = 0.1
E_J_Agua = 0.1                        # Insira o erro de medição do sensor de vazão
E_J_Ar = 0.1                          # Insira o erro de medição do sensor de vazão
####################################################################################################################################################
#       '                                   END INPUTS
####################################################################################################################################################

def find_min_std_window(df, column_name, min_window_size, max_window_size):
    """
    Encontra a janela de tempo com menor desvio padrão para uma coluna específica,
    testando diferentes tamanhos de janela dentro do intervalo especificado.
    Se min_window_size = max_window_size, usa um tamanho fixo de janela.
    
    Args:
        df (pandas.DataFrame): DataFrame com os dados
        column_name (str): Nome da coluna a ser analisada
        min_window_size (float): Tamanho mínimo da janela em segundos
        max_window_size (float): Tamanho máximo da janela em segundos
        
    Returns:
        tuple: (índice inicial, índice final, desvio padrão mínimo, tamanho ótimo da janela)
    """
    if column_name not in df.columns:
        raise ValueError(f"Coluna '{column_name}' não encontrada no DataFrame")
    
    if min_window_size > max_window_size:
        raise ValueError("O tamanho mínimo da janela deve ser menor ou igual ao tamanho máximo")
    
    min_std = float('inf')
    best_start_idx = 0
    best_end_idx = 0
    best_window_size = 0
    
    # Se min = max, usa um tamanho fixo de janela
    if min_window_size == max_window_size:
        window_size = min_window_size
        # Percorre o DataFrame procurando a janela com menor desvio padrão
        for i in range(len(df)):
            # Encontra o índice final da janela que corresponde ao tempo inicial + window_size
            start_time = df['X_Value'].iloc[i]
            end_time = start_time + window_size
            
            # Encontra o índice do último ponto que está dentro da janela
            end_idx = df[df['X_Value'] <= end_time].index[-1]
            
            # Se a janela não tiver o tamanho mínimo necessário, pula para o próximo ponto
            if end_idx - i < 2:  # Pelo menos 2 pontos para calcular desvio padrão
                continue
                
            # Calcula o desvio padrão para a janela atual
            window = df[column_name].iloc[i:end_idx+1]
            current_std = window.std()
            
            # Verifica se o tamanho real da janela está próximo do desejado (com margem de 1%)
            actual_window_size = df['X_Value'].iloc[end_idx] - start_time
            if abs(actual_window_size - window_size) > window_size * 0.01:
                continue
            
            if current_std < min_std:
                min_std = current_std
                best_start_idx = i
                best_end_idx = end_idx
                best_window_size = window_size
    else:
        # Testa diferentes tamanhos de janela
        for window_size in np.arange(min_window_size, max_window_size + 1, 1):
            # Percorre o DataFrame procurando a janela com menor desvio padrão
            for i in range(len(df)):
                # Encontra o índice final da janela que corresponde ao tempo inicial + window_size
                start_time = df['X_Value'].iloc[i]
                end_time = start_time + window_size
                
                # Encontra o índice do último ponto que está dentro da janela
                end_idx = df[df['X_Value'] <= end_time].index[-1]
                
                # Se a janela não tiver o tamanho mínimo necessário, pula para o próximo ponto
                if end_idx - i < 2:  # Pelo menos 2 pontos para calcular desvio padrão
                    continue
                    
                # Calcula o desvio padrão para a janela atual
                window = df[column_name].iloc[i:end_idx+1]
                current_std = window.std()
                
                # Verifica se o tamanho real da janela está próximo do desejado (com margem de 1%)
                actual_window_size = df['X_Value'].iloc[end_idx] - start_time
                if abs(actual_window_size - window_size) > window_size * 0.01:
                    continue
                
                if current_std < min_std:
                    min_std = current_std
                    best_start_idx = i
                    best_end_idx = end_idx
                    best_window_size = window_size
    
    if min_std == float('inf'):
        raise ValueError(f"Não foi possível encontrar uma janela válida entre {min_window_size} e {max_window_size} segundos")
    
    return best_start_idx, best_end_idx + 1, min_std, best_window_size

def read_file(file_path):
    """
    Lê o arquivo e retorna um DataFrame do pandas.
    Os dados são lidos a partir do segundo ***End_of_Header***.
    Os nomes das colunas são lidos da linha após o segundo ***End_of_Header***.
    
    Args:
        file_path (str): Caminho para o arquivo
        
    Returns:
        tuple: (DataFrame com os dados, data do teste experimental)
    """
    # Lê o arquivo ignorando as primeiras linhas do cabeçalho
    # O cabeçalho termina com o segundo ***End_of_Header***
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Procura a data do teste experimental no cabeçalho
    data_teste = None
    primeiro_header_end = False
    
    for line in lines:
        if '***End_of_Header***' in line:
            primeiro_header_end = True
            continue
            
        if not primeiro_header_end:
            if 'Date' in line:
                try:
                    # Extrai a data da linha
                    data = line.strip().split('Date')[1].strip()
                    # Converte para o formato DD/MM/AAAA
                    partes = data.split('/')
                    if len(partes) == 3:
                        data_teste = f"{partes[0]}/{partes[1]}/{partes[2]}"
                except:
                    pass
    
    # Encontra o índice onde o segundo cabeçalho termina
    header_count = 0
    header_end_idx = 0
    for i, line in enumerate(lines):
        if '***End_of_Header***' in line:
            header_count += 1
            if header_count == 2:
                header_end_idx = i + 1
                break
            
    # Lê os nomes das colunas da linha após o segundo ***End_of_Header***
    column_names = [name.strip() for name in lines[header_end_idx].strip().split('	')]

    # Lê os dados usando pandas, pulando as linhas do cabeçalho
    df = pd.read_csv(file_path, 
                     sep='\t',  # Separador é tabulação
                     skiprows=header_end_idx+1,  # Pula as linhas do cabeçalho e a linha dos nomes
                     decimal=',',  # Separador decimal é vírgula
                     na_values=[''],  # Valores vazios são considerados NaN
                     encoding='utf-8',  # Codificação do arquivo
                     names=column_names)  # Usa os nomes das colunas lidos do arquivo
    
    return df, data_teste

def save_results(df, coluna_escolhida, start_idx, end_idx, min_std, media_janela, 
                min_window_size, max_window_size, best_window_size, file_path, data_teste, nomes=None, medias=None, desvios=None, uAs=None):
    """
    Salva os resultados da análise em um arquivo de saída.
    Agora inclui as estatísticas (média, desvio padrão, uA) de cada variável de interesse.
    
    Args:
        df (pandas.DataFrame): DataFrame com os dados originais
        coluna_escolhida (str): Nome da coluna usada como critério
        start_idx (int): Índice inicial da janela
        end_idx (int): Índice final da janela
        min_std (float): Desvio padrão mínimo encontrado
        media_janela (float): Média da janela
        min_window_size (float): Tamanho mínimo da janela configurado
        max_window_size (float): Tamanho máximo da janela configurado
        best_window_size (float): Tamanho ótimo da janela encontrado
        file_path (str): Caminho do arquivo original
        data_teste (str): Data do teste experimental
        nomes (list): Lista de nomes das variáveis de interesse
        medias (list): Lista de médias das variáveis
        desvios (list): Lista de desvios padrão das variáveis
        uAs (list): Lista de incertezas tipo A para cada variável
    """
    # Obtém o diretório e nome base do arquivo original
    diretorio = os.path.dirname(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Cria o nome do arquivo de saída no mesmo diretório
    output_file = os.path.join(diretorio, f"{base_name}_tratado.txt")
    
    # Obtém a data atual no formato DD/MM/AAAA
    data_atual = datetime.now().strftime('%d/%m/%Y')
    
    # Prepara o cabeçalho com as informações gerais
    header = [
        "***Resultados da Análise***",
        f"Data do teste experimental: {data_teste if data_teste else 'Não encontrada'}",
        f"Data tratamento: {data_atual}",
        f"Arquivo Original: {file_path}",
        f"Coluna Critério: {coluna_escolhida}",
        f"Tamanho Mínimo da Janela: {min_window_size:.1f} segundos",
        f"Tamanho Máximo da Janela: {max_window_size:.1f} segundos",
        f"Tamanho Ótimo da Janela: {best_window_size:.1f} segundos",
        f"Média da Janela: {media_janela:.4f}",
        f"Desvio Padrão: {min_std:.4f}",
        f"Tempo Inicial: {df['X_Value'].iloc[start_idx]:.2f} segundos",
        f"Tempo Final: {df['X_Value'].iloc[end_idx-1]:.2f} segundos",
        f"Número de Pontos: {end_idx - start_idx}",
    ]
    
    # Adiciona a seção de estatísticas se fornecidas
    if nomes is not None and medias is not None and desvios is not None and uAs is not None:
        header.append("***Estatísticas das variáveis na janela***")
        header.append("Variável: Média | Desvio padrão | Incerteza tipo A")
        for nome, media, desvio, uA in zip(nomes, medias, desvios, uAs):
            header.append(f"{nome}: {media:.6f} | {desvio:.6f} | {uA:.6f}")
    
    header += [
        "***Dados da Janela***",
        "***End_of_Header***"
    ]
    
    # Seleciona os dados da janela para todas as colunas
    window_data = df.iloc[start_idx:end_idx]
    
    # Salva o arquivo
    with open(output_file, 'w', encoding='utf-8') as f:
        # Escreve o cabeçalho
        f.write('\n'.join(header))
        f.write('\n')
        
        # Escreve os nomes das colunas
        f.write('\t'.join(df.columns))
        f.write('\n')
        
        # Escreve os dados
        for _, row in window_data.iterrows():
            f.write('\t'.join([f"{val:.6f}" if isinstance(val, (int, float)) else str(val) 
                             for val in row]))
            f.write('\n')
    
    print(f"\nResultados salvos no arquivo: {output_file}")

def plot_time_series(df, colunas, output_dir, base_name):
    """
    Plota as séries temporais em subplots organizados em duas colunas e salva a imagem.
    colunas: lista de listas [nome_coluna, apelido, unidade]
    """
    n_colunas = 3
    n_linhas = (len(colunas) + n_colunas - 1) // n_colunas
    
    fig, axs = plt.subplots(n_linhas, n_colunas, figsize=(16, 3.8*n_linhas), constrained_layout=True)
    fig.suptitle('Séries Temporais das Variáveis', fontsize=18, y=1.03)
    
    for idx, coluna_info in enumerate(colunas):
        if isinstance(coluna_info, (list, tuple)) and len(coluna_info) == 3:
            nome_coluna, apelido, unidade = coluna_info
        else:
            nome_coluna = coluna_info
            apelido = nome_coluna
            unidade = ''
        linha = idx // n_colunas
        col = idx % n_colunas
        ax = axs[linha, col] if n_linhas > 1 else axs[col]
        # Se for série PDT, plota o valor absoluto
        if nome_coluna.startswith('PDT-'):
            y_data = np.abs(df[nome_coluna])
            ax.plot(df['X_Value'], y_data, 'b-', alpha=0.8)
        else:
            y_data = df[nome_coluna]
            ax.plot(df['X_Value'], y_data, 'b-', alpha=0.8)
        # Linha da média da série completa
        media_serie = y_data.mean()
        ax.axhline(y=media_serie, color='g', linestyle='--', label=f'Mean: {media_serie:.4f}')
        ax.legend(fontsize=8, loc='upper right')
        if linha == n_linhas - 1:
            ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel(f"${apelido}$ {unidade}", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=9)
        # Ajuste automático do eixo y com margem de 10%
        y_min = y_data.min()
        y_max = y_data.max()
        # margem = 0.40 * (y_max - y_min) if y_max != y_min else 1
        margem_max = 1.01*y_max
        margem_min = 0.99*y_min
        ax.set_ylim(margem_min, margem_max)
    for idx in range(len(colunas), n_linhas * n_colunas):
        linha = idx // n_colunas
        col = idx % n_colunas
        fig.delaxes(axs[linha, col])
    output_path = os.path.join(output_dir, f"series_full-{base_name}.png")
    plt.savefig(output_path)
    print(f"\nGráfico das séries temporais salvo em: {output_path}")
    plt.show()
    plt.close(fig)

def plot_windows(df, colunas, start_idx, end_idx, best_window_size, output_dir, base_name):
    """
    Plota as janelas de todas as variáveis em subplots organizados em duas colunas e salva a imagem.
    colunas: lista de listas [nome_coluna, apelido, unidade]
    """
    n_colunas = 3
    n_linhas = (len(colunas) + n_colunas - 1) // n_colunas
    fig, axs = plt.subplots(n_linhas, n_colunas, figsize=(16, 3.8*n_linhas), constrained_layout=True)
    fig.suptitle(f'Janelas das Variáveis (Tamanho: {best_window_size:.1f}s)', fontsize=18, y=1.03)
    for idx, (coluna, apelido, unidade) in enumerate(colunas):
        linha = idx // n_colunas
        col = idx % n_colunas
        ax = axs[linha, col] if n_linhas > 1 else axs[col]
        # Se for série PDT, plota o valor absoluto
        nome_coluna = coluna  # coluna já é o nome real da coluna
        if nome_coluna.startswith('PDT-'):
            y_data = np.abs(df[nome_coluna].iloc[start_idx:end_idx])
            ax.plot(df['X_Value'], np.abs(df[nome_coluna]), 'b-', alpha=0.3, label='Full Series (abs)')
            ax.plot(df['X_Value'].iloc[start_idx:end_idx], y_data, 'r-', alpha=0.8, label=f'Window = {best_window_size:.0f} s')
        else:
            y_data = df[nome_coluna].iloc[start_idx:end_idx]
            ax.plot(df['X_Value'], df[nome_coluna], 'b-', alpha=0.3, label='Full Series')
            ax.plot(df['X_Value'].iloc[start_idx:end_idx], y_data, 'r-', alpha=0.8, label=f'Window = {best_window_size:.0f} s')
        media_janela = y_data.mean()
        ax.axhline(y=media_janela, color='g', linestyle='--', label=f'Mean: {media_janela:.4f}')
        ax.legend(fontsize=8, loc='upper right')
        if linha == n_linhas - 1:
            ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel(f"${apelido}$ {unidade}", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=9)
        # Ajuste automático do eixo y com margem de 40% (igual ao plot_time_series)
        y_min = y_data.min()
        y_max = y_data.max()
        margem = 0.40 * (y_max - y_min) if y_max != y_min else 1
        ax.set_ylim(y_min - margem, y_max + margem)
    for idx in range(len(colunas), n_linhas * n_colunas):
        linha = idx // n_colunas
        col = idx % n_colunas
        fig.delaxes(axs[linha, col])
    output_path = os.path.join(output_dir, f"janelas-{base_name}.png")
    plt.savefig(output_path)
    print(f"\nGráfico das janelas salvo em: {output_path}")
    plt.close(fig)

def uncert_propagation(df, colunas, start_idx, end_idx, best_window_size):
    """
    Propaga as incertezas das variáveis para a variável critério.
    Para cada coluna de interesse, calcula a média, o desvio padrão e a incerteza estatística tipo A (padrão da média) na janela selecionada.
    Retorna listas com os resultados para uso posterior.
    """
    print("\nCálculo da média, desvio padrão e incerteza estatística tipo A para cada variável na janela:")
    n = end_idx - start_idx
    medias = []
    desvios = []
    uAs = []
    for coluna_info in colunas:
        nome_coluna = coluna_info[0]
        dados_janela = df[nome_coluna].iloc[start_idx:end_idx]
        media = dados_janela.mean()
        desvio = dados_janela.std(ddof=1)
        uA = desvio / (n ** 0.5)
        medias.append(media)
        desvios.append(desvio)
        uAs.append(uA)
        print(f"{nome_coluna:20} | Média: {media:.6f} | Desvio padrão: {desvio:.6f} | Incerteza tipo A: {uA:.6f}")
    return [c[0] for c in colunas], medias, desvios, uAs

def calc_alpha(df, start_idx, end_idx):
    """
    Determina a fração de vazio (alpha) na mistura a partir dos dados do densitômetro na janela selecionada.
    Retorna um DataFrame contendo a série temporal de Alpha na janela.
    """

    # Calcula a série temporal de Alpha na janela
    dados_densitometro = df['Densitometro'].iloc[start_idx:end_idx]
    alpha_series = np.log(dados_densitometro / I_f) / np.log(I_g / I_f)
    
    # Cria um DataFrame com X_Value e a série de Alpha
    alpha_df = pd.DataFrame({
        'X_Value': df['X_Value'].iloc[start_idx:end_idx],
        'Alpha': alpha_series
    })
    
    print("\nSérie temporal de Alpha calculada na janela.")
    # print(alpha_df.head()) # Opcional: mostrar as primeiras linhas do DataFrame
    
    return alpha_df # Retorna o DataFrame com a série de Alpha

def calc_frictional_pressure_gradient(df, colunas, start_idx, end_idx, best_window_size, alpha_series):
    """
    Calcula o gradiente de pressão friccional para cada variável na janela selecionada.
    Também encontra os índices das colunas que começam com 'PDT'.
    Calcula a série temporal da densidade do ar na janela usando a equação de estado.
    A pressão é lida como manométrica em bares e convertida para absoluta em Pascal.
    Armazena os resultados em um DataFrame dP_F_df e o retorna.
    """
    indices_pdt = [i for i, col in enumerate(colunas) if col[0].startswith("PDT")]

    # Obtém as séries temporais de pressão (PIT-M-0101) e temperatura (TIT-M-0101) na janela
    try:
        pressao_ar_bar = df['PIT-M-0101'].iloc[start_idx:end_idx]
        temp_ar_celsius = df['TIT-M-0101'].iloc[start_idx:end_idx]
        
        # Converte pressão manométrica de bar para Pascal (adicionando 1 bar atmosférico)
        pressao_ar_pa = (pressao_ar_bar + 1) * 1e5
        
        # Converte temperatura para Kelvin (assumindo que está em Celsius)
        temp_ar_k = temp_ar_celsius + 273.15
        
        # Cálculo da densidade do ar usando CoolProp
        rho_ar = [PropsSI('D', 'P', p, 'T', t, 'Air') for p, t in zip(pressao_ar_pa, temp_ar_k)]
        rho_ar = pd.Series(rho_ar, index=pressao_ar_pa.index)
        print("\nSérie temporal da densidade do ar calculada com CoolProp na janela.")
    
        # Cálculo da densidade da água usando CoolProp (assumindo pressão atmosférica)
        try:
            temp_agua_celsius = df['TIT-M-0101'].iloc[start_idx:end_idx]
            temp_agua_k = temp_agua_celsius + 273.15
            rho_agua = [PropsSI('D', 'T', t, 'P', 101325, 'Water') for t in temp_agua_k]
            rho_agua = pd.Series(rho_agua, index=temp_agua_celsius.index)
            print("Série temporal da densidade da água calculada com CoolProp na janela.")
        except Exception as e:
            print(f"Erro ao calcular densidade da água com CoolProp: {e}")
            rho_agua = 1000  # fallback

    except KeyError as e:
        print(f"Erro ao encontrar colunas para cálculo da densidade do ar: {e}")
        rho_ar = None # Garante que densidade_ar seja None se ocorrer erro

    # Inicializa o DataFrame para armazenar os resultados de dP_F
    dP_F_df = pd.DataFrame()

    # Calcula e armazena as séries temporais de dP_F para cada coluna PDT
    for i in indices_pdt:
        coluna_pdt_nome = colunas[i][0]  # Corrigido: pega só o nome real da coluna
        # Assumindo que alpha e theta são constantes para este cálculo
        # Lembre-se de que np.sin espera radianos
        theta_rad = np.deg2rad(theta) # Converte theta para radianos
        # Calcula a série temporal dP_F usando a fórmula fornecida e a série alpha_series
        delta_p_prime = df[coluna_pdt_nome].iloc[start_idx:end_idx]
        print(coluna_pdt_nome)
        if coluna_pdt_nome == sensor_Yokogawa:
            print('passou aqui')
            rho_tubbing = rho_s
        else:
            rho_tubbing = rho_agua
        
        termo_gravitacional = ((1-alpha_series)*rho_agua + alpha_series*rho_ar - rho_tubbing) * g * np.sin(theta_rad)   # Pa/m
        print('TERMO GRAVITACIONAL:',np.mean(termo_gravitacional))
        print('GRADIENTE DE PRESSÃO:',np.mean(delta_p_prime / L))
        
        if direction == 'Upward':
            dP_F_over_dz_series = (delta_p_prime / L) - termo_gravitacional                 # Calcula o gradiente de pressão friccional
        elif direction == 'Downward':
            print('Estamos em Downward')
            dP_F_over_dz_series = -(delta_p_prime / L) + termo_gravitacional                            # Calcula o gradiente de pressão friccional
        
        # Armazena a série calculada no DataFrame dP_F_df com o nome da coluna original
        dP_F_df[coluna_pdt_nome] = dP_F_over_dz_series
        print('GRADIENTE FRICCIONAL: ', np.mean(dP_F_over_dz_series))

    # Agora dP_F_df contém as séries temporais de dP_F/dz para cada coluna PDT
    print("\nDataFrame dP_F_df criado com as séries temporais de gradiente de pressão friccional para colunas PDT.")
    # print(dP_F_df.head()) # Opcional: mostrar as primeiras linhas do DataFrame

    return dP_F_df # Retorna o DataFrame

# Exemplo de uso:
if __name__ == "__main__":
    
    # file_path = "example/FSC2_Agua_Ar_Downward_45_graus/ID15"
    df, data_teste = read_file(file_path)    
    print("Dimensões do DataFrame:", df.shape)
    print("\nNomes das colunas:")
    print(df.columns.tolist())
    
    # Obtém o diretório e nome base do arquivo original para salvar as imagens
    output_dir = os.path.dirname(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Checa e calcula Alpha se necessário
    if any(col[0] == 'Alpha' for col in colunas_analise):
        alpha_df_full = calc_alpha(df, 0, len(df))
        df['Alpha'] = alpha_df_full['Alpha'].values

    # Checa e calcula rho_g se necessário
    if any(col[0] == 'rho_g' for col in colunas_analise):
        try:
            pressao_ar_bar_full = df['PIT-M-0101']
            temp_ar_celsius_full = df['TIT-M-0101']
            pressao_ar_pa_full = (pressao_ar_bar_full + 1) * 1e5
            temp_ar_k_full = temp_ar_celsius_full + 273.15
            rho_g_full = [PropsSI('D', 'P', p, 'T', t, 'Air') for p, t in zip(pressao_ar_pa_full, temp_ar_k_full)]
            df['rho_g'] = rho_g_full
        except Exception as e:
            print(f"Erro ao calcular densidade do ar para toda a série: {e}")
            df['rho_g'] = (pressao_ar_pa_full) / (8.314 * temp_ar_k_full)

    # Cálculo do gradiente de pressão friccional para toda a série
    dP_F_df_full = calc_frictional_pressure_gradient(df, colunas_analise, 0, len(df), len(df), alpha_df_full['Alpha'])
    for col in dP_F_df_full.columns:
        df[f'dP_F/dz {col}'] = dP_F_df_full[col].values

    # Filtra colunas para plotar apenas as que existem no DataFrame
    colunas_analise_filtradas = [col for col in colunas_analise if col[0] in df.columns]
    if len(colunas_analise_filtradas) < len(colunas_analise):
        print("Atenção: Algumas colunas de análise não existem no DataFrame e não serão plotadas.")
        for col in colunas_analise:
            if col[0] not in df.columns:
                print(f"Coluna ausente: {col[0]}")

    print("\nVisualizando as séries temporais das variáveis disponíveis...")
    plot_time_series(df, colunas_analise_filtradas, output_dir, base_name)
    
    # Mostra as colunas disponíveis
    print("\nColunas disponíveis para análise:")
    for i, col in enumerate(colunas_analise_filtradas, 1):
        print(f"{i}. {col}")
    
    # Não é mais necessário perguntar ao usuário a variável de critério
    coluna_escolhida = variavel_criterio

    # Pergunta ao usuário o tipo de janela desejada
    print("\nComo deseja definir a janela de análise?")
    print("1. Definir manualmente (tempo inicial e tempo final)")
    print("2. Encontrar janela ótima pelo tamanho (automático)")
    while True:
        escolha_janela = input("Digite 1 para manual ou 2 para automático: ").strip()
        if escolha_janela in ['1', '2']:
            break
        print("Opção inválida. Digite 1 ou 2.")

    if escolha_janela == '1':
        # Janela manual
        while True:
            try:
                tempo_inicial = float(input("Digite o tempo inicial da janela (em segundos): "))
                tempo_final = float(input("Digite o tempo final da janela (em segundos): "))
                if tempo_final > tempo_inicial:
                    break
                else:
                    print("O tempo final deve ser maior que o inicial.")
            except ValueError:
                print("Entrada inválida. Digite valores numéricos.")
        # Encontrar os índices correspondentes
        start_idx = df[df['X_Value'] >= tempo_inicial].index[0]
        end_idx = df[df['X_Value'] <= tempo_final].index[-1] + 1  # +1 para incluir o último ponto
        min_std = df[coluna_escolhida].iloc[start_idx:end_idx].std()
        best_window_size = df['X_Value'].iloc[end_idx-1] - df['X_Value'].iloc[start_idx]
        media_janela = df[coluna_escolhida].iloc[start_idx:end_idx].mean()
        print(f"\nJanela manual selecionada: {tempo_inicial:.2f}s a {tempo_final:.2f}s")
        print(f"Média da janela: {media_janela:.4f}")
        print(f"Desvio padrão: {min_std:.4f}")
        print(f"Tamanho da janela: {best_window_size:.1f} segundos")
        min_window_size = best_window_size
        max_window_size = best_window_size
    else:
        # Janela ótima automática (fluxo padrão)
        # Mostra as colunas disponíveis para escolha
        print("\nColunas disponíveis para análise:")
        for i, col in enumerate(colunas_analise_filtradas, 1):
            print(f"{i}. {col[0]} ({col[1]})")
        
        # Pergunta qual variável usar como critério
        while True:
            try:
                escolha = int(input("\nEscolha o número da variável para usar como critério de janelamento: "))
                if 1 <= escolha <= len(colunas_analise_filtradas):
                    coluna_escolhida = colunas_analise_filtradas[escolha-1][0]
                    print(f"\nVariável escolhida: {coluna_escolhida}")
                    break
                else:
                    print(f"Por favor, escolha um número entre 1 e {len(colunas_analise_filtradas)}")
            except ValueError:
                print("Entrada inválida. Digite um número válido.")

        while True:
            try:
                min_window_size = float(input("\nDigite o tamanho mínimo da janela em segundos: "))
                if min_window_size > 0:
                    break
                else:
                    print("O tamanho mínimo da janela deve ser maior que zero.")
            except ValueError:
                print("Entrada inválida. Por favor, digite um número válido.")
        while True:
            try:
                max_window_size = float(input("\nDigite o tamanho máximo da janela em segundos: "))
                if max_window_size >= min_window_size:
                    break
                else:
                    print("O tamanho máximo da janela deve ser maior ou igual ao tamanho mínimo.")
            except ValueError:
                print("Entrada inválida. Por favor, digite um número válido.")
        start_idx, end_idx, min_std, best_window_size = find_min_std_window(
            df, coluna_escolhida, min_window_size, max_window_size)
        media_janela = df[coluna_escolhida].iloc[start_idx:end_idx].mean()

    # Calcula e exibe a incerteza tipo A para cada variável na janela (e salva arrays)
    nomes, medias, desvios, uAs = uncert_propagation(df, colunas_analise_filtradas, start_idx, end_idx, best_window_size)
    
    # Salva os resultados em um arquivo
    save_results(df, coluna_escolhida, start_idx, end_idx, min_std, media_janela,
                min_window_size, max_window_size, best_window_size, file_path, data_teste,
                nomes=colunas_analise_filtradas, medias=medias, desvios=desvios, uAs=uAs)
    
    # Plota o gráfico da variável critério e salva a imagem
    print(f"\nPlotando a variável critério ({coluna_escolhida}) e a janela...")
    plt.figure(figsize=(15, 8))
    
    # Plota a série temporal completa
    plt.plot(df['X_Value'], df[coluna_escolhida], 'b-', label='Full Series', alpha=0.7)
    
    # Destaca a janela com menor desvio padrão
    plt.axvspan(df['X_Value'].iloc[start_idx], df['X_Value'].iloc[end_idx-1], alpha=0.3, color='red', label=f'Window = {best_window_size:.0f} s')
    
    # Adiciona a média como uma linha horizontal na janela
    plt.axhline(y=media_janela, color='g', linestyle='--', label=f'Mean: {media_janela:.4f}')
    
    # Configurações do gráfico
    # Buscar apelido e unidade da coluna escolhida
    apelido_escolhido = coluna_escolhida
    unidade_escolhida = ''
    for col in colunas_analise_filtradas:
        if isinstance(col, (list, tuple)) and col[0] == coluna_escolhida:
            apelido_escolhido = col[1]
            unidade_escolhida = col[2]
            break
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel(f"${apelido_escolhido}$ {unidade_escolhida}", fontsize=12)
    plt.title(f"Time Series of ${apelido_escolhido}$", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.xlim(df['X_Value'].min(), df['X_Value'].max())
    y_min = df[coluna_escolhida].min() * 0.95
    y_max = df[coluna_escolhida].max() * 1.05
    plt.ylim(y_min, y_max)
    plt.tight_layout()
    output_path_criterion = os.path.join(output_dir, f"{coluna_escolhida}_{base_name}_janela.png")
    plt.savefig(output_path_criterion)
    print(f"Gráfico da variável critério salvo em: {output_path_criterion}")
    plt.close(plt.gcf())
    
    # Plota as janelas de todas as variáveis e salva a imagem
    print("\nVisualizando as janelas de todas as variáveis...")
    plot_windows(df, colunas_analise_filtradas, start_idx, end_idx, best_window_size, output_dir, base_name)
    
    # Calcula e plota Alpha e salva a imagem
    print("\nCalculando e plotando a fração de vazio (Alpha) na janela...")
    alpha_df = calc_alpha(df, start_idx, end_idx)
    
    # Plota a série temporal de Alpha
    if alpha_df is not None and not alpha_df.empty:
        plt.figure(figsize=(15, 8))
        # Linha da média de alpha
        media_alpha = alpha_df['Alpha'].mean()
        plt.axhline(y=media_alpha, color='g', linestyle='--', label=f'Mean: {media_alpha:.4f}')
        plt.plot(alpha_df['X_Value'], alpha_df['Alpha'], label='Alpha')
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel(r'$\alpha$', fontsize=12)
        plt.title(r'Void Fraction ($\alpha$)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        
        plt.tight_layout()
        output_path_alpha = os.path.join(output_dir, f"alpha-{base_name}.png")
        plt.savefig(output_path_alpha)
        print(f"Gráfico Alpha salvo em: {output_path_alpha}")
        plt.close(plt.gcf()) # Fecha a figura atual
    else:
        print("Não foi possível calcular Alpha ou plotar.")
    
    # Calcula o gradiente de pressão friccional e plota e salva a imagem
    print("\nCalculando e plotando o gradiente de pressão friccional (dP_F/dz) para colunas PDT...")
    # Passa a série 'Alpha' do DataFrame alpha_df para a função (se alpha_df não for None)
    alpha_series = alpha_df['Alpha'] if alpha_df is not None else None
    dP_F_df = calc_frictional_pressure_gradient(df, colunas_analise_filtradas, start_idx, end_idx, best_window_size, alpha_series=alpha_series)
    
    # Plota as séries temporais de dP_F_df em um único gráfico
    if dP_F_df is not None and not dP_F_df.empty:
        plt.figure(figsize=(15, 8))
        for col in dP_F_df.columns:
            # Usa o X_Value original do DataFrame principal para o eixo x
            plt.plot(df['X_Value'].iloc[start_idx:end_idx], dP_F_df[col], label=col)
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel(r'$ \frac{\Delta P_{f}}{L}$ [Pa/m]', fontsize=12)
        plt.title(r'Frictional Pressure Gradient $\frac{\Delta P_{f}}{L}$', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()
        output_path_dpf = os.path.join(output_dir, f"dP_F_dz-{base_name}.png")
        plt.savefig(output_path_dpf)
        print(f"Gráfico dP_F/dz salvo em: {output_path_dpf}")
        plt.close(plt.gcf()) # Fecha a figura atual
    else:
        print("Não foi possível calcular o gradiente de pressão friccional para plotar.")
