import uncertainties as unc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from CoolProp.CoolProp import PropsSI
import sys

####################################################################################################################################################
#                                            INPUTS
####################################################################################################################################################
file_path = 'example/AWD45/AWD45P04/AWD45P04' #Insira o caminho do arquivo a ser analisado NOTE: USE SEMPRE A BARRA NORMAL '/', SE ESTIVER INVERTIDA, MODIFIQUE-A

L = 1.70         # m comprimento entre as tomadas de diferencial de pressão

# Valores de calibração do densitômetro IMPORTANTE
I_g = 252883                     # Insira a intensidade padrão para o gás (Calibração do densitômetro)
I_f = 151287                      # Insira a intensidade padrão para o líquido (Calibração do densitômetro)

sensor_Yokogawa = 'PDT-M-0101D-30Kpa_mA'
sensor_Endress = 'PDT-M-0101-40kPa_mA'

### Colunas de interesse -> Insira o nome das colunas a plotar e avaliar do arquivo .dat
# Lista de colunas para análise: [nome_coluna, apelido, unidade]
colunas_analise = [
    [sensor_Yokogawa, r'\Delta P_{40\,kPa} / L', r'[Pa/m]'],
    [sensor_Endress, r'\Delta P_{30\,kPa} / L', r'[Pa/m]'],
    ['Alpha', r'\alpha', r''],
    ['J Agua', r'J_{water}', r'[m/s]'],
    ['J Ar corrigido', r'J_{air}', r'[m/s]'],
    ['FT-A-0302', r'Q_{air}', r'[m³/h]'],
    ['PIT-M-0101', r'Gauge\ Pressure', r'[Bar]'],
    ['TIT-M-0101', r'Temperature', r'[°C]'],
    ['rho_g', r'\rho_{air}', r'[kg/m³]']
]

####################################################################################################################################################
#       '                                   END INPUTS
####################################################################################################################################################
g = 9.81        # m/s² 
rho_s = 962     # kg/m³         Densidade do silicone nas tomadas do sensor Yokogawa

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
    ##########################################################
    df['J Ar corrigido'] = df['J Ar'] * (1 - 0.06675) ###CUIDADO
    ##########################################################
    return df, data_teste

def format_filename(apelido, unidade):
    """
    Formata o nome do arquivo removendo caracteres LaTeX e adicionando a unidade.
    Exemplo: r'\Delta P_{40\,kPa}' [Pa] -> Delta_P_40kPa [Pa]
    """
    # Remove caracteres LaTeX comuns
    nome = apelido.replace('\\', '')  # Remove barras invertidas
    nome = nome.replace('{', '')      # Remove chaves
    nome = nome.replace('}', '')
    nome = nome.replace('\\,', '')    # Remove espaços LaTeX
    nome = nome.replace('\\frac', '') # Remove frações
    nome = nome.replace('$', '')      # Remove símbolos de dólar
    
    # Remove caracteres especiais que podem causar problemas no nome do arquivo
    caracteres_invalidos = ['|', '/', '\\', ':', '*', '?', '"', '<', '>', ',', ';', '=', ' ']
    for char in caracteres_invalidos:
        nome = nome.replace(char, '_')
    
    # Remove underscores múltiplos
    while '__' in nome:
        nome = nome.replace('__', '_')
    
    # Remove underscores no início e fim
    nome = nome.strip('_')
    
    # Adiciona a unidade
    if unidade:
        # Remove colchetes da unidade
        unidade = unidade.replace('[', '').replace(']', '')
        nome = f"{nome}_{unidade}"
    
    return nome

def save_results(df, coluna_escolhida, start_idx, end_idx, min_std, media_janela, 
                min_window_size, max_window_size, best_window_size, file_path, data_teste, 
                fluid_1, fluid_2, direction, theta, ID, nomes=None, medias=None, desvios=None, uAs=None,
                escolha_janela=None):
    """
    Salva os resultados da análise em um arquivo de saída.
    Agora inclui as estatísticas (média, desvio padrão, uA) de cada variável de interesse.
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
        f"ID do ponto: {ID}",
        f"Fluido 1: {fluid_1}",
        f"Fluido 2: {fluid_2}",
        f"Direção do escoamento: {direction}",
        f"Inclinação (theta): {theta}°",
        f"Intensidadedo gas densitometro (Ig): {I_g}",
        f"Intensidadedo fluido densitometro (If): {I_f}",
    ]
    
    # Adiciona informações específicas baseado no tipo de janela
    if escolha_janela == '1':  # Janela manual
        header.extend([
            "***Janela Manual***",
            f"Tamanho da Janela: {best_window_size:.1f} segundos",
            f"Tempo Inicial: {df['X_Value'].iloc[start_idx]:.2f} segundos",
            f"Tempo Final: {df['X_Value'].iloc[end_idx-1]:.2f} segundos",
            f"Número de Pontos: {end_idx - start_idx}",
        ])
    else:  # Janela automática
        header.extend([
            f"Coluna Critério: {coluna_escolhida}",
            f"Tamanho Mínimo da Janela: {min_window_size:.1f} segundos",
            f"Tamanho Máximo da Janela: {max_window_size:.1f} segundos",
            f"Tamanho Ótimo da Janela: {best_window_size:.1f} segundos",
            f"Média da Janela: {media_janela:.4f}",
            f"Desvio Padrão: {min_std:.4f}",
            f"Tempo Inicial: {df['X_Value'].iloc[start_idx]:.2f} segundos",
            f"Tempo Final: {df['X_Value'].iloc[end_idx-1]:.2f} segundos",
            f"Número de Pontos: {end_idx - start_idx}",
        ])
    
    # Adiciona a seção de estatísticas se fornecidas
    if nomes is not None and medias is not None and desvios is not None and uAs is not None:
        header.append("***Estatísticas das variáveis na janela***")
        header.append("Variável: Média | Desvio padrão | Incerteza tipo A")
        
        # Adiciona as estatísticas das variáveis originais
        for nome, media, desvio, uA in zip(nomes, medias, desvios, uAs):
            # Encontra o apelido e unidade para cada variável
            apelido_var = nome[0]
            unidade_var = ''
            for col in colunas_analise_filtradas:
                if isinstance(col, (list, tuple)) and col[0] == nome[0]:
                    apelido_var = col[1]
                    unidade_var = col[2]
                    break
            
            # Formata o nome da variável para exibição
            nome_display = apelido_var.replace('\\', '')  # Remove barras invertidas
            nome_display = nome_display.replace('{', '')  # Remove chaves
            nome_display = nome_display.replace('}', '')
            nome_display = nome_display.replace('\\,', '')  # Remove espaços LaTeX
            nome_display = nome_display.replace('\\frac', '')  # Remove frações
            nome_display = nome_display.replace('$', '')  # Remove símbolos de dólar
            nome_display = nome_display.replace('|', '')  # Remove barras verticais
            
            # Adiciona a unidade se existir
            if unidade_var:
                nome_display = f"{nome_display} {unidade_var}"
            
            header.append(f"{nome_display}: {media:.6f} | {desvio:.6f} | {uA:.6f}")
        
        # Adiciona as estatísticas do alpha na janela
        if 'Alpha' in df.columns:
            alpha_janela = df['Alpha'].iloc[start_idx:end_idx]
            media_alpha = alpha_janela.mean()
            desvio_alpha = alpha_janela.std(ddof=1)
            uA_alpha = desvio_alpha / np.sqrt(len(alpha_janela))
            header.append(f"alpha: {media_alpha:.6f} | {desvio_alpha:.6f} | {uA_alpha:.6f}")
        
        # Adiciona as estatísticas do dP_F para cada sensor PDT
        for col in colunas_analise_filtradas:
            if isinstance(col, (list, tuple)) and col[0].startswith('PDT'):
                dP_F_col = f'dP_F/dz {col[0]}'
                if dP_F_col in df.columns:
                    dP_F_janela = df[dP_F_col].iloc[start_idx:end_idx]
                    media_dP_F = dP_F_janela.mean()
                    desvio_dP_F = dP_F_janela.std(ddof=1)
                    uA_dP_F = desvio_dP_F / np.sqrt(len(dP_F_janela))
                    header.append(f"dP_F/dz {col[0]} [Pa/m]: {media_dP_F:.6f} | {desvio_dP_F:.6f} | {uA_dP_F:.6f}")
    
    header += [
        "***Dados da Janela***",
        "***End_of_Header***"
    ]
    
    # Seleciona os dados da janela para todas as colunas
    window_data = df.iloc[start_idx:end_idx]
    
    # Calcula as médias de todas as colunas na janela
    medias_janela = window_data.mean()
    
    # Salva o arquivo
    with open(output_file, 'w', encoding='utf-8') as f:
        # Escreve o cabeçalho
        f.write('\n'.join(header))
        f.write('\n')
        
        # Escreve a seção de médias
        f.write('***Resumo Medias***\n')
        # Escreve os nomes das colunas, corrigindo os nomes dos gradientes e reorganizando
        colunas_corrigidas = []
        coluna_gravitacional = None
        
        for col in df.columns:
            if col.startswith('dP_F/dz dP_dz_total_'):
                colunas_corrigidas.append(col.replace('dP_F/dz dP_dz_total_', 'dP_dz_total_'))
            elif col.startswith('dP_F/dz dP_dz_gravitacional'):
                coluna_gravitacional = 'dP_dz_gravitacional'
            else:
                colunas_corrigidas.append(col)
        
        # Adiciona a coluna gravitacional por último
        if coluna_gravitacional:
            colunas_corrigidas.append(coluna_gravitacional)
        
        f.write('\t'.join(colunas_corrigidas))
        f.write('\n')
        
        # Reorganiza as médias na mesma ordem das colunas corrigidas
        medias_corrigidas = []
        for col in colunas_corrigidas:
            if col == 'dP_dz_gravitacional':
                medias_corrigidas.append(medias_janela['dP_F/dz dP_dz_gravitacional'])
            elif col.startswith('dP_dz_total_'):
                col_original = 'dP_F/dz dP_dz_total_' + col.replace('dP_dz_total_', '')
                medias_corrigidas.append(medias_janela[col_original])
            else:
                medias_corrigidas.append(medias_janela[col])
        
        # Escreve as médias
        f.write('\t'.join([f"{val:.6f}" if isinstance(val, (int, float)) else str(val) 
                         for val in medias_corrigidas]))
        f.write('\n\n')
        
        f.write('***Serie Temporal Janelada***\n')
        # Escreve os dados
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
            y_data = df[nome_coluna]/L
            rms_pdt = np.sqrt(np.mean(np.square(y_data)))
            ax.plot(df['X_Value'], y_data, 'b-', alpha=0.8)
        else:
            y_data = df[nome_coluna]
            ax.plot(df['X_Value'], y_data, 'b-', alpha=0.8)
        # Linha da média da série completa
        media_serie = y_data.mean()
        desvio_serie = y_data.std()
         # Ajuste automático do eixo y com margem de 1%
        y_min = y_data.min()
        y_max = y_data.max()

        if 'PDT' in nome_coluna:
            # Se todos os valores são positivos
            if y_min >= 0:
                margem_min = y_min * 0.99  # 1% abaixo do mínimo
                margem_max = y_max * 1.01  # 1% acima do máximo
            # Se todos os valores são negativos
            elif y_max <= 0:
                margem_min = y_min * 1.01  # 1% acima do mínimo (menos negativo)
                margem_max = y_max * 0.99  # 1% abaixo do máximo (mais próximo de zero)
            # Se os valores oscilam em torno de zero
            else:
                max_abs = max(abs(y_min), abs(y_max))
                margem_min = -max_abs * 1.01 # 1% abaixo do maior valor absoluto
                margem_max = max_abs * 1.01  # 1% acima do maior valor absoluto
        else:
            # Margem padrão de 1% para outras colunas (se não houver lógica específica)
            margem_min = y_min * 0.99
            margem_max = y_max * 1.01
            
        ax.set_ylim(margem_min, margem_max)
        
        ax.axhline(y=media_serie, color='g', linestyle='--', label=f'Mean: {media_serie:.4f}')
        ax.axhline(y=media_serie + desvio_serie, color='r', linestyle=':', label=f'std: ±{desvio_serie:.4f}')
        if nome_coluna.startswith('PDT-') and (y_min/abs(y_min)) != (y_max/abs(y_max)):
            ax.axhline(y=rms_pdt, color='g', linestyle='-.', label=f'RMS: {rms_pdt:.4f}')
        ax.axhline(y=media_serie - desvio_serie, color='r', linestyle=':')
        ax.legend(fontsize=8, loc='upper right')
        if linha == n_linhas - 1:
            ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel(f"${apelido}$ {unidade}", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=9)
       
        
        
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
            y_data = (df[nome_coluna].iloc[start_idx:end_idx])/L
            y_data_full = (df[nome_coluna])/L
            rms_pdt_win = np.sqrt(np.mean(np.square(y_data)))
            ax.plot(df['X_Value'], y_data_full, 'b-', alpha=0.3, label='Full Series (abs)')
            ax.plot(df['X_Value'].iloc[start_idx:end_idx], y_data, 'r-', alpha=0.8, label=f'Window = {best_window_size:.0f} s')
        else:
            y_data = df[nome_coluna].iloc[start_idx:end_idx]
            y_data_full = df[nome_coluna]
            ax.plot(df['X_Value'], df[nome_coluna], 'b-', alpha=0.3, label='Full Series')
            ax.plot(df['X_Value'].iloc[start_idx:end_idx], y_data, 'r-', alpha=0.8, label=f'Window = {best_window_size:.0f} s')
        media_janela = y_data.mean()
        media_serie = np.mean(df[nome_coluna])
        desvio_janela = y_data.std()
        
        # Ajuste automático do eixo y com margem de 1% para a série completa
        y_min = y_data_full.min()
        y_max = y_data_full.max()
        
        if 'PDT' in nome_coluna:
            # Se todos os valores são positivos
            if y_min >= 0:
                margem_min = y_min * 0.99  # 1% abaixo do mínimo
                margem_max = y_max * 1.01  # 1% acima do máximo
            # Se todos os valores são negativos
            elif y_max <= 0:
                margem_min = y_min * 1.01  # 1% acima do mínimo (menos negativo)
                margem_max = y_max * 0.99  # 1% abaixo do máximo (mais próximo de zero)
            # Se os valores oscilam em torno de zero
            else:
                max_abs = max(abs(y_min), abs(y_max))
                margem_min = -max_abs * 1.01 # 1% abaixo do maior valor absoluto
                margem_max = max_abs * 1.01  # 1% acima do maior valor absoluto
        else:
            # Margem padrão de 1% para outras colunas (se não houver lógica específica)
            margem_min = y_min * 0.99
            margem_max = y_max * 1.01

        ax.set_ylim(margem_min, margem_max)
        
        ax.axhline(y=media_janela, color='g', linestyle='--', label=f'Mean: {media_janela:.4f}')
        ax.axhline(y=media_janela + desvio_janela, color='r', linestyle=':', label=f'std: ±{desvio_janela:.4f}')
        if nome_coluna.startswith('PDT-') and (y_min/abs(y_min)) != (y_max/abs(y_max)):
            ax.axhline(y=rms_pdt_win, color='g', linestyle='-.', label=f'RMS: {rms_pdt_win:.4f}')
        ax.axhline(y=media_janela - desvio_janela, color='r', linestyle=':')
        ax.legend(fontsize=8, loc='upper right')
        if linha == n_linhas - 1:
            ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel(f"${apelido}$ {unidade}", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=9)
        
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

        if coluna_pdt_nome == sensor_Yokogawa:
            rho_tubbing = rho_s
        else:
            rho_tubbing = rho_agua
        
        termo_gravitacional = ((1-alpha_series)*rho_agua + alpha_series*rho_ar - rho_tubbing) * g * np.sin(theta_rad)   # Pa/m
        
        dP_dz_gravitacional = ((1-alpha_series)*rho_agua + alpha_series*rho_ar) * g * np.sin(theta_rad)   # Pa/m

        print('Direção do escoamento:', direction)

        if direction in ['Upward', 'Horizontal']:
            dP_F_over_dz_series = (delta_p_prime / L) - termo_gravitacional                 # Calcula o gradiente de pressão friccional
            # Calcula o RMS do sinal dP_F_over_dz_series
            dP_F_over_dz_RMS = np.sqrt(np.mean(np.square(dP_F_over_dz_series)))
            # Calcula o dP_dz_total usando o RMS
            dP_dz_total = dP_F_over_dz_RMS + np.mean(dP_dz_gravitacional) # Calcula o gradiente de pressão Total
        elif direction == 'Downward':
            dP_F_over_dz_series = -(delta_p_prime / L) + termo_gravitacional 
            # Calcula o RMS do sinal dP_F_over_dz_series
            dP_F_over_dz_series_mean = np.mean(dP_F_over_dz_series)
            dP_F_over_dz_RMS = np.sqrt(np.mean(np.square(dP_F_over_dz_series)))

            # Calcula o dP_dz_total usando o RMS
            dP_dz_total = -(dP_F_over_dz_RMS) + np.mean(dP_dz_gravitacional) # Calcula o gradiente de pressão Total
            dP_dz_total_mean = -np.mean(dP_F_over_dz_series) + np.mean(dP_dz_gravitacional)
            
        
        # Armazena a série calculada no DataFrame dP_F_df com o nome da coluna original
        dP_F_df[coluna_pdt_nome] = dP_F_over_dz_series
        # Armazena também o dP_dz_total
        dP_F_df[f'dP_dz_total_{coluna_pdt_nome}'] = dP_dz_total
        # Armazena o dP_dz_gravitacional (apenas na primeira iteração)
        if i == indices_pdt[0]:
            dP_F_df['dP_dz_gravitacional'] = dP_dz_gravitacional

    # Agora dP_F_df contém as séries temporais de dP_F/dz para cada coluna PDT
    print("\nDataFrame dP_F_df criado com as séries temporais de gradiente de pressão friccional para colunas PDT.")

    return dP_F_df # Retorna o DataFrame

def extract_info_from_filename(filename):
    """
    Extrai informações do nome do arquivo experimental.
    Formato esperado: XXX##ID## onde:
    - X: letra indicando o fluido (A:Air, W:Water, O:Oil, S:SF6)
    - #: número indicando a inclinação em graus
    - ID: identificador do ponto experimental
    """
    # Dicionários para mapear letras para fluidos e direções
    fluid_map = {
        'A': 'Air',
        'W': 'Water',
        'O': 'Oil',
        'S': 'SF6'
    }
    
    direction_map = {
        'H': 'Horizontal',
        'U': 'Upward',
        'D': 'Downward'
    }
    
    # Obtém apenas o nome do arquivo sem extensão e caminho
    base_name = os.path.splitext(os.path.basename(filename))[0]
    
    # Extrai as informações
    fluid_1 = fluid_map.get(base_name[0], 'Unknown')
    fluid_2 = fluid_map.get(base_name[1], 'Unknown')
    direction = direction_map.get(base_name[2], 'Unknown')
    theta = int(base_name[3:5])  # Extrai os dois dígitos da inclinação
    ID = base_name[5:]  # Resto do nome é o ID
    
    return fluid_1, fluid_2, direction, theta, ID

def check_required_columns(df, colunas_analise):
    """
    Verifica se as colunas necessárias existem no DataFrame.
    Retorna True se todas existirem, False caso contrário.
    """
    colunas_faltantes = []
    
    # Lista de colunas que são calculadas internamente
    colunas_calculadas = ['Alpha', 'rho_g', 'J Ar corrigido']
    
    # Verifica cada coluna do array colunas_analise
    for coluna_info in colunas_analise:
        nome_coluna = coluna_info[0]
        # Pula as colunas que são calculadas internamente
        if nome_coluna in colunas_calculadas:
            continue
        # Verifica se a coluna existe no DataFrame
        if nome_coluna not in df.columns:
            colunas_faltantes.append(nome_coluna)
    
    if colunas_faltantes:
        print("\nERRO: As seguintes colunas não foram encontradas no arquivo:")
        for coluna in colunas_faltantes:
            print(f"- {coluna}")
        print("\nPor favor, verifique se o arquivo de entrada está correto.")
        return False
    
    return True

# Exemplo de uso:
if __name__ == "__main__":
    
    # file_path = "example/FSC2_Agua_Ar_Downward_45_graus/ID15"
    df, data_teste = read_file(file_path)    
    print("Dimensões do DataFrame:", df.shape)
    print("\nNomes das colunas:")
    print(df.columns.tolist())
    
    # Verifica se as colunas necessárias existem
    if not check_required_columns(df, colunas_analise):
        sys.exit(1)
    
    # Extrai informações do nome do arquivo
    fluid_1, fluid_2, direction, theta, ID = extract_info_from_filename(file_path)
    print(f"\nInformações extraídas do nome do arquivo:")
    print(f"Fluido 1: {fluid_1}")
    print(f"Fluido 2: {fluid_2}")
    print(f"Direção: {direction}")
    print(f"Inclinação (theta): {theta}°")
    print(f"ID do ponto: {ID}")
    
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
        # Para janela manual, usamos a primeira coluna como critério apenas para cálculo do desvio padrão
        coluna_escolhida = colunas_analise_filtradas[0][0]
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
                fluid_1, fluid_2, direction, theta, ID,
                nomes=colunas_analise_filtradas, medias=medias, desvios=desvios, uAs=uAs,
                escolha_janela=escolha_janela)
    
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
        # Fixa o range do eixo y entre 0 e 1
        # plt.ylim(0, 1)
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
        
        # Plota o dP_dz_gravitacional
        grav_series = dP_F_df['dP_dz_gravitacional']
        grav_mean = grav_series.mean()
        grav_std = grav_series.std()
        plt.plot(df['X_Value'].iloc[start_idx:end_idx], grav_series, 
                label=f'dP/dz gravitacional\nmean: {grav_mean:.2f} ± {grav_std:.2f}', 
                color='black', linestyle='--')
        
        # Plota as séries de dP_F para cada sensor
        for col in dP_F_df.columns:
            if not col.startswith('dP_dz_total') and col != 'dP_dz_gravitacional':  # Plota apenas as séries de dP_F
                # Formata o label baseado no tipo de sensor
                if col == sensor_Yokogawa:
                    sensor_range = col.split('-')[3]  # Pega o range após o terceiro hífen
                    label_base = f'Yokogawa {sensor_range}'
                elif col == sensor_Endress:
                    sensor_range = col.split('-')[3]  # Pega o range após o terceiro hífen
                    label_base = f'Endress {sensor_range}'
                else:
                    label_base = col
                
                # Calcula RMS e desvio padrão
                series = dP_F_df[col]
                rms = np.sqrt(np.mean(np.square(series)))
                
                plt.plot(df['X_Value'].iloc[start_idx:end_idx], series, 
                        label=f'dP_F/dz {label_base}\nRMS: {rms:.2f}', 
                        alpha=0.7)
        
        # Plota as séries de dP_dz_total para cada sensor
        for col in dP_F_df.columns:
            if col.startswith('dP_dz_total'):  # Plota apenas as séries de dP_dz_total
                # Extrai o nome do sensor original
                sensor_name = col.replace('dP_dz_total_', '')
                # Formata o label baseado no tipo de sensor
                if sensor_name == sensor_Yokogawa:
                    sensor_range = sensor_name.split('-')[3]  # Pega o range após o terceiro hífen
                    label_base = f'Yokogawa {sensor_range}'
                elif sensor_name == sensor_Endress:
                    sensor_range = sensor_name.split('-')[3]  # Pega o range após o terceiro hífen
                    label_base = f'Endress {sensor_range}'
                else:
                    label_base = sensor_name
            
                
                # Calcula RMS e desvio padrão
                series = abs(dP_F_df[col])
                
                plt.plot(df['X_Value'].iloc[start_idx:end_idx], series, 
                        label=f'dP/dz total {label_base}\nValue: {np.mean(abs(series)):.2f}', 
                        linestyle=':', linewidth=2)
        
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel(r'$ \frac{\Delta P}{L}$ [Pa/m]', fontsize=12)
        plt.title(r'Pressure Gradients', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        output_path_dpf = os.path.join(output_dir, f"dP_F_dz-{base_name}.png")
        plt.savefig(output_path_dpf, bbox_inches='tight')
        print(f"Gráfico dP_F/dz salvo em: {output_path_dpf}")
        plt.close(plt.gcf()) # Fecha a figura atual
    else:
        print("Não foi possível calcular o gradiente de pressão friccional para plotar.")
