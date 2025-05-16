import uncertainties as unc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# a = unc.ufloat(5.67,0.12)
# b = unc.ufloat(9.23,0.2)

# result = a * b
# print("{:.4uP}".format(result))

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
    column_names = lines[header_end_idx].strip().split('\t')
    
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
                min_window_size, max_window_size, best_window_size, file_path, data_teste):
    """
    Salva os resultados da análise em um arquivo de saída.
    
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

def plot_time_series(df, colunas):
    """
    Plota as séries temporais em subplots organizados em duas colunas.
    
    Args:
        df (pandas.DataFrame): DataFrame com os dados
        colunas (list): Lista com os nomes das colunas a serem plotadas
    """
    n_colunas = 2
    n_linhas = (len(colunas) + 1) // 2
    
    # Usa constrained_layout para melhor organização automática
    fig, axs = plt.subplots(n_linhas, n_colunas, figsize=(16, 3.8*n_linhas), constrained_layout=True)
    fig.suptitle('Séries Temporais das Variáveis', fontsize=18, y=1.03)
    
    # Plota cada série temporal
    for idx, coluna in enumerate(colunas):
        linha = idx // n_colunas
        col = idx % n_colunas
        ax = axs[linha, col] if n_linhas > 1 else axs[col]
        ax.plot(df['X_Value'], df[coluna], 'b-', alpha=0.8)
        ax.set_title(coluna, pad=10, fontsize=13, fontweight='bold')
        if linha == n_linhas - 1:
            ax.set_xlabel('Tempo (s)', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=9)
        y_min = df[coluna].min() * 0.99
        y_max = df[coluna].max() * 1.01
        ax.set_ylim(y_min, y_max)
    # Remove subplots vazios se houver
    for idx in range(len(colunas), n_linhas * n_colunas):
        linha = idx // n_colunas
        col = idx % n_colunas
        fig.delaxes(axs[linha, col])
    plt.show()

def plot_windows(df, colunas, start_idx, end_idx, best_window_size):
    """
    Plota as janelas de todas as variáveis em subplots organizados em duas colunas.
    
    Args:
        df (pandas.DataFrame): DataFrame com os dados
        colunas (list): Lista com os nomes das colunas a serem plotadas
        start_idx (int): Índice inicial da janela
        end_idx (int): Índice final da janela
        best_window_size (float): Tamanho da janela encontrada
    """
    n_colunas = 2
    n_linhas = (len(colunas) + 1) // 2
    
    # Usa constrained_layout para melhor organização automática
    fig, axs = plt.subplots(n_linhas, n_colunas, figsize=(16, 3.8*n_linhas), constrained_layout=True)
    fig.suptitle(f'Janelas das Variáveis (Tamanho: {best_window_size:.1f}s)', fontsize=18, y=1.03)
    
    # Plota cada série temporal
    for idx, coluna in enumerate(colunas):
        linha = idx // n_colunas
        col = idx % n_colunas
        ax = axs[linha, col] if n_linhas > 1 else axs[col]
        
        # Plota a série temporal completa
        ax.plot(df['X_Value'], df[coluna], 'b-', alpha=0.3, label='Série Completa')
        
        # Plota a janela
        ax.plot(df['X_Value'].iloc[start_idx:end_idx], 
                df[coluna].iloc[start_idx:end_idx], 
                'r-', alpha=0.8, label='Janela')
        
        # Adiciona a média da janela
        media_janela = df[coluna].iloc[start_idx:end_idx].mean()
        ax.axhline(y=media_janela, color='g', linestyle='--', 
                  label=f'Média: {media_janela:.4f}')
        
        ax.set_title(coluna, pad=10, fontsize=13, fontweight='bold')
        if linha == n_linhas - 1:
            ax.set_xlabel('Tempo (s)', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=9)
        
        # Ajusta os limites do eixo y para melhor visualização
        y_min = df[coluna].min() * 0.99
        y_max = df[coluna].max() * 1.01
        ax.set_ylim(y_min, y_max)
        
        # Adiciona legenda
        ax.legend(fontsize=8, loc='upper right')
    
    # Remove subplots vazios se houver
    for idx in range(len(colunas), n_linhas * n_colunas):
        linha = idx // n_colunas
        col = idx % n_colunas
        fig.delaxes(axs[linha, col])
    
    plt.show()

# Exemplo de uso:
if __name__ == "__main__":
    file_path = "example/ID4"
    df, data_teste = read_file(file_path)    
    print("Dimensões do DataFrame:", df.shape)
    print("\nNomes das colunas:")
    print(df.columns.tolist())
    
    # Lista de colunas para análise
    colunas_analise = [
        'PIT-M-0101',
        'PDT-M-0101-40kPa',
        'PDT-M-0101B-10kPa',
        'PDT-M-0101C-3kPa',
        'TIT-M-0101',
        'Densitometro',
        'J Ar',
        'J Água'
    ]
    
    # Mostra as séries temporais antes da escolha da variável
    print("\nVisualizando as séries temporais das variáveis disponíveis...")
    plot_time_series(df, colunas_analise)
    
    # Mostra as colunas disponíveis
    print("\nColunas disponíveis para análise:")
    for i, col in enumerate(colunas_analise, 1):
        print(f"{i}. {col}")
    
    # Solicita a coluna para análise
    while True:
        try:
            coluna_escolhida = input("\nDigite o nome exato da coluna para análise: ")
            if coluna_escolhida in colunas_analise:
                break
            else:
                print("Coluna não encontrada. Por favor, digite o nome exato da coluna.")
        except ValueError:
            print("Entrada inválida. Por favor, tente novamente.")
    
    # Solicita o tamanho mínimo da janela
    while True:
        try:
            min_window_size = float(input("\nDigite o tamanho mínimo da janela em segundos: "))
            if min_window_size > 0:
                break
            else:
                print("O tamanho mínimo da janela deve ser maior que zero.")
        except ValueError:
            print("Entrada inválida. Por favor, digite um número válido.")
    
    # Solicita o tamanho máximo da janela
    while True:
        try:
            max_window_size = float(input("\nDigite o tamanho máximo da janela em segundos: "))
            if max_window_size >= min_window_size:
                break
            else:
                print("O tamanho máximo da janela deve ser maior ou igual ao tamanho mínimo.")
        except ValueError:
            print("Entrada inválida. Por favor, digite um número válido.")
    
    # Encontra a janela com menor desvio padrão
    try:
        start_idx, end_idx, min_std, best_window_size = find_min_std_window(
            df, coluna_escolhida, min_window_size, max_window_size)
        
        # Calcula a média para a janela encontrada
        media_janela = df[coluna_escolhida].iloc[start_idx:end_idx].mean()
        
        print(f"\nResultados para a coluna '{coluna_escolhida}':")
        if min_window_size == max_window_size:
            print(f"Tamanho fixo da janela: {best_window_size:.1f} segundos")
        else:
            print(f"Tamanho ótimo da janela encontrado: {best_window_size:.1f} segundos")
        print(f"Média da janela: {media_janela:.4f}")
        print(f"Menor desvio padrão encontrado: {min_std:.4f}")
        print(f"Tempo inicial da janela: {df['X_Value'].iloc[start_idx]:.2f} segundos")
        print(f"Tempo final da janela: {df['X_Value'].iloc[end_idx-1]:.2f} segundos")
        print(f"Número de pontos na janela: {end_idx - start_idx}")
        
        # Mostra os dados da janela
        print("\nDados da janela com menor desvio padrão:")
        print(df.iloc[start_idx:end_idx][['X_Value', coluna_escolhida]])
        
        # Salva os resultados em um arquivo
        save_results(df, coluna_escolhida, start_idx, end_idx, min_std, media_janela,
                    min_window_size, max_window_size, best_window_size, file_path, data_teste)
        
        # Plota o gráfico da variável critério
        plt.figure(figsize=(15, 8))
        
        # Plota a série temporal completa
        plt.plot(df['X_Value'], df[coluna_escolhida], 'b-', 
                label='Série Temporal Completa', alpha=0.7)
        
        # Destaca a janela com menor desvio padrão
        plt.axvspan(df['X_Value'].iloc[start_idx], df['X_Value'].iloc[end_idx-1], 
                   alpha=0.3, color='red', label='Janela Ótima')
        
        # Adiciona a média como uma linha horizontal na janela
        plt.axhline(y=media_janela, color='g', linestyle='--', 
                   label=f'Média Janela: {media_janela:.4f}')
        
        # Configurações do gráfico
        plt.xlabel('Tempo (s)', fontsize=12)
        plt.ylabel(coluna_escolhida, fontsize=12)
        if min_window_size == max_window_size:
            plt.title(f'Série Temporal de {coluna_escolhida}\nJanela Fixa: {best_window_size:.1f}s', 
                     fontsize=14)
        else:
            plt.title(f'Série Temporal de {coluna_escolhida}\nJanela Ótima: {best_window_size:.1f}s', 
                     fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        
        # Ajusta os limites do eixo x para mostrar toda a série temporal
        plt.xlim(df['X_Value'].min(), df['X_Value'].max())
        
        # Ajusta os limites do eixo y para melhor visualização
        y_min = df[coluna_escolhida].min() * 0.99
        y_max = df[coluna_escolhida].max() * 1.01
        plt.ylim(y_min, y_max)
        
        plt.tight_layout()
        plt.show()
        
        # Plota as janelas de todas as variáveis
        print("\nVisualizando as janelas de todas as variáveis...")
        plot_windows(df, colunas_analise, start_idx, end_idx, best_window_size)
        
    except ValueError as e:
        print(f"\nErro: {e}")
