import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os
import sys
from pathlib import Path

####################################################################################################################################################
#                                            INPUTS
####################################################################################################################################################
file_path = 'data_example/example/Mean_Experimental_Data_FSC2_v2.xlsx' #Insira o caminho do arquivo a ser analisado NOTE: USE SEMPRE A BARRA NORMAL '/', SE ESTIVER INVERTIDA, MODIFIQUE-A

def read_excel_file(file_path):
    """
    Lê um arquivo Excel e retorna um DataFrame com os dados.
    Os nomes das colunas estão na linha 3 e as unidades na linha 4.
    
    Args:
        file_path (str): Caminho para o arquivo Excel
        
    Returns:
        pd.DataFrame: DataFrame com os dados do arquivo Excel
        dict: Dicionário com as unidades das colunas
    """
    try:
        # Verificar se o arquivo existe
        if not os.path.exists(file_path):
            print(f"Erro: Arquivo '{file_path}' não encontrado.")
            return None, None
            
        # Verificar se é um arquivo Excel
        if not file_path.lower().endswith(('.xlsx', '.xls')):
            print(f"Erro: Arquivo '{file_path}' não é um arquivo Excel válido.")
            return None, None
            
        print(f"Lendo arquivo Excel: {file_path}")
        
        # Tentar ler o arquivo Excel
        try:
            # Primeiro, tentar ler todas as abas
            excel_file = pd.ExcelFile(file_path)
            print(f"Abas encontradas: {excel_file.sheet_names}")
            
            # Se houver múltiplas abas, perguntar qual usar
            if len(excel_file.sheet_names) > 1:
                print("\nMúltiplas abas encontradas:")
                for i, sheet in enumerate(excel_file.sheet_names):
                    print(f"{i+1}: {sheet}")
                
                while True:
                    try:
                        choice = input(f"\nEscolha a aba (1-{len(excel_file.sheet_names)}) ou 'all' para todas: ").strip()
                        if choice.lower() == 'all':
                            # Ler todas as abas
                            dataframes = {}
                            units_dict = {}
                            for sheet in excel_file.sheet_names:
                                print(f"Lendo aba: {sheet}")
                                df, units = read_single_sheet(file_path, sheet)
                                dataframes[sheet] = df
                                units_dict[sheet] = units
                            return dataframes, units_dict
                        else:
                            sheet_index = int(choice) - 1
                            if 0 <= sheet_index < len(excel_file.sheet_names):
                                sheet_name = excel_file.sheet_names[sheet_index]
                                print(f"Lendo aba: {sheet_name}")
                                df, units = read_single_sheet(file_path, sheet_name)
                                break
                            else:
                                print("Escolha inválida. Tente novamente.")
                    except ValueError:
                        print("Entrada inválida. Digite um número ou 'all'.")
            else:
                # Apenas uma aba
                sheet_name = excel_file.sheet_names[0]
                print(f"Lendo aba: {sheet_name}")
                df, units = read_single_sheet(file_path, sheet_name)
                
        except Exception as e:
            print(f"Erro ao ler arquivo Excel: {e}")
            return None, None
            
        # Exibir informações básicas do DataFrame
        print(f"DataFrame carregado: {df.shape[0]} linhas x {df.shape[1]} colunas")
        
        return df, units
        
    except Exception as e:
        print(f"Erro inesperado: {e}")
        return None, None

def read_single_sheet(file_path, sheet_name):
    """
    Lê uma aba específica do arquivo Excel, usando linha 3 para nomes das colunas
    e linha 4 para unidades. Lê apenas a partir da coluna B até a linha 20.
    
    Args:
        file_path (str): Caminho para o arquivo Excel
        sheet_name (str): Nome da aba
        
    Returns:
        pd.DataFrame: DataFrame com os dados
        dict: Dicionário com as unidades das colunas
    """
    try:
        # Ler as primeiras linhas para obter nomes das colunas e unidades
        df_header = pd.read_excel(file_path, sheet_name=sheet_name, header=None, nrows=4, usecols="B:Z")
        
        # Obter nomes das colunas da linha 3 (índice 2)
        column_names = df_header.iloc[2].tolist()
        
        # Obter unidades da linha 4 (índice 3)
        units = df_header.iloc[3].tolist()
        
        # Criar dicionário de unidades
        units_dict = {}
        for i, (col_name, unit) in enumerate(zip(column_names, units)):
            if pd.notna(col_name) and pd.notna(unit):
                units_dict[col_name] = unit
            elif pd.notna(col_name):
                units_dict[col_name] = ""
        
        # Ler o DataFrame completo, pulando as primeiras 4 linhas, apenas colunas B em diante, até linha 20
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None, skiprows=4, usecols="B:Z", nrows=16)
        
        # Definir os nomes das colunas
        df.columns = column_names
        
        # Remover linhas vazias no início
        df = df.dropna(how='all')
        
        # Resetar o índice
        df = df.reset_index(drop=True)
        
        return df, units_dict
        
    except Exception as e:
        print(f"Erro ao ler aba {sheet_name}: {e}")
        return None, None

def save_dataframe_to_txt(df, units_dict, output_file, sheet_name=None):
    """
    Salva o DataFrame em formato de tabela em um arquivo .txt
    
    Args:
        df (pd.DataFrame): DataFrame para salvar
        units_dict (dict): Dicionário com as unidades das colunas
        output_file (str): Nome do arquivo de saída
        sheet_name (str): Nome da aba (opcional)
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            # Cabeçalho
            f.write("=" * 80 + "\n")
            f.write("DADOS DO ARQUIVO EXCEL\n")
            f.write("=" * 80 + "\n")
            
            if sheet_name:
                f.write(f"Aba: {sheet_name}\n")
            
            f.write(f"Data/hora de geração: {pd.Timestamp.now()}\n")
            f.write(f"Dimensões: {df.shape[0]} linhas x {df.shape[1]} colunas\n\n")
            
            # Informações das colunas e unidades
            f.write("COLUNAS E UNIDADES:\n")
            f.write("-" * 50 + "\n")
            for col in df.columns:
                unit = units_dict.get(col, "")
                f.write(f"{col}: {unit}\n")
            f.write("\n")
            
            # Dados em formato de tabela
            f.write("DADOS:\n")
            f.write("-" * 50 + "\n")
            
            # Calcular larguras das colunas
            col_widths = {}
            for col in df.columns:
                # Largura mínima baseada no nome da coluna
                col_widths[col] = len(str(col))
                
                # Verificar largura dos dados
                for value in df[col].head(100):  # Verificar apenas as primeiras 100 linhas
                    col_widths[col] = max(col_widths[col], len(str(value)))
                
                # Limitar largura máxima
                col_widths[col] = min(col_widths[col], 20)
            
            # Cabeçalho da tabela
            header_line = "|"
            separator_line = "|"
            for col in df.columns:
                header_line += f" {col:<{col_widths[col]}} |"
                separator_line += "-" * (col_widths[col] + 2) + "|"
            
            f.write(header_line + "\n")
            f.write(separator_line + "\n")
            
            # Dados da tabela
            for idx, row in df.iterrows():
                data_line = "|"
                for col in df.columns:
                    value = str(row[col])
                    if len(value) > col_widths[col]:
                        value = value[:col_widths[col]-3] + "..."
                    data_line += f" {value:<{col_widths[col]}} |"
                f.write(data_line + "\n")
                
                # Limitar o número de linhas para não sobrecarregar o arquivo
                if idx >= 1000:  # Máximo 1000 linhas
                    f.write(f"\n... (mostrando apenas as primeiras 1000 linhas de {len(df)})\n")
                    break
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("FIM DOS DADOS\n")
            f.write("=" * 80 + "\n")
        
        print(f"DataFrame salvo em: {output_file}")
        
    except Exception as e:
        print(f"Erro ao salvar arquivo: {e}")

def analyze_dataframe(df):
    """
    Analisa o DataFrame e fornece informações úteis.
    
    Args:
        df (pd.DataFrame): DataFrame para análise
    """
    if df is None:
        print("DataFrame vazio ou inválido.")
        return
        
    # Informações básicas
    print(f"Análise: {len(df)} linhas, {len(df.columns)} colunas")
    
    # Verificar valores nulos
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print(f"Valores nulos: {null_counts.sum()}")
    
    # Contar colunas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"Colunas numéricas: {len(numeric_cols)}")

def generate_jg_vs_alpha_plot(df, sheet_name):
    """
    Gera um plot científico de jG vs α, onde cada jL é uma série diferente.
    Inclui símbolos diferentes para cada Flow Pattern.
    
    Args:
        df (pd.DataFrame): DataFrame com os dados
        sheet_name (str): Nome da aba/sheet
    """
    try:
        # Verificar se as colunas necessárias existem (com possíveis espaços)
        required_cols = ['jG', 'jL', 'α', 'Flow Pattern']
        available_cols = list(df.columns)
        
        # Mapear nomes de colunas (removendo espaços extras)
        col_mapping = {}
        for col in available_cols:
            if pd.notna(col):
                clean_col = str(col).strip()
                col_mapping[clean_col] = col
        
        # Verificar se as colunas necessárias estão disponíveis
        missing_cols = []
        for req_col in required_cols:
            if req_col not in col_mapping:
                missing_cols.append(req_col)
        
        if missing_cols:
            print(f"Colunas ausentes para plot: {missing_cols}")
            print(f"Colunas disponíveis: {list(col_mapping.keys())}")
            return
        
        # Configurar estilo científico para LaTeX
        plt.style.use('default')
        plt.rcParams.update({
            'font.size': 18,
            'font.family': 'serif',
            'text.usetex': False,  # Habilitar LaTeX para renderização matemática
            'axes.linewidth': 1.0,
            'axes.grid': False,
            'grid.alpha': 0.3,
            'grid.linestyle': '--',
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            'pdf.fonttype': 42,  # Tipo 42 para compatibilidade com LaTeX
            'ps.fonttype': 42,
            'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
            'mathtext.fontset': 'cm'
        })
        
        # Criar figura
        fig, ax = plt.subplots(figsize=(10, 10))  # Figura quadrada 10x10
        
        # Obter valores únicos de jL e agrupar por séries
        jl_col = col_mapping['jL']
        jg_col = col_mapping['jG']
        alpha_col = col_mapping['α']
        flow_pattern_col = col_mapping['Flow Pattern']
        
        # Agrupar dados por jL (arredondando para 1 casa decimal para agrupar séries similares)
        df_plot = df.copy()
        df_plot['jL_rounded'] = df_plot[jl_col].round(1)
        
        # Obter séries únicas de jL
        jl_series = sorted(df_plot['jL_rounded'].unique())
        jl_series = [jl for jl in jl_series if pd.notna(jl)]
        
        print(f"Séries de jL encontradas: {jl_series}")
        
        # # Cores em escala de cinza (maior jL = mais escuro, menor jL = mais claro)
        # # Gerar cores baseadas nos valores de jL
        # gray_colors = []
        # for jl in jl_series:
        #     # Normalizar jL entre 0 e 1, onde 0 = mais claro, 1 = mais escuro
        #     jl_min = min(jl_series)
        #     jl_max = max(jl_series)
        #     if jl_max > jl_min:
        #         normalized_jl = (jl - jl_min) / (jl_max - jl_min)
        #     else:
        #         normalized_jl = 0.5  # Se todos os valores são iguais
            
        #     # Converter para escala de cinza (0.3 = cinza claro, 0.9 = cinza escuro)
        #     # jL = 0.2 → cinza claro, jL = 1.8 → cinza escuro
        #     gray_value = 0.6 * normalized_jl
        #     gray_color = f'#{int(gray_value * 255):02x}{int(gray_value * 255):02x}{int(gray_value * 255):02x}'
        #     gray_colors.append(gray_color)
        
        # colors = gray_colors[::-1]
        
        #Lista de cores
        colors = ['red', 'blue', 'orange', 'yellow', 'silver', 'white']
        # Estilos de linha para diferentes séries
        line_styles = ['-', '--', ':', '-.', '-', '--', ':', '-.', '-', '--']
        # Simbolos para as linhas
        symbol_line = ['o','h', 's', 'D', '^', 'v' ]
        # Símbolos para Flow Patterns (expandido para todos os padrões encontrados)
        # flow_pattern_symbols = {
        #     'Elongated Bubble': 'o',      # Círculo
        #     'Slug': 's',                  # Quadrado
        #     'slug': 's',                  # Quadrado (minúsculo)
        #     'sw': '^',                    # Triângulo para cima
        #     'SW': '^',                    # Triângulo para cima
        #     'SS': 'v',                    # Triângulo para baixo
        #     'Estratificado': 'D',         # Diamante
        #     'SW/MI': 'p',                 # Pentágono
        #     'Intermitent (Slug Wavy)²': 'h',  # Hexágono
        #     'Intermitent (Slug Wavy) ²': 'h', # Hexágono
        #     'Intermitent (Slug Wavy)¹ ²': 'H', # Hexágono grande
        #     'Rolling Wave²': '8',         # Octágono
        #     'DBubbly': 'P',               # Plus
        #     'A': 'X',                     # X
        #     'B': 'd',                     # Diamante pequeno
        #     'EB-A (I)': 'o',              # Círculo
        #     'EB-A  (I)': 'o',             # Círculo
        #     'EB-B (I)': 'o',              # Círculo
        #     'B-SL (I)': 's',              # Quadrado
        #     'SL (I)': 's',                # Quadrado
        #     'annular': 'o',               # Círculo
        #     'churn': 's',                 # Quadrado
        #     'Churn': 's',                 # Quadrado
        #     'Churn/bolhas?': 's',         # Quadrado
        #     'Churn?': 's',                # Quadrado
        #     'bubbles': 'o',               # Círculo
        #     'Intermitent¹ ²': 'h'         # Hexágono
        # }
        
        flow_pattern_symbols = {
            'Elongated Bubble': {'symbol': 'o', 'color': 'blue'},
            'Slug': {'symbol': 's', 'color': 'green'},
            'slug': {'symbol': 's', 'color': 'green'},
            'sw': {'symbol': '^', 'color': 'red'},
            'SW': {'symbol': '^', 'color': 'red'},
            'SS': {'symbol': 'v', 'color': 'purple'},
            'Estratificado': {'symbol': 'D', 'color': 'orange'},
            'SW/MI': {'symbol': 'p', 'color': 'cyan'},
            'Intermitent (Slug Wavy)²': {'symbol': 'h', 'color': 'magenta'},
            'Intermitent (Slug Wavy) ²': {'symbol': 'h', 'color': 'magenta'},
            'Intermitent (Slug Wavy)¹ ²': {'symbol': 'H', 'color': 'brown'},
            'Rolling Wave²': {'symbol': 'D', 'color': 'pink'},
            'DBubbly': {'symbol': 'P', 'color': 'gray'},
            'SL (I)': {'symbol': 's', 'color': 'maroon'},
            'annular': {'symbol': 'o', 'color': 'blue'},
            'churn': {'symbol': 's', 'color': 'green'},
            'Churn': {'symbol': 's', 'color': 'green'},
            'Churn/bolhas?': {'symbol': 's', 'color': 'green'},
            'Churn?': {'symbol': 's', 'color': 'green'},
            'bubbles': {'symbol': 'o', 'color': 'blue'},
            'Intermitent': {'symbol': 'h', 'color': 'magenta'}
        }
        vec_s_lines = []
        # Plotar cada série de jL
        for i, jl in enumerate(jl_series):
            # Filtrar dados para esta série de jL
            mask = df_plot['jL_rounded'] == jl
            
            jg_data = df_plot.loc[mask, jg_col]
            alpha_data = df_plot.loc[mask, alpha_col]
            flow_pattern_data = df_plot.loc[mask, flow_pattern_col]
            
            # Remover valores nulos
            valid_mask = pd.notna(jg_data) & pd.notna(alpha_data) & pd.notna(flow_pattern_data)
            jg_clean = jg_data[valid_mask]
            alpha_clean = alpha_data[valid_mask]
            flow_pattern_clean = flow_pattern_data[valid_mask]
            if len(jg_clean) > 0:
                color = colors[i % len(colors)]
                line_style = line_styles[i % len(line_styles)] # Escolher estilo de linha
                s_lines = symbol_line[i % len(symbol_line)]
                vec_s_lines.append(s_lines)
                # Ordenar dados por jG para conectar corretamente
                sorted_data = sorted(zip(jg_clean, alpha_clean, flow_pattern_clean))
                jg_sorted = [x[0] for x in sorted_data]
                alpha_sorted = [x[1] for x in sorted_data]
                flow_pattern_sorted = [x[2] for x in sorted_data]
                
                
                    
                # Plotar linha conectando os pontos (todas pretas, estilo varia por série)
                ax.plot(jg_sorted, alpha_sorted, line_style, marker=s_lines, markersize=18, color='black', mfc='silver', linewidth=1.5, zorder=1)
                # ax.plot(jg_sorted, alpha_sorted, line_style, marker=s_lines, markersize=16, color='black', mfc='silver', linewidth=1.5, zorder=1)

                # Plotar cada ponto com símbolo baseado no Flow Pattern
                for j, (jg_val, alpha_val, flow_pattern) in enumerate(zip(jg_sorted, alpha_sorted, flow_pattern_sorted)):
                    pattern_data = flow_pattern_symbols.get(flow_pattern, {'symbol': 'x', 'color': 'gray'})
                    symbol = pattern_data['symbol']
                    color = pattern_data['color']
                    ax.scatter(jg_val, alpha_val,c=color, marker=symbol, s=120, 
                             edgecolors='black', linewidth=1, zorder=2)
                    
                
                
                print(f"Plotando série jL = {jl:.1f} m/s com {len(jg_clean)} pontos")
        
        # Adicionar legenda para Flow Patterns (apenas os que aparecem nos dados)
        used_patterns = set()
        for pattern in df_plot[flow_pattern_col].dropna().unique():
            if pattern in flow_pattern_symbols:
                used_patterns.add(pattern)
        
        legend_elements = []
        for pattern in sorted(used_patterns):
            pattern_data = flow_pattern_symbols.get(flow_pattern, {'symbol': 'x', 'color': 'gray'})
            symbol = pattern_data['symbol']
            color = pattern_data['color']
            legend_elements.append(plt.Line2D([0], [0], marker=symbol, color='w', 
                                            markerfacecolor=color, markersize=10, 
                                            markeredgecolor='black', markeredgewidth=1,
                                            label=pattern))
        
        x_label = r'$j_{g} [m/s]$'
        y_label = r'Void fraction ($\alpha_g$)'
        # Configurar eixos com fonte acadêmica
        ax.set_xlabel(x_label, fontsize=24, fontfamily='serif')
        ax.set_ylabel(y_label, fontsize=24, fontfamily='serif')
        # Remover título conforme solicitado
        
        # Configurar grade refinada
        # ax.grid(True, which='major', alpha=0.5, linestyle='-', linewidth=0.8, color='gray')
        # ax.grid(True, which='minor', alpha=0.5, linestyle=':', linewidth=0.8, color='lightgray')
        ax.set_axisbelow(True)
        
        # Configurar ticks menores para grade mais detalhada
        ax.minorticks_on()
        
        # Configurar espaçamento dos ticks principais
        ax.xaxis.set_major_locator(MultipleLocator(2.0))
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        
        # Configurar espaçamento dos ticks menores
        ax.xaxis.set_minor_locator(MultipleLocator(1.0))
        ax.yaxis.set_minor_locator(MultipleLocator(0.2))
        
        # Configurar limites dos eixos
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0, top=1)
        
        # Configurar tamanho dos ticks com fonte acadêmica
        ax.tick_params(axis='both', which='major', labelsize=18)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily('serif')
        
        # Adicionar legenda para jL
        jl_legend_elements = []
        for i, jl in enumerate(jl_series):
            color = colors[i % len(colors)]
            print(i)
            line_style = line_styles[i % len(line_styles)]
            jl_legend_elements.append(plt.Line2D([0], [0], color='black', linestyle=line_style, marker=vec_s_lines[i], mfc='silver',
                                               markersize=10, label=rf'$j_{{l}}$ = {jl:.1f} m/s'))
        # Criar duas legendas
        ax.legend(handles=jl_legend_elements + legend_elements, 
                 loc='lower right', frameon=True, fancybox=True, shadow=True, fontsize=18,
                 prop={'family': 'serif'})
        
        # Ajustar layout
        plt.tight_layout()
        
        # Salvar figura no diretório do file_path
        output_dir = os.path.dirname(file_path)
        if output_dir:
            base_name = f"{Path(file_path).stem}_{sheet_name}_jg_vs_alpha"
            pdf_file = os.path.join(output_dir, f"{base_name}.pdf")
            png_file = os.path.join(output_dir, f"{base_name}.png")
        else:
            base_name = f"{Path(file_path).stem}_{sheet_name}_jg_vs_alpha"
            pdf_file = f"{base_name}.pdf"
            png_file = f"{base_name}.png"
        
        # Salvar em PDF (ideal para LaTeX)
        plt.savefig(pdf_file, format='pdf', dpi=300, bbox_inches='tight', 
                   pad_inches=0.1, facecolor='white', edgecolor='none')
        print(f"Plot PDF salvo: {pdf_file}")
        
        # Salvar também em PNG (para visualização)
        plt.savefig(png_file, format='png', dpi=300, bbox_inches='tight', 
                   pad_inches=0.1, facecolor='white', edgecolor='none')
        print(f"Plot PNG salvo: {png_file}")
        
        # Mostrar figura
        plt.show()
        
    except Exception as e:
        print(f"Erro ao gerar plot: {e}")
        import traceback
        traceback.print_exc()

def main():
    """
    Função principal que executa o programa.
    """
    print("=== LEITOR DE ARQUIVOS EXCEL ===")
    
    # Ler o arquivo Excel
    df, units = read_excel_file(file_path)
    
    if df is not None:
        # Se retornou um dicionário (múltiplas abas)
        if isinstance(df, dict):
            print(f"Arquivo com {len(df)} abas carregado")
            for sheet_name, sheet_df in df.items():
                analyze_dataframe(sheet_df)
                
                # Salvar DataFrame em arquivo .txt
                output_file = f"{Path(file_path).stem}_{sheet_name}_dados.txt"
                save_dataframe_to_txt(sheet_df, units[sheet_name], output_file, sheet_name)
        else:
            # DataFrame único
            analyze_dataframe(df)
            
            # Salvar DataFrame em arquivo .txt
            output_file = f"{Path(file_path).stem}_dados.txt"
            save_dataframe_to_txt(df, units, output_file)
            
        # Salvar informações em um arquivo de log
        log_file = f"excel_analysis_log_{Path(file_path).stem}.txt"
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"Análise do arquivo: {file_path}\n")
            f.write(f"Data/hora: {pd.Timestamp.now()}\n\n")
            
            if isinstance(df, dict):
                for sheet_name, sheet_df in df.items():
                    f.write(f"=== ABA: {sheet_name} ===\n")
                    f.write(f"Dimensões: {sheet_df.shape}\n")
                    f.write(f"Colunas: {list(sheet_df.columns)}\n")
                    f.write(f"Unidades: {units[sheet_name]}\n\n")
            else:
                f.write(f"Dimensões: {df.shape}\n")
                f.write(f"Colunas: {list(df.columns)}\n")
                f.write(f"Unidades: {units}\n\n")
        
        print(f"Arquivos salvos: {output_file}, {log_file}")
        
        # Gerar plot científico jG vs α
        if isinstance(df, dict):
            for sheet_name, sheet_df in df.items():
                generate_jg_vs_alpha_plot(sheet_df, sheet_name)
        else:
            generate_jg_vs_alpha_plot(df, "Data")
        
    else:
        print("Falha ao carregar o arquivo Excel.")

    print(df.head())
if __name__ == "__main__":
    main()
