import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from CoolProp.CoolProp import PropsSI
import os
import sys
from pathlib import Path
import warnings
from contextlib import redirect_stderr

# Suprimir avisos específicos do pandas
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')
warnings.filterwarnings('ignore', category=UserWarning, module='pandas')
# Suprimir avisos de timestamp ao salvar PDF/PNG (backend matplotlib/pdf)
warnings.filterwarnings('ignore', message=r".*timestamp seems very low.*")
warnings.filterwarnings('ignore', message=r".*regarding as unix timestamp.*")

####################################################################################################################################################
#                                            INPUTS
####################################################################################################################################################
file_path = 'data_example/example/NAS/Experimental_Results_v25_NAS_19_feb_2026_revA.xlsm'  # Insira o caminho do arquivo a ser analisado NOTE: USE SEMPRE A BARRA NORMAL '/', SE ESTIVER INVERTIDA, MODIFIQUE-A

# Flag para indicar leitura de arquivo NAS processado (_processed_all_sheets.xlsx)
NAS_file = True

# Abas válidas para arquivos NAS (as demais serão ignoradas)
ALLOWED_SHEETS_NAS = {
    'AWH00', 'AWU05', 'AWU90',
    'AWD05', 'AWD15E', 'AWD30', 'AWD45', 'AWD60', 'AWD60E', 'AWD85', 'AWD90',
    'AOH00', 'AOU05', 'AOU90',
    'AOD05', 'AOD15E', 'AOD30', 'AOD45', 'AOD45E', 'AOD60', 'AOD60E', 'AOD85', 'AOD90',
    'SOH00', 'SOU05', 'SOU90',
    'SOD05', 'SOD15', 'SOD45', 'SOD60', 'SOD85', 'SOD90',
    'ADH00', 'ADU05', 'ADU90',
    'ADD05', 'ADD15', 'ADD45', 'ADD60', 'ADD85', 'ADD90',
}

# Mapeamento de códigos curtos de flow pattern (formato NAS) para os nomes usados nos gráficos
NAS_FLOW_PATTERN_MAP = {
    'AN': 'Annular',
    'SL': 'Slug',
    'CH': 'Churn',
    'SW': 'Stratified wavy',
    'ST': 'Stratified',
    'SS': 'Stratified Smooth',
    'EB': 'Elongated bubble',
    # Variantes com maiúscula/minúscula (normalizadas por strip().upper() na leitura)
    'ANNULAR': 'Annular',
    'SLUG': 'Slug',
    'CHURN': 'Churn',
    'STRATIFIED WAVY': 'Stratified wavy',
    'STRATIFIED': 'Stratified',
    'STRATIFIED SMOOTH': 'Stratified Smooth',
    'ELONGATED BUBBLE': 'Elongated bubble',
}

def standardize_liquid_conditions(
    all_dataframes: dict,
    *,
    target_jl_levels=(0.2, 0.4, 0.8, 1.6),
    jl_group_size=4,
    jl_tolerance=0.05,
    D=0.05251,
):
    """
    Padroniza a condição do líquido para todas as inclinações/abas.
    
    - Força jL em 4 níveis (P01–P04, P05–P08, P09–P12, P13–P16) ou, quando a
      contagem variar, aplica em blocos sucessivos de `jl_group_size`.
    - Usa propriedades termofísicas do fluido líquido (Water/Oil/SF6, etc.)
      em 25°C e 1 atm para calcular o número de Reynolds superficial do líquido Re_sl.
    - Grava/atualiza colunas:
      - Re_sl_raw: Re_sl calculado ponto a ponto com jL padronizado
      - Re_sl_group: Re_sl constante por bloco (igual ao Re_sl do nível jL do bloco)
      - jL_group_id: id do bloco (0,1,2,...)
    """
    if not all_dataframes:
        return all_dataframes

    for sheet_name, df in all_dataframes.items():
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            continue

        # Encontrar coluna de jL (com tolerância a espaços)
        col_mapping = {}
        for col in list(df.columns):
            if pd.notna(col):
                col_mapping[str(col).strip()] = col
        if 'jL' not in col_mapping:
            print(f"Aviso: coluna 'jL' não encontrada em {sheet_name}; padronização de líquido ignorada.")
            continue

        jl_col = col_mapping['jL']
        n = len(df)
        group_ids = (np.arange(n) // jl_group_size).astype(int)
        df['jL_group_id'] = group_ids

        # Padronizar jL por grupo sequencial (p01–p04, p05–p08, ...)
        for g in np.unique(group_ids):
            mask_g = group_ids == g
            jl_vals = pd.to_numeric(df.loc[mask_g, jl_col], errors='coerce')
            jl_mean = jl_vals.mean()

            # Selecionar nível alvo: grupos 0..3 mapeiam para (0.2, 0.4, 0.8, 1.6)
            # Se houver mais grupos, reutiliza o último nível (mais conservador) por padrão.
            if g < len(target_jl_levels):
                target_jl = float(target_jl_levels[g])
            else:
                target_jl = float(target_jl_levels[-1])

            # Checagem de consistência: se a média do grupo estiver muito longe do alvo, avisar
            if pd.notna(jl_mean) and abs(float(jl_mean) - target_jl) > jl_tolerance:
                print(
                    f"Aviso: {sheet_name} grupo {g}: média jL={jl_mean:.3f} m/s distante do alvo "
                    f"{target_jl:.3f} m/s (tol={jl_tolerance}). Sobrescrevendo mesmo assim."
                )

            df.loc[mask_g, jl_col] = target_jl

        # Identificar fluido líquido a partir do nome da aba (mesma convenção de extract_info_from_filename)
        try:
            _, fluid_2, _, _, _, _ = extract_info_from_filename(sheet_name)
        except Exception:
            fluid_2 = 'Water'

        # Densidade e viscosidade do líquido a 25°C e 1 atm
        temp_c = 25.0
        temp_k = temp_c + 273.15
        if fluid_2 == 'Water':
            rho_L_fixed = PropsSI('D', 'P', 101325, 'T', temp_k, 'Water')
            mu_L_fixed = PropsSI('V', 'P', 101325, 'T', temp_k, 'Water')
        elif fluid_2 == 'Oil':
            # Mesmo modelo usado em exp_unc.py (em Pa·s)
            rho_L_fixed = 0.0008 * temp_c**2 - 0.698 * temp_c + 879.154
            mu_cp = 0.031267 * (temp_c**2) - 3.2050 * temp_c + 97.6594
            mu_L_fixed = mu_cp * 1e-3
        else:
            # Tentar usar CoolProp diretamente para outros líquidos
            try:
                rho_L_fixed = PropsSI('D', 'P', 101325, 'T', temp_k, fluid_2)
                mu_L_fixed = PropsSI('V', 'P', 101325, 'T', temp_k, fluid_2)
            except Exception:
                rho_L_fixed = PropsSI('D', 'P', 101325, 'T', temp_k, 'Water')
                mu_L_fixed = PropsSI('V', 'P', 101325, 'T', temp_k, 'Water')

        # Número de Reynolds superficial do líquido (Re_sl) com jL padronizado (constante por grupo)
        jl_numeric = pd.to_numeric(df[jl_col], errors='coerce')
        Re_sl_raw = rho_L_fixed * jl_numeric * D / mu_L_fixed
        df['Re_sl_raw'] = Re_sl_raw
        
        # Re_sl_group: constante dentro do grupo (média do grupo após padronização)
        Re_sl_group = df.groupby('jL_group_id')['Re_sl_raw'].transform('mean')
        df['Re_sl_group'] = Re_sl_group

    return all_dataframes

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
        if not file_path.lower().endswith(('.xlsx', '.xls', '.xlsm')):
            print(f"Erro: Arquivo '{file_path}' não é um arquivo Excel válido.")
            return None, None
            
        print(f"Lendo arquivo Excel: {file_path}")
        
        # Tentar ler o arquivo Excel
        try:
            # Primeiro, tentar ler todas as abas
            excel_file = pd.ExcelFile(file_path)
            all_sheets = excel_file.sheet_names

            # Para arquivos NAS, considerar apenas as abas da lista ALLOWED_SHEETS_NAS
            if NAS_file:
                sheet_names = [s for s in all_sheets if s in ALLOWED_SHEETS_NAS]
                if not sheet_names:
                    sheet_names = all_sheets  # fallback de segurança
            else:
                sheet_names = all_sheets

            print(f"Abas encontradas: {all_sheets}")
            print(f"Abas consideradas para processamento: {sheet_names}")
            
            # Se houver múltiplas abas, perguntar qual usar
            if len(sheet_names) > 1:
                print("\nMúltiplas abas encontradas:")
                for i, sheet in enumerate(sheet_names):
                    print(f"{i+1}: {sheet}")
                
                while True:
                    try:
                        choice = input(f"\nEscolha a aba (1-{len(sheet_names)}) ou 'all' para todas: ").strip()
                        if choice.lower() == 'all':
                            # Perguntar quais abas específicas o usuário quer processar
                            print("\nVocê escolheu 'all'. Agora selecione quais abas específicas deseja processar:")
                            selected_sheets = get_user_sheet_selection(excel_file, sheet_names)
                            
                            if not selected_sheets:
                                print("Nenhuma aba selecionada. Retornando ao menu principal.")
                                continue
                            
                            # Ler apenas as abas selecionadas
                            dataframes = {}
                            units_dict = {}
                            for sheet in selected_sheets:
                                print(f"Lendo aba: {sheet}")
                                if NAS_file:
                                    df, units = read_single_sheet_nas(file_path, sheet)
                                else:
                                    df, units = read_single_sheet(file_path, sheet)
                                if df is not None and units is not None:
                                    dataframes[sheet] = df
                                    units_dict[sheet] = units
                                else:
                                    print(f"Erro ao ler aba {sheet}, pulando...")
                            
                            if dataframes:
                                return dataframes, units_dict, True  # True indica que foi escolhido 'all'
                            else:
                                print("Nenhuma aba foi lida com sucesso.")
                                continue
                        else:
                            sheet_index = int(choice) - 1
                            if 0 <= sheet_index < len(sheet_names):
                                sheet_name = sheet_names[sheet_index]
                                print(f"Lendo aba: {sheet_name}")
                                if NAS_file:
                                    df, units = read_single_sheet_nas(file_path, sheet_name)
                                else:
                                    df, units = read_single_sheet(file_path, sheet_name)
                                break
                            else:
                                print("Escolha inválida. Tente novamente.")
                    except ValueError:
                        print("Entrada inválida. Digite um número ou 'all'.")
            else:
                # Apenas uma aba
                sheet_name = sheet_names[0]
                print(f"Lendo aba: {sheet_name}")
                if NAS_file:
                    df, units = read_single_sheet_nas(file_path, sheet_name)
                else:
                    df, units = read_single_sheet(file_path, sheet_name)
                
        except Exception as e:
            print(f"Erro ao ler arquivo Excel: {e}")
            return None, None
        
        fluid_1, fluid_2, direction, theta, ID, is_validation = extract_info_from_filename(sheet_name)
        print(f"\nInformações extraídas do nome do arquivo:")
        print(f"Fluido 1: {fluid_1}")
        print(f"Fluido 2: {fluid_2}")
        print(f"Direção: {direction}")
        print(f"Inclinação (theta): {theta}°")
        print(f"Ponto de validação: {'Sim' if is_validation else 'Não'}")
        # Exibir informações básicas do DataFrame
        print(f"DataFrame carregado: {df.shape[0]} linhas x {df.shape[1]} colunas")
        if direction == 'Downward':
            theta =-theta
        
        return df, units, sheet_name, fluid_1, fluid_2, theta, False  # False indica que não foi escolhido 'all'
        
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


def read_single_sheet_nas(file_path, sheet_name):
    """
    Lê uma aba específica do arquivo NAS, cujo formato é:
    - Linha 4 (1-indexada): nomes das colunas (B até W)
    - Linha 5: unidades
    - Linhas 6 a 21: pontos experimentais

    Args:
        file_path (str): Caminho para o arquivo Excel
        sheet_name (str): Nome da aba

    Returns:
        pd.DataFrame: DataFrame com os dados
        dict: Dicionário com as unidades das colunas
    """
    try:
        # Ler as primeiras 5 linhas para obter nomes das colunas e unidades (B:W)
        df_header = pd.read_excel(
            file_path,
            sheet_name=sheet_name,
            header=None,
            nrows=5,
            usecols="B:W",
        )

        # Nomes das colunas na linha 4 (índice 3, 0-indexado)
        column_names = df_header.iloc[3].tolist()

        # Unidades na linha 5 (índice 4)
        units = df_header.iloc[4].tolist()

        # Criar dicionário de unidades
        units_dict = {}
        for col_name, unit in zip(column_names, units):
            if pd.notna(col_name) and pd.notna(unit):
                units_dict[col_name] = unit
            elif pd.notna(col_name):
                units_dict[col_name] = ""

        # Ler dados das linhas 6 a 21 (16 linhas), colunas B:W
        df = pd.read_excel(
            file_path,
            sheet_name=sheet_name,
            header=None,
            skiprows=5,   # pula 5 linhas (1–5), começa na linha 6
            nrows=16,     # 6–21
            usecols="B:W",
        )

        # Definir nomes das colunas
        df.columns = column_names

        # Normalizar nomes importantes para se adequarem ao restante da rotina:
        # - 'JL' (NAS) -> 'jL'
        # - 'JG' (NAS) -> 'jG'
        # - '-dP/dz F' (NAS) -> 'dp/dz_F'
        # - '-dP/dz T' (se existir) -> 'dp/dz_T'
        # - 'Alpha' / 'Void fraction' -> 'α'
        # - 'FP' ou 'Flow pattern' -> 'Flow Pattern'
        # - 'T' ou 'Temp' -> 'Temp.'
        # - 'Pressure' / 'Gauge P' ou 'P' -> 'Gauge Pressure' (para cálculo de Re_sg)
        def _nas_clean(s):
            if not isinstance(s, str):
                return ""
            return s.strip().replace('\n', '').replace('\r', '').strip()
        rename_map = {}
        for col in df.columns:
            clean = _nas_clean(col)
            if clean:
                if clean == 'JL':
                    rename_map[col] = 'jL'
                elif clean == 'JG':
                    rename_map[col] = 'jG'
                elif clean == '-dP/dz F':
                    rename_map[col] = 'dp/dz_F'
                elif clean == '-dP/dz T':
                    rename_map[col] = 'dp/dz_T'
                elif clean in ('Alpha', 'alpha', 'Void fraction', 'Void Fraction', 'void fraction', 'α'):
                    rename_map[col] = 'α'
                elif clean in ('FP', 'Flow pattern'):
                    rename_map[col] = 'Flow Pattern'
                elif clean in ('T', 'Temp', 'Temp.', 'T (C)', 'T(°C)'):
                    rename_map[col] = 'Temp.'
                elif clean in ('Pressure', 'P', 'Gauge P', 'Gauge P.', 'Gauge P (kPa)', 'P (kPa)'):
                    rename_map[col] = 'Gauge Pressure'
                # Fallback: variantes comuns no NAS (ex.: "Gauge P. " com espaço, "JG " etc.)
                elif clean.upper() == 'JG':
                    rename_map[col] = 'jG'
                elif clean.upper().startswith('TEMP') or clean == 'T':
                    rename_map[col] = 'Temp.'
                elif clean.upper() == 'PRESSURE' or (clean.upper().startswith('GAUGE') and 'P' in clean.upper()):
                    rename_map[col] = 'Gauge Pressure'

        if rename_map:
            df = df.rename(columns=rename_map)
            # Atualizar também o dicionário de unidades para manter consistência
            new_units_dict = {}
            for col_name, unit in units_dict.items():
                new_name = rename_map.get(col_name, col_name)
                new_units_dict[new_name] = unit
            units_dict = new_units_dict

        # Interpretar flow pattern em curto (NAS) e converter para nomes completos
        fp_col = None
        for c in df.columns:
            if isinstance(c, str) and c.strip() == 'Flow Pattern':
                fp_col = c
                break
        if fp_col is not None:
            def _nas_fp_to_full(val):
                if pd.isna(val):
                    return val
                s = str(val).strip()
                key = s.upper()
                return NAS_FLOW_PATTERN_MAP.get(key, NAS_FLOW_PATTERN_MAP.get(s, val))
            df[fp_col] = df[fp_col].apply(_nas_fp_to_full)

        # Remover linhas totalmente vazias
        df = df.dropna(how='all').reset_index(drop=True)

        return df, units_dict

    except Exception as e:
        print(f"Erro ao ler aba NAS {sheet_name}: {e}")
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

def setup_plot_style():
    """Configura o estilo científico para os plots"""
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 20,
        'font.family': 'serif',
        'text.usetex': False,
        'axes.linewidth': 1.0,
        'axes.grid': False,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
        'mathtext.fontset': 'cm'
    })

def get_flow_pattern_symbols():
    """Retorna o dicionário de símbolos para Flow Patterns baseado nos dados atuais do Excel"""
    return {
        # Padrões encontrados no arquivo Excel atual (análise detalhada)
        'Annular': {'symbol': 'o', 'color': 'blue'},              # Annular
        'Churn': {'symbol': 's', 'color': 'yellow'},               # Churn
        'Elongated bubble': {'symbol': 'o', 'color': 'lightblue'}, # Elongated bubble (minúsculo)
        'Stratified Smooth': {'symbol': 'v', 'color': 'purple'},                 # Stratified Smooth
        'Stratified': {'symbol': '^', 'color': 'red'},                    # Stratified
        'Stratified wavy': {'symbol': '^', 'color': 'orange'},                 # Stratified Wavy
        'Slug': {'symbol': 's', 'color': 'darkgreen'},            # Slug
        'Stratified': {'symbol': '^', 'color': 'darkred'},        # Stratified
        
        # # Padrões legados (mantidos para compatibilidade)
        # 'Elongated Bubble': {'symbol': 'o', 'color': 'lightblue'}, # Versão com maiúscula
        # 'slug': {'symbol': 's', 'color': 'green'},
        # 'Estratificado': {'symbol': 'D', 'color': 'orange'},
        # 'SW/MI': {'symbol': 'p', 'color': 'cyan'},
        # 'Intermittent (Slug Wavy)²': {'symbol': 'h', 'color': 'magenta'},
        # 'Intermittent (Slug Wavy) ²': {'symbol': 'h', 'color': 'magenta'},
        # 'Intermittent (Slug Wavy)¹ ²': {'symbol': 'H', 'color': 'brown'},
        # 'Rolling Wave²': {'symbol': 'D', 'color': 'pink'},
        # 'DBubbly': {'symbol': 'P', 'color': 'gray'},
        # 'SL (I)': {'symbol': 's', 'color': 'maroon'},
        # 'annular': {'symbol': 'o', 'color': 'blue'},
        # 'churn': {'symbol': 's', 'color': 'green'},
        # 'Churn/bolhas?': {'symbol': 's', 'color': 'green'},
        # 'Churn?': {'symbol': 's', 'color': 'green'},
        # 'bubbles': {'symbol': 'o', 'color': 'blue'},
        # 'Intermittent': {'symbol': 'h', 'color': 'magenta'},
        # 'A': {'symbol': 'o', 'color': 'blue'},                    # Annular abreviado
        # 'AN': {'symbol': 'o', 'color': 'blue'},                   # Annular abreviado
        # 'CH': {'symbol': 's', 'color': 'green'},                  # Churn abreviado
        # 'SL': {'symbol': 's', 'color': 'darkgreen'},              # Slug abreviado
        # 'sw': {'symbol': '^', 'color': 'orange'}                 # Stratified Wavy minúsculo
    }

def generate_alpha_vs_jg_plot(df, sheet_name, fluid_1, fluid_2, theta):
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
        
        # Coluna de void fraction: aceitar vários nomes (NAS e formato inicial)
        alpha_col_key = None
        for name in ('α', 'Alpha', 'alpha', 'Void fraction', 'Void Fraction', 'void fraction'):
            if name in col_mapping:
                alpha_col_key = name
                break
        if alpha_col_key is None:
            # Tentar por similaridade (strip e case)
            for col in available_cols:
                if col is None or pd.isna(col):
                    continue
                c = str(col).strip().lower()
                if c in ('α', 'alpha', 'void fraction') or c.replace(' ', '') == 'voidfraction':
                    alpha_col_key = str(col).strip()
                    col_mapping['α'] = col  # passar a usar 'α' daqui pra frente
                    break
        if alpha_col_key is None:
            missing_cols = ['α (ou Alpha / Void fraction)']
        else:
            if alpha_col_key != 'α':
                col_mapping['α'] = col_mapping[alpha_col_key]
            missing_cols = []
            for req_col in ['jG', 'jL', 'Flow Pattern']:
                if req_col not in col_mapping:
                    missing_cols.append(req_col)

        if missing_cols:
            print(f"Colunas ausentes para plot jG vs α: {missing_cols}")
            print(f"Colunas disponíveis: {list(col_mapping.keys())}")
            return

        # Configurar estilo científico para LaTeX
        setup_plot_style()

        # Criar figura
        fig, ax = plt.subplots(figsize=(10, 10))  # Figura quadrada 10x10

        # Obter valores únicos de jL e agrupar por séries
        jl_col = col_mapping['jL']
        jg_col = col_mapping['jG']
        alpha_col = col_mapping['α']
        flow_pattern_col = col_mapping['Flow Pattern']
        
        # Agrupar dados por jL (arredondando para 1 casa decimal para agrupar séries similares)
        df_plot = df.copy()
        # Garantir α numérico (NAS pode trazer como string)
        df_plot[alpha_col] = pd.to_numeric(df_plot[alpha_col], errors='coerce')
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
        symbol_line = ['o','h', 'p', 'D', 's', '^', 'v' ]
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
        #     'Intermittent (Slug Wavy)²': 'h',  # Hexágono
        #     'Intermittent (Slug Wavy) ²': 'h', # Hexágono
        #     'Intermittent (Slug Wavy)¹ ²': 'H', # Hexágono grande
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
        #     'Intermittent¹ ²': 'h'         # Hexágono
        # }
        
        flow_pattern_symbols = get_flow_pattern_symbols()
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
                ax.plot(jg_sorted, alpha_sorted, line_style, marker=s_lines, markersize=20, color='black', mfc='silver', linewidth=1.5, zorder=1)
                # ax.plot(jg_sorted, alpha_sorted, line_style, marker=s_lines, markersize=16, color='black', mfc='silver', linewidth=1.5, zorder=1)

                # Plotar cada ponto com símbolo baseado no Flow Pattern
                for j, (jg_val, alpha_val, flow_pattern) in enumerate(zip(jg_sorted, alpha_sorted, flow_pattern_sorted)):
                    pattern_data = flow_pattern_symbols.get(flow_pattern, {'symbol': 'o', 'color': 'gray'})
                    symbol = pattern_data['symbol']
                    color = pattern_data['color']
                    ax.scatter(jg_val, alpha_val, c=color, marker=symbol, s=100, 
                             edgecolors='black', linewidth=1, zorder=2)
                    
                
                
                print(f"Plotando série jL = {jl:.1f} m/s com {len(jg_clean)} pontos")
        
        # Adicionar legenda para Flow Patterns (apenas os que aparecem nos dados)
        used_patterns = set()
        for pattern in df_plot[flow_pattern_col].dropna().unique():
            if pattern in flow_pattern_symbols:
                used_patterns.add(pattern)
        print('used_patterns',used_patterns)
        legend_elements = []
        for pattern in sorted(used_patterns):
            pattern_data = flow_pattern_symbols.get(pattern, {'symbol': 'o', 'color': 'gray'})
            symbol = pattern_data['symbol']
            color = pattern_data['color']
            print(symbol, color)
            legend_elements.append(plt.Line2D([0], [0], marker=symbol, color='w', 
                                            markerfacecolor=color, markersize=8, 
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
        ax.xaxis.set_major_locator(MultipleLocator(1.0))
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        
        # Configurar espaçamento dos ticks menores
        ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        # ax.yaxis.set_minor_locator(MultipleLocator(0.05))
        
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
                                               markersize=12, label=rf'$j_{{l}}$ = {jl:.1f} m/s'))
        # Criar duas legendas
        ax.legend(handles=jl_legend_elements + legend_elements, 
                 loc='lower right', frameon=True, fancybox=True, shadow=True, fontsize=22,
                 prop={'family': 'serif'})
        
        plt.text(2, 0.1, rf'$\theta = {theta}^\circ$', fontsize=20)

        # Ajustar layout
        plt.tight_layout()
        
        # Salvar figura no diretório do file_path/SheetName
        output_dir = os.path.dirname(file_path)
        sheet_dir = os.path.join(output_dir, sheet_name)
        os.makedirs(sheet_dir, exist_ok=True)
        base_name = f"{sheet_name}_alpha_vs_jg"
        pdf_file = os.path.join(sheet_dir, f"{base_name}.pdf")
        png_file = os.path.join(sheet_dir, f"{base_name}.png")

        # Salvar em PDF (ideal para LaTeX)
        plt.savefig(pdf_file, format='pdf', dpi=300, bbox_inches='tight', 
                   pad_inches=0.1, facecolor='white', edgecolor='none')
        print(f"Plot PDF salvo: {pdf_file}")
        
        # Salvar também em PNG (para visualização)
        plt.savefig(png_file, format='png', dpi=300, bbox_inches='tight', 
                   pad_inches=0.1, facecolor='white', edgecolor='none')
        print(f"Plot PNG salvo: {png_file}")
        
        # Mostrar figura
        # plt.show()
        
    except Exception as e:
        print(f"Erro ao gerar plot: {e}")
        import traceback
        traceback.print_exc()

def generate_dpdzf_vs_jg_plot(df, sheet_name, fluid_1, fluid_2, theta):
    """
    Gera um plot científico de jG vs α, onde cada jL é uma série diferente.
    Inclui símbolos diferentes para cada Flow Pattern.
    
    Args:
        df (pd.DataFrame): DataFrame com os dados
        sheet_name (str): Nome da aba/sheet
    """
    try:
        # Verificar se as colunas necessárias existem (com possíveis espaços)
        required_cols = ['jG', 'jL', 'dp/dz_F', 'Flow Pattern']
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
        setup_plot_style()
        
        # Criar figura
        fig, ax = plt.subplots(figsize=(10, 10))  # Figura quadrada 10x10
        
        # Obter valores únicos de jL e agrupar por séries
        jl_col = col_mapping['jL']
        jg_col = col_mapping['jG']
        dp_dz_f_col = col_mapping['dp/dz_F']
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
        symbol_line = ['o','h', 'p', 'D', 's', '^', 'v' ]
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
        #     'Intermittent (Slug Wavy)²': 'h',  # Hexágono
        #     'Intermittent (Slug Wavy) ²': 'h', # Hexágono
        #     'Intermittent (Slug Wavy)¹ ²': 'H', # Hexágono grande
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
        #     'Intermittent¹ ²': 'h'         # Hexágono
        # }
        
        flow_pattern_symbols = get_flow_pattern_symbols()
        vec_s_lines = []

        # Plotar cada série de jL
        for i, jl in enumerate(jl_series):
            # Filtrar dados para esta série de jL
            mask = df_plot['jL_rounded'] == jl
            
            jg_data = df_plot.loc[mask, jg_col]
            frictional_data = df_plot.loc[mask, dp_dz_f_col]
            flow_pattern_data = df_plot.loc[mask, flow_pattern_col]
            
            # Remover valores nulos
            valid_mask = pd.notna(jg_data) & pd.notna(frictional_data) & pd.notna(flow_pattern_data)
            jg_clean = jg_data[valid_mask]
            frictional_clean = frictional_data[valid_mask]/1000     #To plot in kPa
            flow_pattern_clean = flow_pattern_data[valid_mask]
            if len(jg_clean) > 0:
                color = colors[i % len(colors)]
                line_style = line_styles[i % len(line_styles)] # Escolher estilo de linha
                s_lines = symbol_line[i % len(symbol_line)]
                vec_s_lines.append(s_lines)
                # Ordenar dados por jG para conectar corretamente
                sorted_data = sorted(zip(jg_clean, frictional_clean, flow_pattern_clean))
                jg_sorted = [x[0] for x in sorted_data]
                frictional_sorted = [x[1] for x in sorted_data]
                flow_pattern_sorted = [x[2] for x in sorted_data]
                
                
                    
                # Plotar linha conectando os pontos (todas pretas, estilo varia por série)
                ax.plot(jg_sorted, frictional_sorted, line_style, marker=s_lines, markersize=20, color='black', mfc='silver', linewidth=1.5, zorder=1)
                # ax.plot(jg_sorted, frictional_sorted, line_style, marker=s_lines, markersize=16, color='black', mfc='silver', linewidth=1.5, zorder=1)

                # Plotar cada ponto com símbolo baseado no Flow Pattern
                for j, (jg_val, frictional_val, flow_pattern) in enumerate(zip(jg_sorted, frictional_sorted, flow_pattern_sorted)):
                    pattern_data = flow_pattern_symbols.get(flow_pattern, {'symbol': 'o', 'color': 'gray'})
                    symbol = pattern_data['symbol']
                    color = pattern_data['color']
                    ax.scatter(jg_val, frictional_val, c=color, marker=symbol, s=100, 
                             edgecolors='black', linewidth=1, zorder=2)
                    
                
                
                print(f"Plotando série jL = {jl:.1f} m/s com {len(jg_clean)} pontos")
        
        # Adicionar legenda para Flow Patterns (apenas os que aparecem nos dados)
        used_patterns = set()
        for pattern in df_plot[flow_pattern_col].dropna().unique():
            if pattern in flow_pattern_symbols:
                used_patterns.add(pattern)
        print('used_patterns',used_patterns)
        legend_elements = []
        for pattern in sorted(used_patterns):
            pattern_data = flow_pattern_symbols.get(pattern, {'symbol': 'o', 'color': 'gray'})
            symbol = pattern_data['symbol']
            color = pattern_data['color']
            print(symbol, color)
            legend_elements.append(plt.Line2D([0], [0], marker=symbol, color='w', 
                                            markerfacecolor=color, markersize=8, 
                                            markeredgecolor='black', markeredgewidth=1,
                                            label=pattern))
        
        x_label = r'$j_{g} [m/s]$'
        y_label = r'$\left(\frac{\partial P}{\partial z}\right)_{\text{frictional}} \; \left[\frac{\text{kPa}}{\text{m}}\right]$'

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
        ax.xaxis.set_major_locator(MultipleLocator(1.0))
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        
        # Configurar espaçamento dos ticks menores
        ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        # ax.yaxis.set_minor_locator(MultipleLocator(0.05))
        
        # Configurar limites dos eixos
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        
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
                                               markersize=12, label=rf'$j_{{l}}$ = {jl:.1f} m/s'))
        # Criar duas legendas
        # ax.legend(handles=jl_legend_elements + legend_elements, 
        #          loc='lower right', frameon=True, fancybox=True, shadow=True, fontsize=22,
        #          prop={'family': 'serif'})
        
        # Criar legenda única estilo painel em cima
        ax.legend(
            handles=jl_legend_elements + legend_elements,
            loc='lower center',
            bbox_to_anchor=(0.5, 1.02),   # coloca acima do gráfico
            ncol=3,                       # nº de colunas
            frameon=False,                # sem borda
            fontsize=16,                  # fonte menor (estilo artigo)
            prop={'family': 'serif'}      # fonte serifada
        )
        
        # Adicionar texto de inclinação apenas se houver dados
        if 'frictional_sorted' in locals() and frictional_sorted:
            plt.text(1, 7*max(frictional_sorted)/8, rf'$\theta = {theta}^\circ$', fontsize=20)

        # Ajustar layout
        plt.tight_layout()
        
        # Salvar figura no diretório do file_path/SheetName
        output_dir = os.path.dirname(file_path)
        sheet_dir = os.path.join(output_dir, sheet_name)
        os.makedirs(sheet_dir, exist_ok=True)
        base_name = f"{sheet_name}_frictional_vs_jg"
        pdf_file = os.path.join(sheet_dir, f"{base_name}.pdf")
        png_file = os.path.join(sheet_dir, f"{base_name}.png")
        
        # Salvar em PDF (ideal para LaTeX)
        plt.savefig(pdf_file, format='pdf', dpi=300, bbox_inches='tight', 
                   pad_inches=0.1, facecolor='white', edgecolor='none')
        print(f"Plot PDF salvo: {pdf_file}")
        
        # Salvar também em PNG (para visualização)
        plt.savefig(png_file, format='png', dpi=300, bbox_inches='tight', 
                   pad_inches=0.1, facecolor='white', edgecolor='none')
        print(f"Plot PNG salvo: {png_file}")
        
    except Exception as e:
        print(f"Erro ao gerar plot: {e}")
        import traceback
        traceback.print_exc()

def generate_dpdzf_vs_Reg_plot(df, sheet_name, fluid_1, fluid_2, theta):
    """
    Gera um plot científico de Re_sg vs dp/dz_F, onde cada Re_sl é uma série diferente.
    Inclui símbolos diferentes para cada Flow Pattern.
    """
    try:
        required_cols = ['jG', 'jL', 'dp/dz_F', 'Flow Pattern', 'Temp.', 'Gauge Pressure']
        available_cols = list(df.columns)
        
        col_mapping = {}
        for col in available_cols:
            if pd.notna(col):
                clean_col = str(col).strip()
                col_mapping[clean_col] = col
        
        missing_cols = []
        for req_col in required_cols:
            if req_col not in col_mapping:
                missing_cols.append(req_col)
        
        if missing_cols:
            print(f"Colunas ausentes para plot: {missing_cols}")
            print(f"Colunas disponíveis: {list(col_mapping.keys())}")
            return
        
        setup_plot_style()
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        jl_col = col_mapping['jL']
        jg_col = col_mapping['jG']
        D = 0.05251
        P_col = col_mapping['Gauge Pressure']
        P_Pa = df[P_col] + 101325
        T_col = col_mapping['Temp.']
        T_K = df[T_col] + 273
        # Propriedades do gás (Re_sg) de acordo com fluid_1 (Air, SF6, etc.)
        rho_G = [PropsSI('D', 'P', p, 'T', t, fluid_1) for p, t in zip(P_Pa, T_K)]
        mu_G = [PropsSI('V', 'P', p, 'T', t, fluid_1) for p, t in zip(P_Pa, T_K)]
        
        Re_sg = pd.Series(rho_G * df[jg_col] * D / mu_G, index=df.index)
        # Guardar Re_sg no DataFrame original para uso/visualização posterior
        df['Re_sg'] = Re_sg

        dp_dz_f_col = col_mapping['dp/dz_F']
        flow_pattern_col = col_mapping['Flow Pattern']
        
        # Primeiro agrupar dados por jL (arredondando para 1 casa decimal para agrupar séries similares)
        df_plot = df.copy()
        df_plot['jL_rounded'] = df_plot[jl_col].round(1)
        
        # Obter séries únicas de jL
        jl_series = sorted(df_plot['jL_rounded'].unique())
        jl_series = [jl for jl in jl_series if pd.notna(jl)]

        # Re_sl_group deve ser padronizado previamente (main) para ficar constante por grupo e por inclinação
        if 'Re_sl_group' not in df.columns:
            standardize_liquid_conditions({sheet_name: df})

        df_plot['Re_sl_group'] = df['Re_sl_group']

        # Obter séries únicas de Re_sl (agrupado/padronizado)
        Re_sl_series = sorted(df_plot['Re_sl_group'].dropna().unique())
        Re_sl_series = [re_l for re_l in Re_sl_series if pd.notna(re_l)]
        
        print(f"Séries de jL encontradas: {jl_series}")
        print(f"Séries de Re_sl (padronizado) encontradas: {Re_sl_series}")
        
        colors = ['red', 'blue', 'orange', 'yellow', 'silver', 'white']
        line_styles = ['-', '--', ':', '-.', '-', '--', ':', '-.', '-', '--']
        symbol_line = ['o','h', 'p', 'D', 's', '^', 'v' ]

        flow_pattern_symbols = get_flow_pattern_symbols()
        vec_s_lines = []
        
        # Inicializar variáveis para uso fora do loop
        frictional_sorted = []
        Re_sg_sorted = []

        # Plotar cada série de Re_sl (agrupado em blocos de 4 pontos)
        for i, re_l in enumerate(Re_sl_series):
            # Filtrar dados para esta série de Re_sl
            mask = df_plot['Re_sl_group'] == re_l
            
            jg_data = df_plot.loc[mask, jg_col]
            frictional_data = df_plot.loc[mask, dp_dz_f_col]
            flow_pattern_data = df_plot.loc[mask, flow_pattern_col]
            
            # Remover valores nulos
            valid_mask = pd.notna(Re_sg) & pd.notna(frictional_data) & pd.notna(flow_pattern_data)
            Re_sg_clean = Re_sg[valid_mask]
            frictional_clean = frictional_data[valid_mask]/1000
            flow_pattern_clean = flow_pattern_data[valid_mask]
            
            if len(Re_sg_clean) > 0:
                color = colors[i % len(colors)]
                line_style = line_styles[i % len(line_styles)]
                s_lines = symbol_line[i % len(symbol_line)]
                vec_s_lines.append(s_lines)
                
                sorted_data = sorted(zip(Re_sg_clean, frictional_clean, flow_pattern_clean))
                Re_sg_sorted = [x[0] for x in sorted_data]
                frictional_sorted = [x[1] for x in sorted_data]
                flow_pattern_sorted = [x[2] for x in sorted_data]
                
                ax.plot(Re_sg_sorted, frictional_sorted, line_style, marker=s_lines, markersize=20, 
                       color='black', mfc='silver', linewidth=1.5, zorder=1)

                for j, (Re_sg_val, frictional_val, flow_pattern) in enumerate(zip(Re_sg_sorted, frictional_sorted, flow_pattern_sorted)):
                    pattern_data = flow_pattern_symbols.get(flow_pattern, {'symbol': 'o', 'color': 'gray'})
                    symbol = pattern_data['symbol']
                    color = pattern_data['color']
                    ax.scatter(Re_sg_val, frictional_val, c=color, marker=symbol, s=100, 
                             edgecolors='black', linewidth=1, zorder=2)
                
                print(f"Plotando série Re_sl = {re_l:.1f} com {len(Re_sg_clean)} pontos")
        
        used_patterns = set()
        for pattern in df_plot[flow_pattern_col].dropna().unique():
            if pattern in flow_pattern_symbols:
                used_patterns.add(pattern)
        
        legend_elements = []
        for pattern in sorted(used_patterns):
            pattern_data = flow_pattern_symbols.get(pattern, {'symbol': 'x', 'color': 'gray'})
            symbol = pattern_data['symbol']
            color = pattern_data['color']
            print(symbol, color)
            legend_elements.append(plt.Line2D([0], [0], marker=symbol, color='w', 
                                            markerfacecolor=color, markersize=8, 
                                            markeredgecolor='black', markeredgewidth=1,
                                            label=pattern))
        
        x_label = r'$Re_{sg}$'
        y_label = r'$\frac{\partial P}{\partial z} \text{f} \; \left[\frac{\text{kPa}}{\text{m}}\right]$'

        ax.set_xlabel(x_label, fontsize=24, fontfamily='serif')
        ax.set_ylabel(y_label, fontsize=24, fontfamily='serif')
        
        ax.set_axisbelow(True)
        ax.minorticks_on()
        
        ax.xaxis.set_major_locator(MultipleLocator(1.0))
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        
        ax.set_xlim(left=1000, right=250000)
        ax.set_ylim(bottom=0)
        ax.set_xscale('log')

        ax.tick_params(axis='both', which='major', labelsize=18, size=8)
        ax.tick_params(axis='both', which='minor', labelsize=18, size=6)

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily('serif')
        
        # Adicionar legenda para Re_sl com Re_sg nominal (médio) por série
        Re_sl_legend_elements = []
        for i, re_l in enumerate(Re_sl_series):
            line_style = line_styles[i % len(line_styles)]
            # Re_sg nominal (médio) para esta série de Re_sl
            mask_re = df_plot['Re_sl_group'] == re_l
            Re_sg_re = pd.to_numeric(Re_sg[mask_re], errors='coerce').dropna()
            if len(Re_sg_re) > 0:
                mean_Re_sg = float(Re_sg_re.mean())
                mean_Re_sg_rounded = int(round(mean_Re_sg / 100.0) * 100)
                series_label = rf'$Re_{{sl}}$ = {int(re_l)}, $Re_{{sg}} \approx$ {mean_Re_sg_rounded}'
            else:
                series_label = rf'$Re_{{sl}}$ = {int(re_l)}'

            Re_sl_legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    color='black',
                    linestyle=line_style,
                    marker=vec_s_lines[i],
                    mfc='silver',
                    markersize=12,
                    label=series_label,
                )
            )
        
        # Criar legenda única estilo painel em cima
        ax.legend(
            handles=Re_sl_legend_elements + legend_elements, 
            loc='lower center',
            bbox_to_anchor=(0.5, 1.02),
            ncol=4,
            frameon=False,
            fontsize=16,
            prop={'family': 'serif'},
            title=rf'$\theta = {theta}^\circ$'  # Adicionar título com a inclinação
        )
        
        # Adicionar texto de inclinação apenas se houver dados
        if frictional_sorted:
            plt.text(2000, 7*max(frictional_sorted)/8, rf'$\theta = {theta}^\circ$', fontsize=20)
        plt.tight_layout()
        
        output_dir = os.path.dirname(file_path)
        sheet_dir = os.path.join(output_dir, sheet_name)
        os.makedirs(sheet_dir, exist_ok=True)
        base_name = f"{sheet_name}_frictional_vs_Re_g"
        pdf_file = os.path.join(sheet_dir, f"{base_name}.pdf")
        png_file = os.path.join(sheet_dir, f"{base_name}.png")
        
        plt.savefig(pdf_file, format='pdf', dpi=300, bbox_inches='tight', 
                   pad_inches=0.1, facecolor='white', edgecolor='none')
        print(f"Plot PDF salvo: {pdf_file}")
        
        plt.savefig(png_file, format='png', dpi=300, bbox_inches='tight', 
                   pad_inches=0.1, facecolor='white', edgecolor='none')
        print(f"Plot PNG salvo: {png_file}")
        
    except Exception as e:
        print(f"Erro ao gerar plot: {e}")
        import traceback
        traceback.print_exc()

def generate_dpdzt_vs_Reg_plot(df, sheet_name, fluid_1, fluid_2, theta):
    """
    Gera um plot científico de Re_sg vs dp/dz_T, onde cada Re_sl é uma série diferente.
    Inclui símbolos diferentes para cada Flow Pattern.
    """
    try:
        required_cols = ['jG', 'jL', 'dp/dz_T', 'Flow Pattern', 'Temp.', 'Gauge Pressure']
        available_cols = list(df.columns)
        
        col_mapping = {}
        for col in available_cols:
            if pd.notna(col):
                clean_col = str(col).strip()
                col_mapping[clean_col] = col
        
        missing_cols = []
        for req_col in required_cols:
            if req_col not in col_mapping:
                missing_cols.append(req_col)
        
        if missing_cols:
            print(f"Colunas ausentes para plot: {missing_cols}")
            print(f"Colunas disponíveis: {list(col_mapping.keys())}")
            return
        
        setup_plot_style()
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        jl_col = col_mapping['jL']
        jg_col = col_mapping['jG']
        D = 0.05251
        P_col = col_mapping['Gauge Pressure']
        P_Pa = df[P_col] + 101325
        T_col = col_mapping['Temp.']
        T_K = df[T_col] + 273
        rho_G = [PropsSI('D', 'P', p, 'T', t, fluid_1) for p, t in zip(P_Pa, T_K)]
        mu_G = [PropsSI('V', 'P', p, 'T', t, fluid_1) for p, t in zip(P_Pa, T_K)]
        
        Re_sg = pd.Series(rho_G * df[jg_col] * D / mu_G, index=df.index)
        # Guardar Re_sg no DataFrame original para uso/visualização posterior
        df['Re_sg'] = Re_sg

        dp_dz_t_col = col_mapping['dp/dz_T']
        flow_pattern_col = col_mapping['Flow Pattern']
        
        # Primeiro agrupar dados por jL (arredondando para 1 casa decimal para agrupar séries similares)
        df_plot = df.copy()
        df_plot['jL_rounded'] = df_plot[jl_col].round(1)
        
        # Obter séries únicas de jL
        jl_series = sorted(df_plot['jL_rounded'].unique())
        jl_series = [jl for jl in jl_series if pd.notna(jl)]

        # Re_sl_group deve ser padronizado previamente (main) para ficar constante por grupo e por inclinação
        if 'Re_sl_group' not in df.columns:
            standardize_liquid_conditions({sheet_name: df})

        df_plot['Re_sl_group'] = df['Re_sl_group']

        # Obter séries únicas de Re_sl (agrupado/padronizado)
        Re_sl_series = sorted(df_plot['Re_sl_group'].dropna().unique())
        Re_sl_series = [re_l for re_l in Re_sl_series if pd.notna(re_l)]
        
        print(f"Séries de jL encontradas: {jl_series}")
        print(f"Séries de Re_sl (padronizado) encontradas: {Re_sl_series}")
        
        colors = ['red', 'blue', 'orange', 'yellow', 'silver', 'white']
        line_styles = ['-', '--', ':', '-.', '-', '--', ':', '-.', '-', '--']
        symbol_line = ['o','h', 'p', 'D', 's', '^', 'v' ]

        flow_pattern_symbols = get_flow_pattern_symbols()
        vec_s_lines = []
        
        # Inicializar variáveis para uso fora do loop
        total_sorted = []
        Re_sg_sorted = []

        # Plotar cada série de Re_sl (agrupado em blocos de 4 pontos)
        for i, re_l in enumerate(Re_sl_series):
            # Filtrar dados para esta série de Re_sl
            mask = df_plot['Re_sl_group'] == re_l
            
            jg_data = df_plot.loc[mask, jg_col]
            total_data = df_plot.loc[mask, dp_dz_t_col]
            flow_pattern_data = df_plot.loc[mask, flow_pattern_col]
            
            # Remover valores nulos
            valid_mask = pd.notna(Re_sg) & pd.notna(total_data) & pd.notna(flow_pattern_data)
            Re_sg_clean = Re_sg[valid_mask]
            total_clean = total_data[valid_mask]/1000
            flow_pattern_clean = flow_pattern_data[valid_mask]
            
            if len(Re_sg_clean) > 0:
                color = colors[i % len(colors)]
                line_style = line_styles[i % len(line_styles)]
                s_lines = symbol_line[i % len(symbol_line)]
                vec_s_lines.append(s_lines)
                
                sorted_data = sorted(zip(Re_sg_clean, total_clean, flow_pattern_clean))
                Re_sg_sorted = [x[0] for x in sorted_data]
                total_sorted = [x[1] for x in sorted_data]
                flow_pattern_sorted = [x[2] for x in sorted_data]
                
                ax.plot(Re_sg_sorted, total_sorted, line_style, marker=s_lines, markersize=20, 
                       color='black', mfc='silver', linewidth=1.5, zorder=1)

                for j, (Re_sg_val, total_val, flow_pattern) in enumerate(zip(Re_sg_sorted, total_sorted, flow_pattern_sorted)):
                    pattern_data = flow_pattern_symbols.get(flow_pattern, {'symbol': 'o', 'color': 'gray'})
                    symbol = pattern_data['symbol']
                    color = pattern_data['color']
                    ax.scatter(Re_sg_val, total_val, c=color, marker=symbol, s=100, 
                             edgecolors='black', linewidth=1, zorder=2)
                
                print(f"Plotando série Re_sl = {re_l:.1f} com {len(Re_sg_clean)} pontos")
        
        used_patterns = set()
        for pattern in df_plot[flow_pattern_col].dropna().unique():
            if pattern in flow_pattern_symbols:
                used_patterns.add(pattern)
        
        legend_elements = []
        for pattern in sorted(used_patterns):
            pattern_data = flow_pattern_symbols.get(pattern, {'symbol': 'x', 'color': 'gray'})
            symbol = pattern_data['symbol']
            color = pattern_data['color']
            print(symbol, color)
            legend_elements.append(plt.Line2D([0], [0], marker=symbol, color='w', 
                                            markerfacecolor=color, markersize=8, 
                                            markeredgecolor='black', markeredgewidth=1,
                                            label=pattern))
        
        x_label = r'$Re_{sg}$'
        y_label = r'$\frac{\partial P}{\partial z} \text{t} \; \left[\frac{\text{kPa}}{\text{m}}\right]$'

        ax.set_xlabel(x_label, fontsize=24, fontfamily='serif')
        ax.set_ylabel(y_label, fontsize=24, fontfamily='serif')
        
        ax.set_axisbelow(True)
        ax.minorticks_on()
        
        ax.xaxis.set_major_locator(MultipleLocator(1.0))
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        
        ax.set_xlim(left=1000, right=250000)
        # ax.set_ylim(bottom=0)
        ax.set_xscale('log')

        ax.tick_params(axis='both', which='major', labelsize=18, size=8)
        ax.tick_params(axis='both', which='minor', labelsize=18, size=6)

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily('serif')
        
        # Adicionar legenda para Re_sl com Re_sg nominal (médio) por série
        Re_sl_legend_elements = []
        for i, re_l in enumerate(Re_sl_series):
            line_style = line_styles[i % len(line_styles)]
            # Re_sg nominal (médio) para esta série de Re_sl
            mask_re = df_plot['Re_sl_group'] == re_l
            Re_sg_re = pd.to_numeric(Re_sg[mask_re], errors='coerce').dropna()
            if len(Re_sg_re) > 0:
                mean_Re_sg = float(Re_sg_re.mean())
                mean_Re_sg_rounded = int(round(mean_Re_sg / 100.0) * 100)
                series_label = rf'$Re_{{sl}}$ = {int(re_l)}, $Re_{{sg}} \approx$ {mean_Re_sg_rounded}'
            else:
                series_label = rf'$Re_{{sl}}$ = {int(re_l)}'

            Re_sl_legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    color='black',
                    linestyle=line_styles[i % len(line_styles)],
                    marker=vec_s_lines[i],
                    mfc='silver',
                    markersize=12,
                    label=series_label,
                )
            )

        ax.legend(
            handles=Re_sl_legend_elements + legend_elements,
            loc='lower center',
            bbox_to_anchor=(0.5, 1.02),
            ncol=4,
            frameon=False,
            fontsize=16,
            prop={'family': 'serif'},
            title=rf'$\theta = {theta}^\circ$'  # Adicionar título com a inclinação
        )
        
        # plt.text(2000, 7*max(total_sorted)/8, rf'$\theta = {theta}^\circ$', fontsize=20)
        plt.tight_layout()
        
        output_dir = os.path.dirname(file_path)
        sheet_dir = os.path.join(output_dir, sheet_name)
        os.makedirs(sheet_dir, exist_ok=True)
        base_name = f"{sheet_name}_total_vs_Re_g"
        pdf_file = os.path.join(sheet_dir, f"{base_name}.pdf")
        png_file = os.path.join(sheet_dir, f"{base_name}.png")
        
        plt.savefig(pdf_file, format='pdf', dpi=300, bbox_inches='tight', 
                   pad_inches=0.1, facecolor='white', edgecolor='none')
        print(f"Plot PDF salvo: {pdf_file}")
        
        plt.savefig(png_file, format='png', dpi=300, bbox_inches='tight', 
                   pad_inches=0.1, facecolor='white', edgecolor='none')
        print(f"Plot PNG salvo: {png_file}")
        
    except Exception as e:
        print(f"Erro ao gerar plot: {e}")
        import traceback
        traceback.print_exc()

def create_orientation_summary_dataframe(all_dataframes, selected_sheets):
    """
    Cria um DataFrame *detalhado* com dados de fricção vs orientação para cada ponto,
    mantendo todos os pontos individuais. Nenhuma média adicional é feita sobre dp/dz_F.
    
    Args:
        all_dataframes (dict): Dicionário com todos os DataFrames das abas
        selected_sheets (list): Lista das abas selecionadas pelo usuário
        
    Returns:
        pd.DataFrame: DataFrame com colunas
            [sheet_name, theta, Re_sl, frictional, alpha, total, flow_pattern, point_id, Re_sg]
    """
    summary_data = []
    
    for sheet_name in selected_sheets:
        if sheet_name not in all_dataframes:
            print(f"Aba {sheet_name} não encontrada nos dados.")
            continue
            
        df = all_dataframes[sheet_name]
        
        # Verificar colunas necessárias
        required_cols = ['jL', 'dp/dz_F', 'Flow Pattern']
        available_cols = list(df.columns)
        
        col_mapping = {}
        for col in available_cols:
            if pd.notna(col):
                clean_col = str(col).strip()
                col_mapping[clean_col] = col
        
        missing_cols = []
        for req_col in required_cols:
            if req_col not in col_mapping:
                missing_cols.append(req_col)
        
        if missing_cols:
            print(f"Aba {sheet_name}: Colunas ausentes: {missing_cols}")
            continue
        
        # Extrair informações da aba
        fluid_1, fluid_2, direction, theta, ID, is_validation = extract_info_from_filename(sheet_name)
        if direction == 'Downward':
            theta = -theta
        
        # Garantir que temos Re_sl_group padronizado
        if 'Re_sl_group' in df.columns:
            df_temp = df.copy()
        else:
            standardize_liquid_conditions({sheet_name: df})
            df_temp = df.copy()

        # Usar diretamente o Re_sl padronizado para o resumo/orientação
        df_temp['Re_sl_rounded'] = pd.to_numeric(df_temp['Re_sl_group'], errors='coerce').round(0)
        
        dp_dz_f_col = col_mapping['dp/dz_F']
        flow_pattern_col = col_mapping['Flow Pattern']
        alpha_col = col_mapping.get('α', None)
        dp_dz_t_col = col_mapping.get('dp/dz_T', None)

        # Garantir Re_sg disponível (para cálculo de Re_sg médio por série e legenda)
        if 'Re_sg' in df.columns:
            Re_sg_series = pd.to_numeric(df['Re_sg'], errors='coerce')
        else:
            # Tentar calcular Re_sg usando jG, Temp e Pressure (nomes podem variar, ex.: NAS)
            Re_sg_series = pd.Series([np.nan] * len(df_temp), index=df_temp.index)
            try:
                # Buscar colunas por qualquer nome usado no NAS ou formato padrão
                def _find_col(*candidates):
                    for k in candidates:
                        if k in col_mapping:
                            return col_mapping[k]
                    norm = {str(k).strip().upper(): col_mapping[k] for k in col_mapping}
                    for c in candidates:
                        if c.upper() in norm:
                            return norm[c.upper()]
                    return None
                jg_col = _find_col('jG', 'JG')
                T_col = _find_col('Temp.', 'Temp', 'T', 'T (C)', 'T(°C)')
                P_col = _find_col('Gauge Pressure', 'Gauge P', 'Gauge P.', 'P', 'Pressure', 'Gauge P (kPa)', 'P (kPa)')
                if jg_col is not None and T_col is not None and P_col is not None:
                    D = 0.05251
                    P_vals = pd.to_numeric(df[P_col], errors='coerce')
                    T_vals = pd.to_numeric(df[T_col], errors='coerce')
                    P_Pa = P_vals + 101325
                    T_K = T_vals + 273.15
                    rho_G = [PropsSI('D', 'P', p, 'T', t, fluid_1) for p, t in zip(P_Pa, T_K)]
                    mu_G = [PropsSI('V', 'P', p, 'T', t, fluid_1) for p, t in zip(P_Pa, T_K)]
                    Re_sg_series = pd.Series(
                        np.array(rho_G) * pd.to_numeric(df[jg_col], errors='coerce') * D / np.array(mu_G),
                        index=df.index
                    )
                else:
                    print(f"Aviso: não foi possível calcular Re_sg para {sheet_name} (colunas ausentes: jG/Temp/Pressure).")
            except Exception as e:
                print(f"Erro ao calcular Re_sg para {sheet_name}: {e}")

        # Converter friccional e total para kPa/m e manter todos os pontos individuais
        # Garantir que os dados sejam numéricos (NAS_file pode trazer strings ou colunas duplicadas)
        fric_col = df_temp[dp_dz_f_col]
        # Se houver colunas duplicadas com o mesmo nome, df_temp[dp_dz_f_col] é um DataFrame.
        if isinstance(fric_col, pd.DataFrame):
            fric_col = fric_col.iloc[:, 0]
        frictional_numeric = pd.to_numeric(fric_col, errors='coerce')
        frictional_kpa = frictional_numeric / 1000.0

        if dp_dz_t_col in df_temp.columns:
            total_col = df_temp[dp_dz_t_col]
            if isinstance(total_col, pd.DataFrame):
                total_col = total_col.iloc[:, 0]
            total_numeric = pd.to_numeric(total_col, errors='coerce')
            total_kpa = total_numeric / 1000.0
        else:
            total_kpa = pd.Series([np.nan] * len(df_temp), index=df_temp.index)
        flow_pattern_series = df_temp[flow_pattern_col]
        if isinstance(flow_pattern_series, pd.DataFrame):
            flow_pattern_series = flow_pattern_series.iloc[:, 0]
        re_sl_series = df_temp['Re_sl_rounded']

        # Preparar série de alpha (void fraction) garantindo tipo escalar/numerico
        if alpha_col is not None and alpha_col in df_temp.columns:
            alpha_data = df_temp[alpha_col]
            if isinstance(alpha_data, pd.DataFrame):
                alpha_data = alpha_data.iloc[:, 0]
            alpha_series = pd.to_numeric(alpha_data, errors='coerce')
        else:
            alpha_series = pd.Series([np.nan] * len(df_temp), index=df_temp.index)

        # Identificador de ponto experimental (P01, P02, ...) baseado no índice da linha
        # Isso assume que a ordem das linhas é consistente entre as inclinações.
        for idx in range(len(df_temp)):
            re_sl_val = re_sl_series.iloc[idx]
            fric_val = frictional_kpa.iloc[idx]
            total_val = total_kpa.iloc[idx]
            fp_val = flow_pattern_series.iloc[idx]
            # Se no NAS o flow pattern ainda vier em curto, normalizar para o nome completo
            if pd.notna(fp_val) and isinstance(fp_val, str):
                key = fp_val.strip().upper()
                fp_val = NAS_FLOW_PATTERN_MAP.get(key, NAS_FLOW_PATTERN_MAP.get(fp_val.strip(), fp_val))
            alpha_val = alpha_series.iloc[idx]
            re_sg_val = Re_sg_series.iloc[idx] if not Re_sg_series.isna().all() else np.nan
            point_id = f"P{idx+1:02d}"
            
            if pd.isna(re_sl_val) or pd.isna(fric_val) or pd.isna(fp_val):
                continue
            
            # Sistema bifásico: primeiros 2 caracteres da aba (AW, AO, SO, AD, etc.)
            system = str(sheet_name)[:2] if sheet_name else ""
            summary_data.append({
                'sheet_name': sheet_name,
                'system': system,
                'theta': theta,
                'Re_sl': int(re_sl_val),
                'frictional': fric_val,
                'alpha': alpha_val,
                'total': total_val,
                'flow_pattern': fp_val,
                'point_id': point_id,
                'Re_sg': re_sg_val
            })
    
    return pd.DataFrame(summary_data)

def group_similar_re_sl(summary_df, tolerance_percent=10):
    """
    Agrupa Re_sl próximos baseado em uma tolerância percentual e atribui a média do grupo.
    
    Args:
        summary_df (pd.DataFrame): DataFrame com dados de fricção vs orientação
        tolerance_percent (float): Tolerância percentual para agrupar Re_sl próximos (padrão: 10%)
        
    Returns:
        pd.DataFrame: DataFrame com Re_sl agrupados
    """
    try:
        # Obter valores únicos de Re_sl
        unique_re_sl = sorted(summary_df['Re_sl'].unique())
        print(f"Re_sl originais: {unique_re_sl}")
        
        # Criar grupos de Re_sl próximos
        groups = []
        current_group = [unique_re_sl[0]]
        
        for i in range(1, len(unique_re_sl)):
            current_re_sl = unique_re_sl[i]
            group_mean = np.mean(current_group)
            
            # Calcular tolerância baseada na média do grupo atual
            tolerance = group_mean * tolerance_percent / 100
            
            # Se o Re_sl atual está dentro da tolerância, adicionar ao grupo
            if abs(current_re_sl - group_mean) <= tolerance:
                current_group.append(current_re_sl)
            else:
                # Finalizar grupo atual e iniciar novo
                groups.append(current_group)
                current_group = [current_re_sl]
        
        # Adicionar último grupo
        groups.append(current_group)
        
        print(f"Grupos criados: {groups}")
        
        # Criar mapeamento de Re_sl original para Re_sl agrupado
        re_sl_mapping = {}
        for group in groups:
            group_mean = int(np.mean(group))
            for re_l in group:
                re_sl_mapping[re_l] = group_mean
        
        print(f"Mapeamento Re_sl: {re_sl_mapping}")
        
        # Aplicar mapeamento ao DataFrame - apenas substituir Re_sl pelos valores médios dos grupos
        summary_df_grouped = summary_df.copy()
        summary_df_grouped['Re_sl'] = summary_df_grouped['Re_sl'].map(re_sl_mapping)
        
        # Manter todos os pontos originais, apenas com Re_sl substituído pela média do grupo
        result_df = summary_df_grouped
        print(f"DataFrame com Re_sl agrupado: {len(result_df)} linhas (mantendo todos os pontos originais)")
        
        return result_df
        
    except Exception as e:
        print(f"Erro ao agrupar Re_sl: {e}")
        return summary_df


def prepare_orientation_summary(all_dataframes, selected_sheets, tolerance_percent=10):
    """
    Prepara o DataFrame resumo para os gráficos de orientação (friccional, alpha, total).
    Faz:
    - criação do summary_df detalhado,
    - agrupamento de Re_sl próximos,
    - salvamento em Excel para inspeção.
    """
    if not selected_sheets or not all_dataframes:
        print("Nenhuma aba selecionada ou dados não encontrados.")
        return None, None, []

    print("Criando DataFrame resumo com dados de orientação...")
    summary_df = create_orientation_summary_dataframe(all_dataframes, selected_sheets)
    if summary_df.empty:
        print("Nenhum dado válido encontrado para gerar o gráfico.")
        return None, None, []

    output_dir = os.path.dirname(file_path)
    orientation_plots_dir = os.path.join(output_dir, "orientation_plots")
    os.makedirs(orientation_plots_dir, exist_ok=True)
    print(f"Diretório criado: {orientation_plots_dir}")

    print("Agrupando Re_sl próximos...")
    summary_df = group_similar_re_sl(summary_df, tolerance_percent)

    try:
        orientation_summary_file = os.path.join(
            orientation_plots_dir, "orientation_summary_grouped_Re_sl.xlsx"
        )
        summary_df.to_excel(orientation_summary_file, sheet_name="summary", index=False)
        print(f"Resumo de orientação salvo em: {orientation_summary_file}")
    except Exception as e:
        print(f"Aviso: não foi possível salvar o resumo de orientação em Excel: {e}")

    unique_re_sl = sorted(summary_df['Re_sl'].unique())
    print(f"Condições de Re_sl agrupadas encontradas: {unique_re_sl}")

    return summary_df, orientation_plots_dir, unique_re_sl


# Cores por sistema bifásico (AW, AO, SO, AD) para gráficos de orientação com múltiplos sistemas
SYSTEM_COLORS = {
    'AW': 'C0',   # azul
    'AO': 'C2',   # verde
    'SO': 'C3',   # vermelho
    'AD': 'C4',   # castanho
}
SYSTEM_COLOR_LIST = ['C0', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']


def _generate_orientation_plot_for_quantity(
    summary_df,
    orientation_plots_dir,
    unique_re_sl,
    y_column,
    y_label,
    base_name_prefix,
):
    """
    Gera gráficos genéricos de (quantidade y_column) vs orientação θ para cada Re_sl,
    conectando pontos com o mesmo point_id (P01, P02, ...) entre inclinações.
    Se houver mais de um sistema bifásico (AW, AO, SO, AD), cada sistema usa uma cor
    e a legenda indica o sistema; com um único sistema as linhas permanecem pretas.
    """
    setup_plot_style()
    flow_pattern_symbols = get_flow_pattern_symbols()

    for re_l in unique_re_sl:
        print(f"\nGerando {base_name_prefix} para Re_sl = {re_l}...")

        data_re = summary_df[summary_df['Re_sl'] == re_l]
        if data_re.empty or y_column not in data_re.columns:
            continue

        data_re = data_re.sort_values('theta')
        point_ids = sorted(data_re['point_id'].unique())
        line_styles = ['-', '--', ':', '-.']

        # Verificar se há mais de um sistema bifásico (coluna 'system' = primeiros 2 chars da aba)
        systems = []
        if 'system' in data_re.columns:
            systems = sorted(data_re['system'].dropna().unique())
            systems = [s for s in systems if str(s).strip()]
        multi_system = len(systems) > 1
        if multi_system:
            system_to_color = {}
            for idx, sys in enumerate(systems):
                sys_str = str(sys).strip().upper()
                system_to_color[sys] = SYSTEM_COLORS.get(sys_str, SYSTEM_COLOR_LIST[idx % len(SYSTEM_COLOR_LIST)])
        else:
            line_color = 'black'

        fig, ax = plt.subplots(figsize=(12, 8))
        series_legend_elements = []

        if multi_system:
            # Uma linha por (point_id, system); cor = sistema, estilo = point_id
            for i, pid in enumerate(point_ids):
                for sys in systems:
                    subset = data_re[(data_re['point_id'] == pid) & (data_re['system'] == sys)]
                    subset = subset.sort_values('theta')
                    if subset.empty or subset[y_column].isna().all():
                        continue
                    color = system_to_color[sys]
                    ax.plot(
                        subset['theta'],
                        subset[y_column],
                        linestyle=line_styles[i % len(line_styles)],
                        color=color,
                        linewidth=1.2,
                        zorder=1,
                        alpha=0.7,
                    )
                    for _, row in subset.iterrows():
                        if pd.isna(row[y_column]):
                            continue
                        pattern_data = flow_pattern_symbols.get(
                            row['flow_pattern'], {'symbol': 'o', 'color': 'gray'}
                        )
                        symbol = pattern_data['symbol']
                        pattern_color = pattern_data['color']
                        ax.scatter(
                            row['theta'],
                            row[y_column],
                            c=pattern_color,
                            marker=symbol,
                            s=150,
                            edgecolors=color,
                            linewidth=1.5,
                            zorder=2,
                            alpha=0.9,
                        )
            # Legenda: primeiro uma entrada por sistema (cor)
            for sys in systems:
                series_legend_elements.append(
                    plt.Line2D(
                        [0], [0],
                        linestyle='-',
                        color=system_to_color[sys],
                        linewidth=2.5,
                        label=str(sys),
                    )
                )
            # Depois uma entrada por point_id (Re_sg, estilo de linha em cinza)
            for i, pid in enumerate(point_ids):
                serie = data_re[data_re['point_id'] == pid].sort_values('theta')
                if serie.empty:
                    continue
                if 'Re_sg' in serie.columns and not serie['Re_sg'].isna().all():
                    mean_Re_sg = float(serie['Re_sg'].mean())
                    mean_Re_sg_rounded = int(round(mean_Re_sg / 100.0) * 100)
                    series_label = rf"$Re_{{sg}} \approx$ {mean_Re_sg_rounded}"
                else:
                    series_label = f"{pid}"
                series_legend_elements.append(
                    plt.Line2D(
                        [0], [0],
                        linestyle=line_styles[i % len(line_styles)],
                        color='gray',
                        linewidth=1.5,
                        label=series_label,
                    )
                )
        else:
            # Um único sistema (ou nenhum): linhas em preto como antes
            for i, pid in enumerate(point_ids):
                serie = data_re[data_re['point_id'] == pid].sort_values('theta')
                if serie.empty or serie[y_column].isna().all():
                    continue

                ax.plot(
                    serie['theta'],
                    serie[y_column],
                    linestyle=line_styles[i % len(line_styles)],
                    color=line_color,
                    linewidth=1.2,
                    zorder=1,
                    alpha=0.7,
                )

                for _, row in serie.iterrows():
                    if pd.isna(row[y_column]):
                        continue
                    pattern_data = flow_pattern_symbols.get(
                        row['flow_pattern'], {'symbol': 'o', 'color': 'gray'}
                    )
                    symbol = pattern_data['symbol']
                    pattern_color = pattern_data['color']
                    ax.scatter(
                        row['theta'],
                        row[y_column],
                        c=pattern_color,
                        marker=symbol,
                        s=150,
                        edgecolors='black',
                        linewidth=1.5,
                        zorder=2,
                        alpha=0.9,
                    )

                if 'Re_sg' in serie.columns and not serie['Re_sg'].isna().all():
                    mean_Re_sg = float(serie['Re_sg'].mean())
                    mean_Re_sg_rounded = int(round(mean_Re_sg / 100.0) * 100)
                    series_label = rf"$Re_{{sg}} \approx$ {mean_Re_sg_rounded}"
                else:
                    series_label = f"{pid}"

                series_legend_elements.append(
                    plt.Line2D(
                        [0],
                        [0],
                        linestyle=line_styles[i % len(line_styles)],
                        color='black',
                        linewidth=1.5,
                        label=series_label,
                    )
                )

        if not series_legend_elements:
            plt.close(fig)
            continue

        ax.set_xlabel('Pipe inclination (θ) [°]', fontsize=24, fontfamily='serif')
        ax.set_ylabel(y_label, fontsize=24, fontfamily='serif')
        ax.set_axisbelow(True)
        ax.minorticks_on()
        ax.grid(True, which='major', alpha=0.3, linestyle='-', linewidth=0.8)
        ax.grid(True, which='minor', alpha=0.2, linestyle=':', linewidth=0.5)
        # Tamanho um pouco menor para os valores dos eixos
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=10)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily('serif')

        # Limites de θ com margem de 5° para menos e para mais
        theta_min = float(data_re['theta'].min())
        theta_max = float(data_re['theta'].max())
        x_min = theta_min - 5.0
        x_max = theta_max + 5.0
        ax.set_xlim(left=x_min, right=x_max)

        # Ticks principais apenas em múltiplos de 10° dentro do intervalo
        start_tick = np.ceil(x_min / 10.0) * 10.0
        end_tick = np.floor(x_max / 10.0) * 10.0
        if start_tick <= end_tick:
            ax.set_xticks(np.arange(start_tick, end_tick + 1e-6, 10.0))
        else:
            # Caso degenerado (intervalo muito pequeno), não força ticks especiais
            ax.set_xticks([theta_min, theta_max])

        # Minor ticks continuam em 5° para auxiliar leitura
        ax.xaxis.set_minor_locator(MultipleLocator(5))

        used_patterns = data_re['flow_pattern'].unique()
        pattern_legend_elements = []
        for pattern in sorted(used_patterns):
            pattern_data = flow_pattern_symbols.get(pattern, {'symbol': 'o', 'color': 'gray'})
            symbol = pattern_data['symbol']
            color = pattern_data['color']
            pattern_legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker=symbol,
                    color='w',
                    markerfacecolor=color,
                    markersize=8,
                    markeredgecolor='black',
                    markeredgewidth=1,
                    label=pattern,
                )
            )

        combined_handles = series_legend_elements + pattern_legend_elements
        ax.legend(
            handles=combined_handles,
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
            fancybox=True,
            shadow=True,
            fontsize=12,
            prop={'family': 'serif'},
        )

        # # Inserir texto com o Reynolds de líquido nominal (Re_sl), arredondado para a centena
        # try:
        #     re_sl_nominal = float(re_l)
        #     re_sl_nominal_rounded = int(round(re_sl_nominal / 100.0) * 100)
        #     # Posição: canto superior esquerdo do painel, com pequena margem
        #     x_text = ax.get_xlim()[0] + 0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0])
        #     y_text = ax.get_ylim()[1] - 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        #     ax.text(
        #         x_text,
        #         y_text,
        #         rf"$Re_{{sl}} \approx {re_sl_nominal_rounded}$",
        #         fontsize=16,
        #         fontfamily='serif',
        #     )
        # except Exception:
        #     pass

        # Ajustar limites específicos para alpha (0 a 1)
        if y_column == 'alpha':
            ax.set_ylim(bottom=0.0, top=1.0)

        # Inserir texto com o Reynolds de líquido nominal (Re_sl), arredondado para a centena,
        # posicionado no canto inferior direito do painel.
        try:
            re_sl_nominal = float(re_l)
            re_sl_nominal_rounded = int(round(re_sl_nominal / 100.0) * 100)
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            x_text = x_max - 0.02 * (x_max - x_min)
            y_text = y_min + 0.05 * (y_max - y_min)
            ax.text(
                x_text,
                y_text,
                rf"$Re_{{sl}} \approx {re_sl_nominal_rounded}$",
                fontsize=16,
                fontfamily='serif',
                ha='right',
                va='bottom',
            )
        except Exception:
            pass

        plt.tight_layout()
        base_name = f"{base_name_prefix}_Re_sl_{re_l}"
        pdf_file = os.path.join(orientation_plots_dir, f"{base_name}.pdf")
        png_file = os.path.join(orientation_plots_dir, f"{base_name}.png")

        with open(os.devnull, 'w') as devnull:
            with redirect_stderr(devnull):
                plt.savefig(
                    pdf_file,
                    format='pdf',
                    dpi=300,
                    bbox_inches='tight',
                    pad_inches=0.1,
                    facecolor='white',
                    edgecolor='none',
                )
                plt.savefig(
                    png_file,
                    format='png',
                    dpi=300,
                    bbox_inches='tight',
                    pad_inches=0.1,
                    facecolor='white',
                    edgecolor='none',
                )
        print(f"Plot PDF salvo: {pdf_file}")
        print(f"Plot PNG salvo: {png_file}")

        plt.close(fig)

def generate_frictional_vs_orientation_plot(all_dataframes, selected_sheets, units_dict):
    """
    Gera plots de fricção do tubo vs orientação do tubo agrupados por Re_sl.
    Cada Re_sl gera um plot separado salvo no diretório orientation_plots.
    
    Args:
        all_dataframes (dict): Dicionário com todos os DataFrames das abas
        selected_sheets (list): Lista das abas selecionadas pelo usuário
        units_dict (dict): Dicionário com as unidades das colunas
    """
    try:
        summary_df, orientation_plots_dir, unique_re_sl = prepare_orientation_summary(
            all_dataframes, selected_sheets
        )
        if summary_df is None:
            return

        _generate_orientation_plot_for_quantity(
            summary_df=summary_df,
            orientation_plots_dir=orientation_plots_dir,
            unique_re_sl=unique_re_sl,
            y_column='frictional',
            y_label=r'$\left(\frac{\partial P}{\partial z}\right)_{\text{frictional}} \; \left[\frac{\text{kPa}}{\text{m}}\right]$',
            base_name_prefix='frictional_vs_orientation',
        )
    except Exception as e:
        print(f"Erro ao gerar plots de fricção vs orientação: {e}")
        import traceback
        traceback.print_exc()


def generate_alpha_vs_orientation_plot(all_dataframes, selected_sheets, units_dict):
    """
    Gera plots de alpha vs orientação do tubo agrupados por Re_sl.
    Cada Re_sl gera um plot separado salvo no diretório orientation_plots.
    """
    try:
        summary_df, orientation_plots_dir, unique_re_sl = prepare_orientation_summary(
            all_dataframes, selected_sheets
        )
        if summary_df is None:
            return

        _generate_orientation_plot_for_quantity(
            summary_df=summary_df,
            orientation_plots_dir=orientation_plots_dir,
            unique_re_sl=unique_re_sl,
            y_column='alpha',
            y_label=r'Void fraction $\alpha$',
            base_name_prefix='alpha_vs_orientation',
        )
    except Exception as e:
        print(f"Erro ao gerar plots de alpha vs orientação: {e}")
        import traceback
        traceback.print_exc()


def generate_total_vs_orientation_plot(all_dataframes, selected_sheets, units_dict):
    """
    Gera plots de gradiente total de pressão vs orientação do tubo agrupados por Re_sl.
    Cada Re_sl gera um plot separado salvo no diretório orientation_plots.
    """
    try:
        summary_df, orientation_plots_dir, unique_re_sl = prepare_orientation_summary(
            all_dataframes, selected_sheets
        )
        if summary_df is None:
            return

        _generate_orientation_plot_for_quantity(
            summary_df=summary_df,
            orientation_plots_dir=orientation_plots_dir,
            unique_re_sl=unique_re_sl,
            y_column='total',
            y_label=r'$\left(\frac{\partial P}{\partial z}\right)_{\text{total}} \; \left[\frac{\text{kPa}}{\text{m}}\right]$',
            base_name_prefix='total_vs_orientation',
        )
    except Exception as e:
        print(f"Erro ao gerar plots de gradiente total vs orientação: {e}")
        import traceback
        traceback.print_exc()

def get_user_sheet_selection(excel_file, sheet_names=None):
    """
    Solicita ao usuário que escolha quais abas processar.
    
    Args:
        excel_file: Objeto ExcelFile com as abas disponíveis
        
    Returns:
        list: Lista das abas selecionadas pelo usuário
    """
    # Se não for fornecida uma lista de abas, usar todas as abas do arquivo
    if sheet_names is None:
        sheet_names = excel_file.sheet_names

    print("\nAbas disponíveis:")
    for i, sheet in enumerate(sheet_names):
        print(f"{i+1}: {sheet}")
    
    while True:
        try:
            choice = input(f"\nEscolha as abas (ex: '2 3 4 5 6 8 10 11') ou 'all' para todas: ").strip()
            
            if choice.lower() == 'all':
                # Retornar todas as abas consideradas (já filtradas, se for NAS_file)
                return sheet_names
            
            # Processar entrada separada por espaços (ex: '2 3 4 5 6 8 10 11')
            selected_indices = []
            numbers = choice.split()
            
            for num_str in numbers:
                if num_str.isdigit():
                    index = int(num_str) - 1
                    if 0 <= index < len(sheet_names):
                        selected_indices.append(index)
                    else:
                        print(f"Número {num_str} inválido. Escolha entre 1 e {len(sheet_names)}")
                        break
                else:
                    print(f"'{num_str}' não é um número válido.")
                    break
            else:
                # Se chegou aqui, todos os números são válidos
                selected_sheets = [sheet_names[i] for i in selected_indices]
                print(f"Abas selecionadas: {selected_sheets}")
                return selected_sheets
                
        except (ValueError, IndexError):
            print("Entrada inválida. Digite números válidos separados por espaços ou 'all'.")

def main():
    """
    Função principal que executa o programa.
    """
    print("=== LEITOR DE ARQUIVOS EXCEL ===")
    
    # Ler o arquivo Excel
    result = read_excel_file(file_path)
    
    # Verificar se retornou múltiplas abas ou uma única aba
    if isinstance(result, tuple) and len(result) == 3:
        # Múltiplas abas retornadas
        df, units, is_all_selected = result
        sheet_name = None
        fluid_1 = None
        fluid_2 = None
        theta = None
    elif isinstance(result, tuple) and len(result) == 7:
        # Uma única aba retornada
        df, units, sheet_name, fluid_1, fluid_2, theta, is_all_selected = result
    else:
        # Erro na leitura
        df = None
        units = None
        sheet_name = None
        fluid_1 = None
        fluid_2 = None
        theta = None
        is_all_selected = False
    
    if df is not None:
        # Padronizar condição do líquido (jL e Reynolds superficiais) antes de qualquer plot/saída
        # Isso garante Re_sl constante por grupo e independente da inclinação.
        if isinstance(df, dict):
            standardize_liquid_conditions(df)
        else:
            standardize_liquid_conditions({sheet_name or "Sheet1": df})

        # Se retornou um dicionário (múltiplas abas)
        if isinstance(df, dict):
            print(f"Arquivo com {len(df)} abas carregado")
            for sheet_name, sheet_df in df.items():
                analyze_dataframe(sheet_df)
        else:
            # DataFrame único
            analyze_dataframe(df)
            
        # Salvar informações em um arquivo de log
        # log_file = f"excel_analysis_log_{Path(file_path).stem}.txt"
        # with open(log_file, 'w', encoding='utf-8') as f:
        #     f.write(f"Análise do arquivo: {file_path}\n")
        #     f.write(f"Data/hora: {pd.Timestamp.now()}\n\n")
            
        #     if isinstance(df, dict):
        #         for sheet_name, sheet_df in df.items():
        #             f.write(f"=== ABA: {sheet_name} ===\n")
        #             f.write(f"Dimensões: {sheet_df.shape}\n")
        #             f.write(f"Colunas: {list(sheet_df.columns)}\n")
        #             f.write(f"Unidades: {units[sheet_name]}\n\n")
        #     else:
        #         f.write(f"Dimensões: {df.shape}\n")
        #         f.write(f"Colunas: {list(df.columns)}\n")
        #         f.write(f"Unidades: {units}\n\n")
        
        # print(f"Arquivos salvos: {output_file}, {log_file}")
        
        # Gerar plots individuais apenas se não foi escolhido 'all'
        if not is_all_selected:
            print("\n" + "="*60)
            print("GERANDO PLOTS INDIVIDUAIS")
            print("="*60)
            
            # Gerar plot científico jG vs α
            if isinstance(df, dict):
                for sheet_name, sheet_df in df.items():
                    # Extrair informações do nome da aba para cada aba
                    fluid_1, fluid_2, direction, theta, ID, is_validation = extract_info_from_filename(sheet_name)
                    if direction == 'Downward':
                        theta = -theta
                    generate_alpha_vs_jg_plot(sheet_df, sheet_name, fluid_1, fluid_2, theta)
            else:
                generate_alpha_vs_jg_plot(df, sheet_name, fluid_1, fluid_2, theta)
            
            if isinstance(df, dict):
                for sheet_name, sheet_df in df.items():
                    # Extrair informações do nome da aba para cada aba
                    fluid_1, fluid_2, direction, theta, ID, is_validation = extract_info_from_filename(sheet_name)
                    if direction == 'Downward':
                        theta = -theta
                    generate_dpdzf_vs_jg_plot(sheet_df, sheet_name, fluid_1, fluid_2, theta)
            else:
                generate_dpdzf_vs_jg_plot(df, sheet_name, fluid_1, fluid_2, theta)
            
            if isinstance(df, dict):
                for sheet_name, sheet_df in df.items():
                    # Extrair informações do nome da aba para cada aba
                    fluid_1, fluid_2, direction, theta, ID, is_validation = extract_info_from_filename(sheet_name)
                    if direction == 'Downward':
                        theta = -theta
                    generate_dpdzf_vs_Reg_plot(sheet_df, sheet_name, fluid_1, fluid_2, theta)
            else:
                generate_dpdzf_vs_Reg_plot(df, sheet_name, fluid_1, fluid_2, theta)

            if isinstance(df, dict):
                for sheet_name, sheet_df in df.items():
                    # Extrair informações do nome da aba para cada aba
                    fluid_1, fluid_2, direction, theta, ID, is_validation = extract_info_from_filename(sheet_name)
                    if direction == 'Downward':
                        theta = -theta
                    generate_dpdzt_vs_Reg_plot(sheet_df, sheet_name, fluid_1, fluid_2, theta)
            else:
                generate_dpdzt_vs_Reg_plot(df, sheet_name, fluid_1, fluid_2, theta)
        else:
            print("\n" + "="*60)
            print("OPÇÃO 'ALL' SELECIONADA - PULANDO PLOTS INDIVIDUAIS")
            print("="*60)
            
        # Nova funcionalidade: Plots de fricção/alpha/gradiente total vs orientação
        if is_all_selected:
            # Se 'all' foi selecionado, gerar automaticamente os gráficos de orientação
            print("\n" + "="*60)
            print("GERANDO GRÁFICOS DE ORIENTAÇÃO (FRICCIONAL / ALPHA / TOTAL)")
            print("="*60)
            
            if isinstance(df, dict):
                selected = list(df.keys())
                print(f"\nProcessando todas as abas carregadas: {selected}")
                generate_frictional_vs_orientation_plot(df, selected, units)
                generate_alpha_vs_orientation_plot(df, selected, units)
                generate_total_vs_orientation_plot(df, selected, units)
            else:
                # Para uma única aba, usar diretamente
                print(f"\nProcessando aba única: {sheet_name}")
                generate_frictional_vs_orientation_plot({sheet_name: df}, [sheet_name], {sheet_name: units})
                generate_alpha_vs_orientation_plot({sheet_name: df}, [sheet_name], {sheet_name: units})
                generate_total_vs_orientation_plot({sheet_name: df}, [sheet_name], {sheet_name: units})
    else:
        print("Falha ao carregar o arquivo Excel.")

    if df is not None:
        # Salvar DataFrame(s) finais em arquivo(s) Excel para visualização
        try:
            base_dir = os.path.dirname(file_path)
            base_name = Path(file_path).stem

            if isinstance(df, dict):
                # Várias abas: um arquivo com uma planilha por aba
                output_excel = os.path.join(base_dir, f"{base_name}_processed_all_sheets.xlsx")
                with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
                    for sheet_name_key, sheet_df in df.items():
                        # Nome de aba no Excel não pode passar de 31 caracteres
                        safe_sheet_name = str(sheet_name_key)[:31] if sheet_name_key else "Sheet1"
                        sheet_df.to_excel(writer, sheet_name=safe_sheet_name, index=False)
                print(f"\nDataFrames finais salvos em: {output_excel}")

                print("\nPrimeiras linhas do primeiro DataFrame:")
                first_sheet = list(df.keys())[0]
                print(df[first_sheet].head())
            else:
                # Única aba/DataFrame
                output_excel = os.path.join(base_dir, f"{base_name}_processed.xlsx")
                sheet_name_to_use = str(sheet_name)[:31] if sheet_name else "Sheet1"
                with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
                    df.to_excel(writer, sheet_name=sheet_name_to_use, index=False)
                print(f"\nDataFrame final salvo em: {output_excel}")

                print("\nPrimeiras linhas do DataFrame:")
                print(df.head())
        except Exception as e:
            print(f"\nAviso: não foi possível salvar o(s) DataFrame(s) em Excel: {e}")
    
    print('\n##########################################################################')
    print('#AEEEE! Parabéns, rotina executada com sucesso!#')
    print('##########################################################################')
    
if __name__ == "__main__":
    main()
