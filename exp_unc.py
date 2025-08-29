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
file_path = 'C:/Users/User/Documents/LEMI/FSC2_pc/2. Air-Water Tests/2. Air-Water Tests/AWD45/AWD45P08/AWD45P08' #Insira o caminho do arquivo a ser analisado NOTE: USE SEMPRE A BARRA NORMAL '/', SE ESTIVER INVERTIDA, MODIFIQUE-A

L = 1.70         # m comprimento entre as tomadas de diferencial de pressão

# Valores de calibração do densitômetro IMPORTANTE
I_g = 248466                     # Insira a intensidade padrão para o gás (Calibração do densitômetro)
I_f = 147950                      # Insira a intensidade padrão para o líquido (Calibração do densitômetro)

sensor_Yokogawa = 'PDT-M-0101D-30Kpa_mA'
sensor_Endress = 'PDT-M-0101-40kPa_mA'

### Colunas de interesse -> Insira o nome das colunas a plotar e avaliar do arquivo .dat
# Lista de colunas para análise: [nome_coluna, apelido, unidade]
colunas_analise = [
    [sensor_Yokogawa, r'\Delta P_{30\,kPa} / L', r'[Pa/m]'],
    ['Alpha', r'\alpha', r''],
    ['J_Water', r'J_{water}', r'[m/s]'],
    ['J_Air', r'J_{air}', r'[m/s]'],
    ['FT-O-0301', r'Q_{air}', r'[m³/h]'],
    ['PIT-M-0101', r'Gauge\ Pressure', r'[Bar]'],
    ['TIT-M-0101', r'Temperature', r'[°C]'],
    ['rho_g', r'\rho_{air}', r'[kg/m³]'],
    ['rho_g_parede', r'\rho_{air} \, line', r'[kg/m³]']
]
####################################################################################################################################################
#       '                                   END INPUTS
####################################################################################################################################################
g = 9.81        # m/s² 
rho_s = 962     # kg/m³         Densidade do silicone nas tomadas do sensor Yokogawa

def get_constants():
    """Returns a dictionary of constants used in the script."""
    return {
        'g': 9.81,         # m/s² 
        'rho_s': 962       # kg/m³         Densidade do silicone nas tomadas do sensor Yokogawa
    }

CONSTANTS = get_constants()
g = CONSTANTS['g']
rho_s = CONSTANTS['rho_s']

def find_min_std_window(df: pd.DataFrame, column_name: str, min_window_size: float, max_window_size: float):
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
    
    window_sizes_to_test = [min_window_size] if min_window_size == max_window_size else np.arange(min_window_size, max_window_size + 1, 1)

    for window_size in window_sizes_to_test:
        for i in range(len(df)):
            start_time = df['X_Value'].iloc[i]
            end_time = start_time + window_size
            
            end_idx_candidate = df[df['X_Value'] <= end_time].index[-1]
            
            if end_idx_candidate - i < 2:
                continue
                
            window_data = df[column_name].iloc[i:end_idx_candidate+1]
            current_std = window_data.std()
            
            actual_window_size = df['X_Value'].iloc[end_idx_candidate] - start_time
            if abs(actual_window_size - window_size) > window_size * 0.01:
                continue
            
            if current_std < min_std:
                min_std = current_std
                best_start_idx = i
                best_end_idx = end_idx_candidate
                best_window_size = window_size
    
    if min_std == float('inf'):
        raise ValueError(f"Não foi possível encontrar uma janela válida entre {min_window_size} e {max_window_size} segundos")
    
    return best_start_idx, best_end_idx + 1, min_std, best_window_size

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
    return df, data_teste

def format_filename(alias: str, unit: str) -> str:
    """
    Formats the filename by removing LaTeX characters and adding the unit.
    Example: r'\Delta P_{40\,kPa}' [Pa] -> Delta_P_40kPa [Pa]
    """
    name = alias.replace('\\', '')
    name = name.replace('{', '')
    name = name.replace('}', '')
    name = name.replace('\\,', '')
    name = name.replace('\\frac', '')
    name = name.replace('$', '')
    
    invalid_chars = ['|', '/', '\\', ':', '*', '?', '"', '<', '>', ',', ';', '=', ' ']
    for char in invalid_chars:
        name = name.replace(char, '_')
    
    while '__' in name:
        name = name.replace('__', '_')
    
    name = name.strip('_')
    
    if unit:
        unit = unit.replace('[', '').replace(']', '')
        name = f"{name}_{unit}"
    
    return name

def uncertainties_calc(resumo_df,window_df):
    if '-30' in sensor_Yokogawa:         #### Mudar nome no Labview
        span_yokogawa = 29E3  
    elif '-10' in sensor_Yokogawa:
        span_yokogawa = 9E3     

    # Incerteza dos sensores Yokogawa
    udP = 0.00055*span_yokogawa
    dP_mean = np.mean(window_df[sensor_Yokogawa])
    dP = unc.ufloat(dP_mean,udP)
    print(sensor_Yokogawa)
    print(f'dP: {dP:.3f}')

    # Incerteza de Alpha estimada
    uAlpha = resumo_df['Alpha'].iloc[0] * 0.05 # Padrão de 5 % da média do alpha
    Alpha = unc.ufloat(resumo_df['Alpha'].iloc[0],uAlpha)
    print('Alpha: ', Alpha)

    T_mean = resumo_df['TIT-M-0101'].iloc[0]
    uT = 0.15 + 0.02*T_mean
    T = unc.ufloat(T_mean,uT)
    T_abs = T + 273.15
    # print('T: ', T)

    P_mean = (resumo_df['PIT-M-0101'].iloc[0] + 1) * 1E5            #Absolute pressure in Pa
    uP = 0.0025*P_mean
    P = unc.ufloat(P_mean,uP)
    # print('P: ', P)
    
    # Cálculo da incerteza da densidade do gás
    M_ar = 28.96e-3 #[kg/kmol]			    # Massa molecular do ar em [kg/kmol] 
    R = 8.314 #[kJ/kmolK]				    # Constante universal dos gases [kJ/kmol.K] 
    rho_G = (P * M_ar) / (R * T_abs)	# Densidade média do ar em [kg/m^3] 
    u_rho_g = rho_G.std_dev
    # print('u_rho_g: ', rho_G)

    # Cálcuo da incerteza da densidade do líquido
    rho_L = - 0.0042*(T**2) - 0.0529*T + 1000.9		# Média da densidade do líquido
    u_rho_l = rho_L.std_dev
    # print('u_rho_l: ', rho_L)

    rho_tubbing = rho_s
    # print('rho_tubbing: ', rho_tubbing)
    theta_rad = np.deg2rad(theta)
    # print('theta_rad: ', theta_rad)

    g = 9.81
    uL = 0.5E-3
    L_ = unc.ufloat(L,uL)
    # print('L: ', L_)


    dPg_dz = ((1-Alpha)*rho_L + Alpha*rho_G) * g * np.sin(theta_rad)
    # print(f'dPg_dz: {dPg_dz:.3f}')
    # Cálculo da incerteza do friccional
            
    if direction in ['Upward', 'Horizontal']:
        dPf_dz = (dP/L_) - ((1-Alpha)*rho_L + Alpha*rho_G - rho_tubbing) * g * np.sin(theta_rad)
        dPt_dz = dPf_dz + dPg_dz
    elif direction == 'Downward':
        print('Downward')
        dPf_dz = -(dP/L_) + ((1-Alpha)*rho_L + Alpha*rho_G - rho_tubbing) * g * np.sin(theta_rad)
        dPt_dz = -(abs(dPf_dz)) + dPg_dz
    
    # print(f'dPf_dz: {dPf_dz:.3f}')
    # print(f'dPt_dz: {dPt_dz:.3f}')
    
    # Alocando as incetezas
    udPf_dz = dPf_dz.std_dev
    udPg_dz = dPg_dz.std_dev
    udPt_dz = dPt_dz.std_dev

    # Adicionar a incerteza ao resumo_df logo após a coluna dP_F_dz do sensor Yokogawa
    col_name = f'dP_F/dz {sensor_Yokogawa}'
    unc_col_name = f'udP_F_dz_{sensor_Yokogawa}'
    if col_name in resumo_df.columns:
        items = list(resumo_df.iloc[0].items())
        idx = [i for i, (k, v) in enumerate(items) if k == col_name]
        if idx:
            insert_pos = idx[0] + 1
            items.insert(insert_pos, (unc_col_name, udPf_dz))
            resumo_df = pd.DataFrame([dict(items)])
        else:
            resumo_df[unc_col_name] = udPf_dz
    else:
        resumo_df[unc_col_name] = udPf_dz

    # Adicionar a incerteza ao resumo_df logo após a coluna Alpha
    alpha_col_name = 'Alpha'
    alpha_unc_col_name = 'uAlpha'
    if alpha_col_name in resumo_df.columns:
        items = list(resumo_df.iloc[0].items())
        idx = [i for i, (k, v) in enumerate(items) if k == alpha_col_name]
        if idx:
            insert_pos = idx[0] + 1
            items.insert(insert_pos, (alpha_unc_col_name, uAlpha))
            resumo_df = pd.DataFrame([dict(items)])
        else:
            resumo_df[alpha_unc_col_name] = uAlpha
    else:
        resumo_df[alpha_unc_col_name] = uAlpha

    # Adicionar a incerteza ao resumo_df logo após a coluna dP_dz_gravitacional
    grav_col_name = 'dP_dz_gravitacional'
    grav_unc_col_name = 'udP_dz_gravitacional'
    if grav_col_name in resumo_df.columns:
        items = list(resumo_df.iloc[0].items())
        idx = [i for i, (k, v) in enumerate(items) if k == grav_col_name]
        if idx:
            insert_pos = idx[0] + 1
            items.insert(insert_pos, (grav_unc_col_name, udPg_dz))
            resumo_df = pd.DataFrame([dict(items)])
        else:
            resumo_df[grav_unc_col_name] = udPg_dz
    else:
        resumo_df[grav_unc_col_name] = udPg_dz

    # Adicionar a incerteza ao resumo_df logo após a coluna dP_dz_total do sensor Yokogawa
    total_col_name = f'dP_dz_total_{sensor_Yokogawa}'
    total_unc_col_name = f'udP_dz_total_{sensor_Yokogawa}'
    if total_col_name in resumo_df.columns:
        items = list(resumo_df.iloc[0].items())
        idx = [i for i, (k, v) in enumerate(items) if k == total_col_name]
        if idx:
            insert_pos = idx[0] + 1
            items.insert(insert_pos, (total_unc_col_name, udPt_dz))
            resumo_df = pd.DataFrame([dict(items)])
        else:
            resumo_df[total_unc_col_name] = udPt_dz
    else:
        resumo_df[total_unc_col_name] = udPt_dz

    return resumo_df

def save_results(df: pd.DataFrame, coluna_escolhida: str, start_idx: int, end_idx: int, min_std: float, media_janela: float, 
                min_window_size: float, max_window_size: float, best_window_size: float, file_path: str, data_teste: str, 
                fluid_1: str, fluid_2: str, direction: str, theta: int, ID: str, nomes: list = None, medias: list = None, desvios: list = None, uAs: list = None,
                escolha_janela: str = None):
    """
    Salva os resultados da análise em um arquivo Excel.
    Agora inclui as estatísticas (média, desvio padrão, uA) de cada variável de interesse.
    """
    diretorio = os.path.dirname(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    output_file = os.path.join(diretorio, f"{base_name}_processed.xlsx")
    
    data_atual = datetime.now().strftime('%d/%m/%Y')
    
    header_info = {
        "Data do teste experimental": data_teste if data_teste else 'Não encontrada',
        "Data tratamento": data_atual,
        "Arquivo Original": file_path,
        "ID do ponto": ID,
        "Fluido 1": fluid_1,
        "Fluido 2": fluid_2,
        "Direção do escoamento": direction,
        "Inclinação (theta)": f"{theta}°",
        "Intensidade do gas densitometro (Ig)": I_g,
        "Intensidade do fluido densitometro (If)": I_f,
    }
    
    if escolha_janela == '1':
        header_info.update({
            "Tipo de Janela": "Manual",
            "Tamanho da Janela": f"{best_window_size:.1f} segundos",
            "Tempo Inicial": f"{df['X_Value'].iloc[start_idx]:.2f} segundos",
            "Tempo Final": f"{df['X_Value'].iloc[end_idx-1]:.2f} segundos",
            "Número de Pontos": end_idx - start_idx,
        })
    else:
        header_info.update({
            "Tipo de Janela": "Automática",
            "Coluna Critério": coluna_escolhida,
            "Tamanho Mínimo da Janela": f"{min_window_size:.1f} segundos",
            "Tamanho Máximo da Janela": f"{max_window_size:.1f} segundos",
            "Tamanho Ótimo da Janela": f"{best_window_size:.1f} segundos",
            "Média da Janela": f"{media_janela:.4f}",
            "Desvio Padrão": f"{min_std:.4f}",
            "Tempo Inicial": f"{df['X_Value'].iloc[start_idx]:.2f} segundos",
            "Tempo Final": f"{df['X_Value'].iloc[end_idx-1]:.2f} segundos",
            "Número de Pontos": end_idx - start_idx,
        })
    
    header_df = pd.DataFrame(list(header_info.items()), columns=['Parâmetro', 'Valor'])
    
    window_data = df.iloc[start_idx:end_idx]
    
    medias_janela = window_data.mean()
    
    colunas_corrigidas = []
    coluna_gravitacional = None
    
    for col in df.columns:
        if col.startswith('dP_F/dz dP_dz_total_'):
            colunas_corrigidas.append(col.replace('dP_F/dz dP_dz_total_', 'dP_dz_total_'))
        elif col.startswith('dP_F/dz dP_dz_gravitacional'):
            coluna_gravitacional = 'dP_dz_gravitacional'
        else:
            colunas_corrigidas.append(col)
    
    if coluna_gravitacional:
        colunas_corrigidas.append(coluna_gravitacional)
    
    medias_corrigidas = []
    dP_dz_total_values = {}  # Dicionário para armazenar os valores calculados
    for col in colunas_corrigidas:
        if col == 'dP_dz_gravitacional':
            medias_corrigidas.append(medias_janela['dP_F/dz dP_dz_gravitacional'])
        elif col.startswith('dP_dz_total_'):
            sensor_name = col.replace('dP_dz_total_', '')
            dP_F_col = f'dP_F/dz {sensor_name}'
            dP_F_series = window_data[dP_F_col]
            dP_F_RMS = np.sqrt(np.mean(np.square(dP_F_series)))
            grav_term = medias_janela['dP_F/dz dP_dz_gravitacional']
            if direction in ['Upward', 'Horizontal']:
                dP_dz_total = dP_F_RMS + grav_term
            else:
                dP_dz_total = -dP_F_RMS + grav_term
            medias_corrigidas.append(dP_dz_total)
            dP_dz_total_values[col] = dP_dz_total
        elif col.startswith('PDT-'):
            y_data = window_data[col]
            y_min = y_data.min()
            y_max = y_data.max()
            if (y_min/abs(y_min)) != (y_max/abs(y_max)):
                medias_corrigidas.append(np.sqrt(np.mean(np.square(y_data))))
            else:
                medias_corrigidas.append(y_data.mean())
        elif col.startswith('dP_F/dz PDT'):
            y_data = window_data[col]
            medias_corrigidas.append(np.sqrt(np.mean(np.square(y_data))))
        else:
            medias_corrigidas.append(medias_janela[col])
    
    resumo_dict = dict(zip(colunas_corrigidas, medias_corrigidas))

    # Calcular rho_liquido médio na janela
    try:
        if 'TIT-M-0101' in window_data.columns:
            temp_liquido_celsius = window_data['TIT-M-0101']
            temp_liquido_k = temp_liquido_celsius + 273.15
            if fluid_1 or fluid_2 == 'Water':
                rho_liquido_vals = [PropsSI('D', 'T', t, 'P', 101325, 'Water') for t in temp_liquido_k]
            elif fluid_1 or fluid_2 == 'Oil':
                rho_liquido_vals = -0.65178*temp_liquido_celsius +879.76961 

            rho_liquido_medio = np.mean(rho_liquido_vals)
        else:
            rho_liquido_medio = 1000
    except Exception as e:
        rho_liquido_medio = 1000

    # Inserir rho_liquido logo após rho_g
    if 'rho_g' in resumo_dict:
        items = list(resumo_dict.items())
        idx = [i for i, (k, v) in enumerate(items) if k == 'rho_g']
        if idx:
            insert_pos = idx[0] + 1
            items.insert(insert_pos, ('rho_liquido', rho_liquido_medio))
            resumo_dict = dict(items)
        else:
            resumo_dict['rho_liquido'] = rho_liquido_medio
    else:
        resumo_dict['rho_liquido'] = rho_liquido_medio

    resumo_df = pd.DataFrame([resumo_dict])
    
    window_df = pd.DataFrame()

    for col in colunas_corrigidas:
        if col == 'dP_dz_gravitacional':
            window_df[col] = window_data['dP_F/dz dP_dz_gravitacional']
        elif col.startswith('dP_dz_total_'):
            window_df[col] = dP_dz_total_values[col]
        else:
            window_df[col] = window_data[col]
    
    window_df.insert(0, 'Time (s)', window_data['X_Value'])
    
    # Atualizar resumo_df com as incertezas e salvar no Excel
    resumo_df = uncertainties_calc(resumo_df, window_df)

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        header_df.to_excel(writer, sheet_name='Info', index=False)
        resumo_df.to_excel(writer, sheet_name='Resumo Medias', index=False, header=True)
        window_df.to_excel(writer, sheet_name='Serie Temporal', index=False)
    

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
        if nome_coluna.startswith('PDT-'):
            y_data = df[nome_coluna]/L
            rms_pdt = np.sqrt(np.mean(np.square(y_data)))
            ax.plot(df['X_Value'], y_data, 'b-', alpha=0.8)
        else:
            y_data = df[nome_coluna]
            ax.plot(df['X_Value'], y_data, 'b-', alpha=0.8)
        media_serie = y_data.mean()
        desvio_serie = y_data.std()
        
        y_min = y_data.min()
        y_max = y_data.max()

        if 'PDT' in nome_coluna:
            if y_min >= 0:
                margem_min = y_min * 0.99
                margem_max = y_max * 1.01
            elif y_max <= 0:
                margem_min = y_min * 1.01
                margem_max = y_max * 0.99
            else:
                max_abs = max(abs(y_min), abs(y_max))
                margem_min = -max_abs * 1.01
                margem_max = max_abs * 1.01
        else:
            margem_min = y_min * 0.99
            margem_max = y_max * 1.01
            
        ax.set_ylim(margem_min, margem_max)
        
        ax.axhline(y=media_serie, color='g', linestyle='--', label=f'Mean: {media_serie:.4f}')
        ax.axhline(y=media_serie + desvio_serie, color='r', linestyle=':', label=f'std: ±{desvio_serie:.4f}')
        if nome_coluna.startswith('PDT-'):
            ax.axhline(y=rms_pdt, color='g', linestyle='-', label=f'RMS: {rms_pdt:.4f}')
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
        nome_coluna = coluna
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
        
        y_min = y_data_full.min()
        y_max = y_data_full.max()
        
        if 'PDT' in nome_coluna:
            if y_min >= 0:
                margem_min = y_min * 0.99
                margem_max = y_max * 1.01
            elif y_max <= 0:
                margem_min = y_min * 1.01
                margem_max = y_max * 0.99
            else:
                max_abs = max(abs(y_min), abs(y_max))
                margem_min = -max_abs * 1.01
                margem_max = max_abs * 1.01
        else:
            margem_min = y_min * 0.99
            margem_max = y_max * 1.01

        ax.set_ylim(margem_min, margem_max)
        
        ax.axhline(y=media_janela, color='g', linestyle='--', label=f'Mean: {media_janela:.4f}')
        ax.axhline(y=media_janela + desvio_janela, color='r', linestyle=':', label=f'std: ±{desvio_janela:.4f}')
        if nome_coluna.startswith('PDT-'):
            ax.axhline(y=rms_pdt_win, color='g', linestyle='-', label=f'RMS: {rms_pdt_win:.4f}')
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
    output_path = os.path.join(output_dir, f"windows-{base_name}.png")
    plt.savefig(output_path)
    plt.close(fig)

def uncert_propagation(df, colunas, start_idx, end_idx, best_window_size):
    """
    Propaga as incertezas das variáveis para a variável critério.
    Para cada coluna de interesse, calcula a média, o desvio padrão e a incerteza estatística tipo A (padrão da média) na janela selecionada.
    Retorna listas com os resultados para uso posterior.
    """
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
    return [c[0] for c in colunas], medias, desvios, uAs

def calc_alpha(df, start_idx, end_idx):
    """
    Determina a fração de vazio (alpha) na mistura a partir dos dados do densitômetro na janela selecionada.
    Retorna um DataFrame contendo a série temporal de Alpha na janela.
    """
    dados_densitometro = df['Densitometro'].iloc[start_idx:end_idx]
    alpha_series = np.log(dados_densitometro / I_f) / np.log(I_g / I_f)
    
    alpha_df = pd.DataFrame({
        'X_Value': df['X_Value'].iloc[start_idx:end_idx],
        'Alpha': alpha_series
    })
    
    return alpha_df

def calc_frictional_pressure_gradient(df, colunas, start_idx, end_idx, best_window_size, alpha_series):
    """
    Calcula o gradiente de pressão friccional para cada variável na janela selecionada.
    Também encontra os índices das colunas que começam com 'PDT'.
    Calcula a série temporal da densidade do ar na janela usando a equação de estado.
    A pressão é lida como manométrica em bares e convertida para absoluta em Pascal.
    Armazena os resultados em um DataFrame dP_F_df e o retorna.
    """
    indices_pdt = [i for i, col in enumerate(colunas) if col[0].startswith("PDT")]

    try:
        if fluid_1 or fluid_2 == 'Air':
            pressao_gas_bar = df['PIT-M-0101'].iloc[start_idx:end_idx]
            temp_gas_celsius = df['TIT-M-0101'].iloc[start_idx:end_idx]
            
            pressao_gas_parede_bar = df['PIT-A-0301'].iloc[start_idx:end_idx]
            temp_gas_parede_celsius = df['TIT-A-0301'].iloc[start_idx:end_idx]
        elif fluid_1 or fluid_2 == 'SF6':
            print('SF6 not implemented yet')
            exit()
       
        pressao_gas_pa = (pressao_gas_bar + 1) * 1e5
        temp_gas_k = temp_gas_celsius + 273.15
        
        pressao_gas_parede_pa = (pressao_gas_parede_bar + 1) * 1e5
        temp_gas_parede_k = temp_gas_parede_celsius + 273.15
        
        rho_gas = [PropsSI('D', 'P', p, 'T', t, 'Air') for p, t in zip(pressao_gas_pa, temp_gas_k)]
        rho_gas_parede = [PropsSI('D', 'P', p, 'T', t, 'Air') for p, t in zip(pressao_gas_parede_pa, temp_gas_parede_k)]
        rho_gas = pd.Series(rho_gas, index=pressao_gas_pa.index)
        rho_gas_parede = pd.Series(rho_gas_parede, index=pressao_gas_parede_pa.index)
    
        try:
            if fluid_1 or fluid_2 == 'Water':
                temp_liquido_celsius = df['TIT-M-0101'].iloc[start_idx:end_idx]
                temp_liquido_k = temp_liquido_celsius + 273.15
                rho_liquido = [PropsSI('D', 'T', t, 'P', 101325, 'Water') for t in temp_liquido_k]
                rho_liquido = pd.Series(rho_liquido, index=temp_liquido_celsius.index)
            elif fluid_1 or fluid_2 == 'Oil':
                temp_liquido_celsius = df['TIT-M-0101'].iloc[start_idx:end_idx]
                rho_liquido = -0.65178*temp_liquido_celsius +879.76961 
        except Exception as e:
            rho_liquido = 1000

    except KeyError as e:
        rho_gas = None

    dP_F_df = pd.DataFrame()

    for i in indices_pdt:
        coluna_pdt_nome = colunas[i][0]
        theta_rad = np.deg2rad(theta)

        if coluna_pdt_nome == sensor_Yokogawa:
            rho_tubbing = rho_s
        else:
            rho_tubbing = rho_liquido
        
        delta_p_prime = df[coluna_pdt_nome].iloc[start_idx:end_idx]
        
        termo_gravitacional = ((1-alpha_series)*rho_liquido + alpha_series*rho_gas - rho_tubbing) * g * np.sin(theta_rad)
        dP_dz_gravitacional = ((1-alpha_series)*rho_liquido + alpha_series*rho_gas) * g * np.sin(theta_rad)

        if direction in ['Upward', 'Horizontal']:
            dP_F_over_dz_series = (delta_p_prime / L) - termo_gravitacional
            dP_F_over_dz_RMS = np.sqrt(np.mean(np.square(dP_F_over_dz_series)))
            dP_dz_total = dP_F_over_dz_RMS + np.mean(dP_dz_gravitacional)
        elif direction == 'Downward':
            dP_F_over_dz_series = -(delta_p_prime / L) + termo_gravitacional 
            dP_F_over_dz_RMS = np.sqrt(np.mean(np.square(dP_F_over_dz_series)))
            dP_dz_total = -(dP_F_over_dz_RMS) + np.mean(dP_dz_gravitacional)

        dP_F_df[coluna_pdt_nome] = dP_F_over_dz_series
        dP_F_df[f'dP_dz_total_{coluna_pdt_nome}'] = dP_dz_total
        if i == indices_pdt[0]:
            dP_F_df['dP_dz_gravitacional'] = dP_dz_gravitacional

    cols = [col for col in dP_F_df.columns if col != 'dP_dz_gravitacional']
    cols.append('dP_dz_gravitacional')
    dP_F_df = dP_F_df[cols]

    return dP_F_df

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

def check_required_columns(df: pd.DataFrame, colunas_analise: list):
    """
    Verifica se as colunas necessárias existem no DataFrame.
    Retorna True se todas existirem, False caso contrário.
    """
    colunas_faltantes = []
    
    colunas_calculadas = ['Alpha', 'rho_g', 'rho_g_parede']
    
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

def plot_alpha(df: pd.DataFrame, start_idx: int, end_idx: int, output_dir: str, base_name: str):
    """
    Plota a série temporal de Alpha e salva a imagem.
    
    Args:
        df (pandas.DataFrame): DataFrame com os dados
        start_idx (int): Índice inicial da janela
        end_idx (int): Índice final da janela
        output_dir (str): Diretório para salvar a imagem
        base_name (str): Nome base para o arquivo de saída
    """
    alpha_df = calc_alpha(df, start_idx, end_idx)
    
    if alpha_df is not None and not alpha_df.empty:
        plt.figure(figsize=(15, 8))
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
        plt.close(plt.gcf())
        return alpha_df
    else:
        return None

def plot_pressure_gradients(df: pd.DataFrame, dP_F_df: pd.DataFrame, start_idx: int, end_idx: int, output_dir: str, base_name: str, sensor_Yokogawa: str, sensor_Endress: str, direction: str):
    """
    Plota as séries temporais de gradientes de pressão e salva a imagem.
    
    Args:
        df (pandas.DataFrame): DataFrame com os dados
        dP_F_df (pandas.DataFrame): DataFrame com os gradientes de pressão
        start_idx (int): Índice inicial da janela
        end_idx (int): Índice final da janela
        output_dir (str): Diretório para salvar a imagem
        base_name (str): Nome base para o arquivo de saída
        sensor_Yokogawa (str): Nome do sensor Yokogawa
        sensor_Endress (str): Nome do sensor Endress
        direction (str): Direção do escoamento
    """
    if dP_F_df is not None and not dP_F_df.empty:
        plt.figure(figsize=(15, 8))
        
        grav_series = dP_F_df['dP_dz_gravitacional']
        grav_mean = grav_series.mean()
        grav_std = grav_series.std()
        plt.plot(df['X_Value'].iloc[start_idx:end_idx], grav_series, 
                label=f'dP/dz gravitacional\nmean: {grav_mean:.2f} ± {grav_std:.2f}', 
                color='black', linestyle='--')
        
        for col in dP_F_df.columns:
            if not col.startswith('dP_dz_total') and col != 'dP_dz_gravitacional':
                if col == sensor_Yokogawa:
                    sensor_range = col.split('-')[3]
                    label_base = f'Yokogawa {sensor_range}'
                elif col == sensor_Endress:
                    sensor_range = col.split('-')[3]
                    label_base = f'Endress {sensor_range}'
                else:
                    label_base = col
                
                series = dP_F_df[col]
                rms = np.sqrt(np.mean(np.square(series)))
                plt.plot(df['X_Value'].iloc[start_idx:end_idx], series, 
                        label=f'dP_F/dz {label_base}\nRMS: {rms:.2f}', 
                        alpha=0.7)

        for col in dP_F_df.columns:
            if col.startswith('dP_dz_total'):
                sensor_name = col.replace('dP_dz_total_', '')
                if sensor_name == sensor_Yokogawa:
                    sensor_range = sensor_name.split('-')[3]
                    label_base = f'Yokogawa {sensor_range}'
                elif sensor_name == sensor_Endress:
                    sensor_range = sensor_name.split('-')[3]
                    label_base = f'Endress {sensor_range}'
                else:
                    label_base = sensor_name
                
                dP_dz_total = dP_F_df[col].iloc[0]
                series = pd.Series([dP_dz_total] * len(dP_F_df), index=dP_F_df.index)
                
                plt.plot(df['X_Value'].iloc[start_idx:end_idx], series, 
                        label=f'dP/dz total {label_base}\nValue: {dP_dz_total:.2f}', 
                        linestyle=':', linewidth=2)
        
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel(r'$ \frac{\Delta P}{L}$ [Pa/m]', fontsize=12)
        plt.title(r'Pressure Gradients', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        output_path_dpf = os.path.join(output_dir, f"dP_F_dz-{base_name}.png")
        plt.savefig(output_path_dpf, bbox_inches='tight')
        plt.close(plt.gcf())
    else:
        pass

if __name__ == "__main__":

    df, data_teste = read_file(file_path)

    if not check_required_columns(df, colunas_analise):
        sys.exit(1)

    fluid_1, fluid_2, direction, theta, ID, is_validation = extract_info_from_filename(file_path)
    print(f"\nInformações extraídas do nome do arquivo:")
    print(f"Fluido 1: {fluid_1}")
    print(f"Fluido 2: {fluid_2}")
    print(f"Direção: {direction}")
    print(f"Inclinação (theta): {theta}°")
    print(f"ID do ponto: {ID}")
    print(f"Ponto de validação: {'Sim' if is_validation else 'Não'}")

    output_dir = os.path.dirname(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    if any(col[0] == 'Alpha' for col in colunas_analise):
        alpha_df_full = calc_alpha(df, 0, len(df))
        df['Alpha'] = alpha_df_full['Alpha'].values

    if any(col[0] == 'rho_g' for col in colunas_analise):
        try:
            pressao_gas_bar_full = df['PIT-M-0101']
            temp_gas_celsius_full = df['TIT-M-0101']
            pressao_gas_pa_full = (pressao_gas_bar_full + 1) * 1e5
            temp_gas_k_full = temp_gas_celsius_full + 273.15
            rho_g_full = [PropsSI('D', 'P', p, 'T', t, 'Air') for p, t in zip(pressao_gas_pa_full, temp_gas_k_full)]
            df['rho_g'] = rho_g_full
        except Exception as e:
            df['rho_g'] = (pressao_gas_pa_full) / (8.314 * temp_gas_k_full)
    
    if any(col[0] == 'rho_g_parede' for col in colunas_analise):
        try:
            pressao_gas_parede_bar_full = df['PIT-A-0301']
            temp_gas_parede_celsius_full = df['TIT-A-0301']
            pressao_gas_parede_pa_full = (pressao_gas_parede_bar_full + 1) * 1e5
            temp_gas_parede_k_full = temp_gas_parede_celsius_full + 273.15
            rho_g_parede_full = [PropsSI('D', 'P', p, 'T', t, 'Air') for p, t in zip(pressao_gas_parede_pa_full, temp_gas_parede_k_full)]
            df['rho_g_parede'] = rho_g_parede_full
        except Exception as e:
            df['rho_g_parede'] = (pressao_gas_parede_pa_full) / (8.314 * temp_gas_parede_k_full)

    dP_F_df_full = calc_frictional_pressure_gradient(df, colunas_analise, 0, len(df), len(df), alpha_df_full['Alpha'])
    for col in dP_F_df_full.columns:
        df[f'dP_F/dz {col}'] = dP_F_df_full[col].values

    colunas_analise_filtradas = [col for col in colunas_analise if col[0] in df.columns]
    if len(colunas_analise_filtradas) < len(colunas_analise):
        print("Atenção: Algumas colunas de análise não existem no DataFrame e não serão plotadas.")
        for col in colunas_analise:
            if col[0] not in df.columns:
                print(f"- Coluna ausente: {col[0]}")

    plot_time_series(df, colunas_analise_filtradas, output_dir, base_name)

    while True:
        escolha_janela = input("Como deseja definir a janela de análise?\n1. Definir manualmente (tempo inicial e tempo final)\n2. Encontrar janela ótima pelo tamanho (automático)\nDigite 1 para manual ou 2 para automático: ").strip()
        if escolha_janela in ['1', '2']:
            break
        print("Opção inválida. Digite 1 ou 2.")

    if escolha_janela == '1':
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
        start_idx = df[df['X_Value'] >= tempo_inicial].index[0]
        end_idx = df[df['X_Value'] <= tempo_final].index[-1] + 1
        coluna_escolhida = colunas_analise_filtradas[0][0]
        min_std = df[coluna_escolhida].iloc[start_idx:end_idx].std()
        best_window_size = df['X_Value'].iloc[end_idx-1] - df['X_Value'].iloc[start_idx]
        media_janela = df[coluna_escolhida].iloc[start_idx:end_idx].mean()
        min_window_size = best_window_size  # Define min_window_size para o caso manual
        max_window_size = best_window_size  # Define max_window_size para o caso manual
        print(f"Janela manual selecionada: {tempo_inicial:.2f}s a {tempo_final:.2f}s")
        print(f"Média da janela: {media_janela:.4f}")
        print(f"Desvio padrão: {min_std:.4f}")
        print(f"Tamanho da janela: {best_window_size:.1f} segundos")
    else:
        print("Colunas disponíveis para análise:")
        for i, col in enumerate(colunas_analise_filtradas, 1):
            print(f"{i}. {col[0]} ({col[1]})")
        
        while True:
            try:
                escolha = int(input("Escolha o número da variável para usar como critério de janelamento: "))
                if 1 <= escolha <= len(colunas_analise_filtradas):
                    coluna_escolhida = colunas_analise_filtradas[escolha-1][0]
                    print(f"Variável escolhida: {coluna_escolhida}")
                    break
                else:
                    print(f"Por favor, escolha um número entre 1 e {len(colunas_analise_filtradas)}")
            except ValueError:
                print("Entrada inválida. Digite um número válido.")

        while True:
            try:
                min_window_size = float(input("Digite o tamanho mínimo da janela em segundos: "))
                if min_window_size > 0:
                    break
                else:
                    print("O tamanho mínimo da janela deve ser maior que zero.")
            except ValueError:
                print("Entrada inválida. Por favor, digite um número válido.")
        while True:
            try:
                max_window_size = float(input("Digite o tamanho máximo da janela em segundos: "))
                if max_window_size >= min_window_size:
                    break
                else:
                    print("O tamanho máximo da janela deve ser maior ou igual ao tamanho mínimo.")
            except ValueError:
                print("Entrada inválida. Por favor, digite um número válido.")
        start_idx, end_idx, min_std, best_window_size = find_min_std_window(
            df, coluna_escolhida, min_window_size, max_window_size)
        media_janela = df[coluna_escolhida].iloc[start_idx:end_idx].mean()
        print(f"Janela ótima selecionada: {df['X_Value'].iloc[start_idx]:.2f}s a {df['X_Value'].iloc[end_idx-1]:.2f}s")
        print(f"Média da janela: {media_janela:.4f}")
        print(f"Desvio padrão: {min_std:.4f}")
        print(f"Tamanho da janela: {best_window_size:.1f} segundos")

    print("\nCalculando e exibindo a incerteza tipo A para cada variável na janela...")
    nomes, medias, desvios, uAs = uncert_propagation(df, colunas_analise_filtradas, start_idx, end_idx, best_window_size)
    
    plt.figure(figsize=(15, 8))
    
    plt.plot(df['X_Value'], df[coluna_escolhida], 'b-', label='Full Series', alpha=0.7)
    
    plt.axvspan(df['X_Value'].iloc[start_idx], df['X_Value'].iloc[end_idx-1], alpha=0.3, color='red', label=f'Window = {best_window_size:.0f} s')
    
    plt.axhline(y=media_janela, color='g', linestyle='--', label=f'Mean: {media_janela:.4f}')
    
    apelido_escolhido = coluna_escolhida
    unidade_escolhida = ''
    for col in colunas_analise_filtradas:
        if isinstance(col, (list, tuple)) and col[0] == coluna_escolhida:
            apelido_escolhido = col[1]
            unidade_escolhida = col[2]
            break
    
    plot_windows(df, colunas_analise_filtradas, start_idx, end_idx, best_window_size, output_dir, base_name)
    
    alpha_df = plot_alpha(df, start_idx, end_idx, output_dir, base_name)
    
    alpha_series = alpha_df['Alpha'] if alpha_df is not None else None
    dP_F_df = calc_frictional_pressure_gradient(df, colunas_analise_filtradas, start_idx, end_idx, best_window_size, alpha_series=alpha_series)
    
    save_results(df, coluna_escolhida, start_idx, end_idx, min_std, media_janela,
                min_window_size, max_window_size, best_window_size, file_path, data_teste,
                fluid_1, fluid_2, direction, theta, ID,
                nomes=colunas_analise_filtradas, medias=medias, desvios=desvios, uAs=uAs,
                escolha_janela=escolha_janela)
    
    plot_pressure_gradients(df, dP_F_df, start_idx, end_idx, output_dir, base_name, sensor_Yokogawa, sensor_Endress, direction)
    print("Arquivo excel dos dados tratados criado...")



    print('##########################################################################')
    print('#AEEEE! Parabéns executado com sucesso, boa sorte com a análise de dados!#')
    print('##########################################################################')