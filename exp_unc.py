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
file_path = 'data_example/example/SF6/SOD30P05/SOD30P05' #Insira o caminho do arquivo a ser analisado NOTE: USE SEMPRE A BARRA NORMAL '/', SE ESTIVER INVERTIDA, MODIFIQUE-A

Di = 0.05251     # m diâmetro interno da tubulação (para número de Reynolds)
L = 1.70         # m comprimento entre as tomadas de diferencial de pressão
L2 = 2.5         # m comprimento entre as tomadas de diferencial de pressão LEFT AND RIGHT ENDRESS
# Valores de calibração do densitômetro IMPORTANTE
# Serão lidos automaticamente da primeira linha da coluna Comment
I_G = None  # Será preenchido automaticamente
I_L = None  # Será preenchido automaticamente

### SENSORES NA TOMADA CENTRAL L = 1.70m ###
sensor_Yokogawa_mid = 'PDT-M-0101D-10kPa(Pa)'
sensor_Endress_mid = 'PDT-M-0101C-3kPa(Pa)' 

### SENSORES NAS TOMADAS TOP E BOTTOM L = 2.50m ###
sensor_Endress_top = 'PDT-M-0101-40kPa(Pa)'
sensor_Endress_bottom = 'PDT-M-0101B-10kPa(Pa)'

pressao_mesa = 'PIT-M-0301(bar)'
temperatura_mesa = 'TIT-M-0301(C)'

pressao_parede = 'PIT-S-0501(bar)'
temperatura_parede = 'TIT-S-0501(C)'
### Colunas de interesse -> Insira o nome das colunas a plotar e avaliar do arquivo .dat
# Lista de colunas para análise: [nome_coluna, apelido, unidade]
colunas_analise = [
    [sensor_Endress_top, r'\Delta P_{40\,kPa} / L2', r'[Pa/m] (top)'],
    [sensor_Endress_mid, r'\Delta P_{3\,kPa} / L', r'[Pa/m] (mid)'],
    [sensor_Endress_bottom, r'\Delta P_{10\,kPa} / L2', r'[Pa/m] (bottom)'],
    ['Alpha', r'\alpha', r''],
    [sensor_Yokogawa_mid, r'\Delta P_{10\,kPa} / L', r'[Pa/m] (mid)'],
    ['J_Oil(m/s)', r'J_{oil}', r'[m/s]'],
    ['J_SF6(m/s)', r'J_{SF_6}', r'[m/s]'],
    #[pressao_mesa, r'Gauge\ Pressure', r'[Bar]'],
    [temperatura_mesa, r'Temperature', r'[°C]'],
    ['rho_g', r'\rho_{SF_6}', r'[kg/m³]'],
]
####################################################################################################################################################
#       '                                   END INPUTS
####################################################################################################################################################
# Constantes físicas
g = 9.81        # m/s² 
rho_s = 962     # kg/m³ - Densidade do silicone nas tomadas do sensor Yokogawa

def calc_liquid_density(temp_celsius, fluid_2):
    """
    Calcula a densidade do líquido baseado na temperatura.
    
    Args:
        temp_celsius: Temperatura em Celsius (pode ser Series, array ou escalar)
        fluid_2: Tipo de fluido ('Water' ou 'Oil')
        
    Returns:
        Densidade do líquido em kg/m³ (Series se entrada for Series, caso contrário escalar)
    """
    if fluid_2 == 'Water':
        if isinstance(temp_celsius, pd.Series):
            temp_k = temp_celsius + 273.15
            return pd.Series([PropsSI('D', 'T', t, 'P', 101325, 'Water') for t in temp_k], index=temp_celsius.index)
        else:
            temp_k = temp_celsius + 273.15
            return PropsSI('D', 'T', temp_k, 'P', 101325, 'Water')
    elif fluid_2 == 'Oil':
        # return -0.65178 * temp_celsius + 879.76961:
        # return 0.031267*temp_celsius**2 - 3.2050*temp_celsius + 97.6594 #Viscosity model
        return 0.0008*temp_celsius**2 - 0.698*temp_celsius + 879.154
    else:
        return 1000  # Valor padrão


def calc_liquid_viscosity(temp_celsius, fluid_2):
    """
    Calcula a viscosidade dinâmica do líquido (Pa·s).
    Water: CoolProp. Oil: correlação em temperatura (°C).
    """
    if fluid_2 == 'Water':
        if isinstance(temp_celsius, pd.Series):
            temp_k = temp_celsius + 273.15
            return pd.Series([PropsSI('V', 'T', t, 'P', 101325, 'Water') for t in temp_k], index=temp_celsius.index)
        temp_k = temp_celsius + 273.15
        return PropsSI('V', 'T', temp_k, 'P', 101325, 'Water')
    elif fluid_2 == 'Oil':
        # Equação em cP; converter para Pa·s (1 cP = 1e-3 Pa·s)
        mu_cp = 0.031267 * (temp_celsius**2) - 3.2050 * temp_celsius + 97.6594
        return mu_cp * 1e-3
    else:
        return 1e-3  # Valor padrão (água aproximada) em Pa·s


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
                     decimal='.',
                     na_values=[''],
                     encoding='utf-8',
                     names=column_names)
    
    # Extrair I_G e I_L da primeira linha da coluna Comment
    i_g = None
    i_l = None
    
    if 'Comment' in df.columns and len(df) > 0:
        first_comment = str(df['Comment'].iloc[0])
        if 'I_G =' in first_comment and 'I_L =' in first_comment:
            try:
                # Extrair I_G
                ig_start = first_comment.find('I_G =') + 5  # +5 para pular "I_G ="
                ig_end = first_comment.find('/', ig_start)
                if ig_end == -1:
                    ig_end = len(first_comment)
                i_g = float(first_comment[ig_start:ig_end])
                
                # Extrair I_L
                il_start = first_comment.find('I_L =') + 5  # +5 para pular "I_L ="
                il_end = first_comment.find('/', il_start)
                if il_end == -1:
                    il_end = len(first_comment)
                i_l = float(first_comment[il_start:il_end])
                
                print(f"Valores extraídos da coluna Comment:")
                print(f"I_G = {i_g}")
                print(f"I_L = {i_l}")
                
            except (ValueError, IndexError) as e:
                print(f"Erro ao extrair I_G e I_L da coluna Comment: {e}")
                print(f"Conteúdo da primeira linha: {first_comment}")
    
    return df, data_teste, i_g, i_l

def uncertainties_calc(resumo_df,window_df):
     

    # Incerteza dos sensores Yokogawa
    dP_yokogawa = None
    if sensor_Yokogawa_mid is not None:
        if '-30' in sensor_Yokogawa_mid:
            span_yokogawa = 29E3  
        elif '-10' in sensor_Yokogawa_mid:
            span_yokogawa = 9E3    
        elif '-50' in sensor_Yokogawa_mid:
            span_yokogawa = 49E3
        else:
            print('Sensor Yokogawa não suportado')
            exit()
        udP = 0.0005*span_yokogawa
        dP_mean = np.mean(window_df[sensor_Yokogawa_mid])
        dP_yokogawa = unc.ufloat(dP_mean,udP)
        print(sensor_Yokogawa_mid)
        print(f'dP: {dP_yokogawa:.3f}')
    # Incerteza de Alpha estimada
    uAlpha = resumo_df['Alpha'].iloc[0] * 0.05 # Padrão de 5 % da média do alpha
    Alpha = unc.ufloat(resumo_df['Alpha'].iloc[0],uAlpha)
    print('Alpha: ', Alpha)

    T_mean = resumo_df[temperatura_mesa].iloc[0]
    uT = 0.15 + 0.02*T_mean
    T = unc.ufloat(T_mean,uT)
    T_abs = T + 273.15

    P_mean = (resumo_df[pressao_mesa].iloc[0] + 1) * 1E5            #Absolute pressure in Pa
    uP = 0.0025*P_mean
    P = unc.ufloat(P_mean,uP)
    print('P: ', P)
    
    # Cálculo da incerteza da densidade do gás
    if fluid_1 == 'Air': M_ar = 28.96e-3 #[kg/kmol]			    # Massa molecular do ar em [kg/kmol] 
    elif fluid_1 == 'SF6': M_ar = 146.06e-3 #[kg/kmol]			    # Massa molecular do SF6 em [kg/kmol] 
    else:
        print('Fluido 1 não suportado')
        exit()
    R = 8.314 #[kJ/kmolK]				    # Constante universal dos gases [kJ/kmol.K] 
    rho_G = (P * M_ar) / (R * T_abs)	# Densidade média do ar em [kg/m^3] 
    u_rho_g = rho_G.std_dev
    print('u_rho_g: ', rho_G)

    # Cálcuo da incerteza da densidade do líquido (mesma equação que calc_liquid_density)
    if fluid_2 == 'Water':
        rho_L = - 0.0042*(T**2) - 0.0529*T + 1000.9		# Média da densidade do líquido
        print('rho_agua')
    elif fluid_2 == 'Oil':
        rho_L = 0.0008*(T**2) - 0.698*T + 879.154
        print('rho_óleo')
    u_rho_l = rho_L.std_dev
    print('u_rho_l: ', rho_L)

    rho_tubbing = rho_s
    theta_rad = np.deg2rad(theta)
    sin_theta = np.sin(theta_rad)
    
    uL = 0.5E-3
    L_ = unc.ufloat(L, uL)
    uL2 = 0.5E-3
    L2_ = unc.ufloat(L2, uL2)

    # Calcular densidade média da mistura uma única vez
    rho_mistura = (1-Alpha)*rho_L + Alpha*rho_G
    dPg_dz = rho_mistura * g * sin_theta
    
    # Cálculo da incerteza do friccional para Yokogawa (usando L)
    if dP_yokogawa is not None:
        termo_gravitacional = (rho_mistura - rho_tubbing) * g * sin_theta
        if direction in ['Upward']:
            dPf_dz = (dP_yokogawa/L_) - termo_gravitacional
            dPt_dz = abs(dPf_dz) + abs(dPg_dz)
        elif direction in ['Downward','Horizontal']:
            print('Downward')
            dPf_dz = -(dP_yokogawa/L_) + termo_gravitacional
            dPt_dz = abs(dPf_dz) - abs(dPg_dz)
        
        print(f'dPf_dz (Yokogawa): {dPf_dz:.3f}')
        print(f'dPg_dz: {dPg_dz:.3f}')
        print(f'dPt_dz (Yokogawa): {dPt_dz:.3f}')
        
        # Alocando as incertezas do Yokogawa
        udPf_dz_yokogawa = dPf_dz.std_dev
        udPt_dz_yokogawa = dPt_dz.std_dev
    
    # Alocando as incertezas gerais
    udPg_dz = dPg_dz.std_dev

    # Adicionar a incerteza ao resumo_df logo após a coluna dP_F_dz do sensor Yokogawa
    if sensor_Yokogawa_mid is not None and dP_yokogawa is not None:
        col_name = f'dP_F/dz {sensor_Yokogawa_mid}'
        unc_col_name = f'udP_F_dz_{sensor_Yokogawa_mid}'
        if col_name in resumo_df.columns:
            items = list(resumo_df.iloc[0].items())
            idx = [i for i, (k, v) in enumerate(items) if k == col_name]
            if idx:
                insert_pos = idx[0] + 1
                items.insert(insert_pos, (unc_col_name, udPf_dz_yokogawa))
                resumo_df = pd.DataFrame([dict(items)])
            else:
                resumo_df[unc_col_name] = udPf_dz_yokogawa
        else:
            resumo_df[unc_col_name] = udPf_dz_yokogawa
    
    # Processar sensores Endress (top e bottom) - usando L2
    # Armazenar valores calculados para evitar recálculo
    endress_results = {}
    for sensor_Endress in [sensor_Endress_top, sensor_Endress_bottom, sensor_Endress_mid]:
        if sensor_Endress is not None:
            if '-3' in sensor_Endress:
                span_endress = 6E3
            elif '-10' in sensor_Endress:
                span_endress = 20E3
            elif '-40' in sensor_Endress:
                span_endress = 80E3
            else:
                print(f'Sensor Endress {sensor_Endress} não suportado')
                continue
            udP = 0.00055*span_endress
            dP_mean = np.mean(window_df[sensor_Endress])
            dP_endress = unc.ufloat(dP_mean, udP)
            print(sensor_Endress)
            print(f'dP: {dP_endress:.3f}')
            
            # Calcular incertezas para este sensor Endress usando L2
            # Para Endress, rho_tubbing é rho_liquido, não rho_s
            rho_tubbing_endress = rho_L
            termo_gravitacional_endress = (rho_mistura - rho_tubbing_endress) * g * sin_theta
            if direction in ['Upward']:
                dPf_dz_endress = (dP_endress/L2_) - termo_gravitacional_endress
                dPt_dz_endress = dPf_dz_endress + dPg_dz
            elif direction in ['Downward','Horizontal']:
                dPf_dz_endress = -(dP_endress/L2_) + termo_gravitacional_endress
                dPt_dz_endress = -(abs(dPf_dz_endress)) + dPg_dz
            
            udPf_dz_endress = dPf_dz_endress.std_dev
            udPt_dz_endress = dPt_dz_endress.std_dev
            
            # Armazenar resultados para uso posterior
            endress_results[sensor_Endress] = {
                'udPf_dz': udPf_dz_endress,
                'udPt_dz': udPt_dz_endress
            }
            
            # Adicionar incerteza friccional
            col_name = f'dP_F/dz {sensor_Endress}'
            unc_col_name = f'udP_F_dz_{sensor_Endress}'
            if col_name in resumo_df.columns:
                items = list(resumo_df.iloc[0].items())
                idx = [i for i, (k, v) in enumerate(items) if k == col_name]
                if idx:
                    insert_pos = idx[0] + 1
                    items.insert(insert_pos, (unc_col_name, udPf_dz_endress))
                    resumo_df = pd.DataFrame([dict(items)])
                else:
                    resumo_df[unc_col_name] = udPf_dz_endress
            else:
                resumo_df[unc_col_name] = udPf_dz_endress


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
    if sensor_Yokogawa_mid is not None and dP_yokogawa is not None:
        total_col_name = f'dP_dz_total_{sensor_Yokogawa_mid}'
        total_unc_col_name = f'udP_dz_total_{sensor_Yokogawa_mid}'
        if total_col_name in resumo_df.columns:
            items = list(resumo_df.iloc[0].items())
            idx = [i for i, (k, v) in enumerate(items) if k == total_col_name]
            if idx:
                insert_pos = idx[0] + 1
                items.insert(insert_pos, (total_unc_col_name, udPt_dz_yokogawa))
                resumo_df = pd.DataFrame([dict(items)])
            else:
                resumo_df[total_unc_col_name] = udPt_dz_yokogawa
        else:
            resumo_df[total_unc_col_name] = udPt_dz_yokogawa
            
    # Adicionar incertezas totais dos sensores Endress (usando valores já calculados)
    for sensor_Endress in [sensor_Endress_top, sensor_Endress_bottom, sensor_Endress_mid]:
        if sensor_Endress is not None and sensor_Endress in endress_results:
            udPt_dz_endress = endress_results[sensor_Endress]['udPt_dz']
            total_col_name = f'dP_dz_total_{sensor_Endress}'
            total_unc_col_name = f'udP_dz_total_{sensor_Endress}'
            if total_col_name in resumo_df.columns:
                items = list(resumo_df.iloc[0].items())
                idx = [i for i, (k, v) in enumerate(items) if k == total_col_name]
                if idx:
                    insert_pos = idx[0] + 1
                    items.insert(insert_pos, (total_unc_col_name, udPt_dz_endress))
                    resumo_df = pd.DataFrame([dict(items)])
                else:
                    resumo_df[total_unc_col_name] = udPt_dz_endress
            else:
                resumo_df[total_unc_col_name] = udPt_dz_endress

    return resumo_df


def _gradient_column_name_for_excel(old_name: str) -> str:
    """
    Converte o nome da coluna de gradiente de pressão para a nomenclatura usada na planilha _processed.
    Friccional (F), gravitacional (G), total (T). Unidades (Pa) -> (Pa/m).
    """
    if old_name.startswith('dP_F/dz ') and 'dP_dz_total_' not in old_name and 'dP_dz_gravitacional' not in old_name:
        sensor = old_name[len('dP_F/dz '):]
        sensor_units = sensor.replace('(Pa)', '(Pa/m)') if '(Pa)' in sensor else sensor + '(Pa/m)'
        return f'-dpdz_F_{sensor_units}'
    if old_name.startswith('udP_F_dz_'):
        sensor = old_name[len('udP_F_dz_'):]
        sensor_units = sensor.replace('(Pa)', '(Pa/m)') if '(Pa)' in sensor else sensor + '(Pa/m)'
        return f'U(-dpdz_F_{sensor_units})'
    if old_name == 'dP_dz_gravitacional':
        return '-dpdz_G_(Pa/m)'
    if old_name == 'udP_dz_gravitacional':
        return 'U(-dpdz_G_(Pa/m))'
    if old_name.startswith('dP_dz_total_'):
        sensor = old_name[len('dP_dz_total_'):]
        sensor_units = sensor.replace('(Pa)', '(Pa/m)') if '(Pa)' in sensor else sensor + '(Pa/m)'
        return f'-dpdz_T_{sensor_units}'
    if old_name.startswith('udP_dz_total_'):
        sensor = old_name[len('udP_dz_total_'):]
        sensor_units = sensor.replace('(Pa)', '(Pa/m)') if '(Pa)' in sensor else sensor + '(Pa/m)'
        return f'U(-dpdz_T_{sensor_units})'
    return old_name


def save_results(df: pd.DataFrame, coluna_escolhida: str, start_idx: int, end_idx: int, min_std: float, media_janela: float, 
                min_window_size: float, max_window_size: float, best_window_size: float, file_path: str, data_teste: str, 
                fluid_1: str, fluid_2: str, direction: str, theta: int, ID: str, escolha_janela: str = None):
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
        "Intensidade do gas densitometro (Ig)": I_G,
        "Intensidade do fluido densitometro (If)": I_L,
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
    
    # Excluir colunas não numéricas do cálculo da média
    colunas_numericas = window_data.select_dtypes(include=[np.number]).columns
    # Remover a coluna Comment se existir
    if 'Comment' in colunas_numericas:
        colunas_numericas = colunas_numericas.drop('Comment')
    medias_janela = window_data[colunas_numericas].mean()
    
    colunas_corrigidas = []
    coluna_gravitacional = None
    
    for col in df.columns:
        # Pular colunas não numéricas como 'Comment'
        if col == 'Comment':
            continue
        elif col.startswith('dP_F/dz dP_dz_total_'):
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
            if direction in ['Upward']:
                dP_dz_total = dP_F_RMS + grav_term
            elif direction in ['Downward', 'Horizontal']:
                dP_dz_total = dP_F_RMS - grav_term
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
        if temperatura_mesa in window_data.columns:
            temp_liquido_celsius = window_data[temperatura_mesa]
            rho_liquido_vals = calc_liquid_density(temp_liquido_celsius, fluid_2)
            if isinstance(rho_liquido_vals, pd.Series):
                rho_liquido_medio = rho_liquido_vals.mean()
            else:
                rho_liquido_medio = float(rho_liquido_vals)
        else:
            rho_liquido_medio = 1000
    except Exception as e:
        rho_liquido_medio = 1000

    # Médias de viscosidade e Reynolds na janela (para planilha)
    mu_g_medio = window_data['mu_g'].mean() if 'mu_g' in window_data.columns else np.nan
    mu_liquido_medio = window_data['mu_liquido'].mean() if 'mu_liquido' in window_data.columns else np.nan
    Re_sg_medio = window_data['Re_sg'].mean() if 'Re_sg' in window_data.columns else np.nan
    Re_sl_medio = window_data['Re_sl'].mean() if 'Re_sl' in window_data.columns else np.nan
    J_gas_col = get_J_column_for_fluid(window_data, fluid_1)
    J_liquido_col = get_J_column_for_fluid(window_data, fluid_2)
    J_gas_medio = medias_janela.get(J_gas_col, np.nan) if J_gas_col else np.nan
    J_liquido_medio = medias_janela.get(J_liquido_col, np.nan) if J_liquido_col else np.nan

    # Inserir na ordem: rho_g, mu_g, rho_liquido, mu_liquido, J_gas, J_liquido, Re_sg, Re_sl
    cols_remover = {'mu_g', 'rho_liquido', 'mu_liquido', 'Re_sg', 'Re_sl'}
    if J_gas_col:
        cols_remover.add(J_gas_col)
    if J_liquido_col:
        cols_remover.add(J_liquido_col)
    if 'rho_g' in resumo_dict:
        items = [(k, v) for k, v in resumo_dict.items() if k not in cols_remover]
        idx = next((i for i, (k, v) in enumerate(items) if k == 'rho_g'), None)
        if idx is not None:
            items.insert(idx + 1, ('mu_g', mu_g_medio))
            items.insert(idx + 2, ('rho_liquido', rho_liquido_medio))
            items.insert(idx + 3, ('mu_liquido', mu_liquido_medio))
            pos = idx + 4
            if J_gas_col:
                items.insert(pos, (J_gas_col, J_gas_medio))
                pos += 1
            if J_liquido_col:
                items.insert(pos, (J_liquido_col, J_liquido_medio))
                pos += 1
            items.insert(pos, ('Re_sg', Re_sg_medio))
            items.insert(pos + 1, ('Re_sl', Re_sl_medio))
            resumo_dict = dict(items)
        else:
            resumo_dict['rho_liquido'] = rho_liquido_medio
            resumo_dict['mu_g'] = mu_g_medio
            resumo_dict['mu_liquido'] = mu_liquido_medio
            if J_gas_col:
                resumo_dict[J_gas_col] = J_gas_medio
            if J_liquido_col:
                resumo_dict[J_liquido_col] = J_liquido_medio
            resumo_dict['Re_sg'] = Re_sg_medio
            resumo_dict['Re_sl'] = Re_sl_medio
    else:
        resumo_dict['rho_liquido'] = rho_liquido_medio
        resumo_dict['mu_g'] = mu_g_medio
        resumo_dict['mu_liquido'] = mu_liquido_medio
        if J_gas_col:
            resumo_dict[J_gas_col] = J_gas_medio
        if J_liquido_col:
            resumo_dict[J_liquido_col] = J_liquido_medio
        resumo_dict['Re_sg'] = Re_sg_medio
        resumo_dict['Re_sl'] = Re_sl_medio

    resumo_df = pd.DataFrame([resumo_dict])
    
    window_df = pd.DataFrame()

    for col in colunas_corrigidas:
        # Pular a coluna Comment
        if col == 'Comment':
            continue
        elif col == 'dP_dz_gravitacional':
            window_df[col] = window_data['dP_F/dz dP_dz_gravitacional']
        elif col.startswith('dP_dz_total_'):
            window_df[col] = dP_dz_total_values[col]
        else:
            window_df[col] = window_data[col]
    
    window_df.insert(0, 'Time (s)', window_data['X_Value'])
    # Ordem: rho_g, mu_g, rho_liquido, mu_liquido, J_gas, J_liquido, Re_sg, Re_sl
    cols = list(window_df.columns)
    for (a, b) in [('rho_g', 'mu_g'), ('mu_g', 'rho_liquido'), ('rho_liquido', 'mu_liquido')]:
        if a in cols and b in cols and cols.index(b) != cols.index(a) + 1:
            cols.remove(b)
            cols.insert(cols.index(a) + 1, b)
    # Bloco J_gas, J_liquido, Re_sg, Re_sl logo após mu_liquido
    block = [c for c in [J_gas_col, J_liquido_col, 'Re_sg', 'Re_sl'] if c and c in cols]
    for c in block:
        cols.remove(c)
    if block and 'mu_liquido' in cols:
        pos = cols.index('mu_liquido') + 1
        for i, c in enumerate(block):
            cols.insert(pos + i, c)
    window_df = window_df[cols]
    
    # Atualizar resumo_df com as incertezas e salvar no Excel
    resumo_df = uncertainties_calc(resumo_df, window_df)

    # Aplicar nova nomenclatura de gradiente (F/G/T e Pa/m) nas colunas da planilha _processed
    resumo_df = resumo_df.rename(columns={c: _gradient_column_name_for_excel(c) for c in resumo_df.columns})
    window_df = window_df.rename(columns={c: _gradient_column_name_for_excel(c) for c in window_df.columns})

    # Unidades no nome das colunas rho, mu e Re na planilha _processed
    _col_units = {
        'rho_g': 'rho_g(kg/m³)',
        'rho_liquido': 'rho_liquido(kg/m³)',
        'mu_g': 'mu_g(Pa.s)',
        'mu_liquido': 'mu_liquido(Pa.s)',
        'Re_sg': 'Re_sg(-)',
        'Re_sl': 'Re_sl(-)',
    }
    resumo_df = resumo_df.rename(columns={c: _col_units.get(c, c) for c in resumo_df.columns})
    window_df = window_df.rename(columns={c: _col_units.get(c, c) for c in window_df.columns})

    def _write_excel(path):
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            header_df.to_excel(writer, sheet_name='Info', index=False)
            resumo_df.to_excel(writer, sheet_name='Resumo Medias', index=False, header=True)
            window_df.to_excel(writer, sheet_name='Serie Temporal', index=False)

    try:
        _write_excel(output_file)
    except PermissionError:
        # Arquivo pode estar aberto no Excel; gravar em temporário e tentar substituir
        tmp_file = output_file + '.tmp'
        _write_excel(tmp_file)
        try:
            os.replace(tmp_file, output_file)
        except OSError:
            # Destino ainda bloqueado; salvar com nome alternativo
            alt_file = os.path.join(diretorio, f"{base_name}_processed_NEW.xlsx")
            os.replace(tmp_file, alt_file)
            print(f"\nAtenção: não foi possível gravar em '{output_file}' (arquivo pode estar aberto).")
            print(f"Resultado salvo em: {alt_file}")
    

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
            # Determinar qual comprimento usar (L para Yokogawa, L2 para Endress)
            if nome_coluna in [sensor_Yokogawa_mid, sensor_Endress_mid]:
                comprimento = L
            elif nome_coluna in [sensor_Endress_top, sensor_Endress_bottom]:
                comprimento = L2
            else:
                comprimento = L  # Padrão para outros sensores PDT
            y_data = df[nome_coluna]/comprimento
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
            # Determinar qual comprimento usar (L para Yokogawa, L2 para Endress)
            if nome_coluna in [sensor_Yokogawa_mid, sensor_Endress_mid]:
                comprimento = L
            elif nome_coluna in [sensor_Endress_top, sensor_Endress_bottom]:
                comprimento = L2
            else:
                comprimento = L  # Padrão para outros sensores PDT
            y_data = (df[nome_coluna].iloc[start_idx:end_idx])/comprimento
            y_data_full = (df[nome_coluna])/comprimento
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

def calc_alpha(df, start_idx, end_idx):
    """
    Determina a fração de vazio (alpha) na mistura a partir dos dados do densitômetro na janela selecionada.
    Retorna um DataFrame contendo a série temporal de Alpha na janela.
    """
    dados_densitometro = df['I_Densitometer'].iloc[start_idx:end_idx]
    alpha_series = np.log(dados_densitometro / I_L) / np.log(I_G / I_L)
    
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
        # Verificar fluido
        pressao_gas_bar = df[pressao_mesa].iloc[start_idx:end_idx]
        temp_gas_celsius = df[temperatura_mesa].iloc[start_idx:end_idx]
        
       
        pressao_gas_pa = (pressao_gas_bar + 1) * 1e5
        temp_gas_k = temp_gas_celsius + 273.15
        
        rho_gas = [PropsSI('D', 'P', p, 'T', t, fluid_1) for p, t in zip(pressao_gas_pa, temp_gas_k)]
        rho_gas = pd.Series(rho_gas, index=pressao_gas_pa.index)
    
        try:
            temp_liquido_celsius = df[temperatura_mesa].iloc[start_idx:end_idx]
            rho_liquido = calc_liquid_density(temp_liquido_celsius, fluid_2)
            if not isinstance(rho_liquido, pd.Series):
                rho_liquido = pd.Series([rho_liquido] * len(temp_liquido_celsius), index=temp_liquido_celsius.index)
        except Exception as e:
            rho_liquido = pd.Series([1000] * len(df[temperatura_mesa].iloc[start_idx:end_idx]), 
                                   index=df[temperatura_mesa].iloc[start_idx:end_idx].index)

    except KeyError as e:
        rho_gas = None

    # Calcular valores comuns uma única vez
    theta_rad = np.deg2rad(theta)
    sin_theta = np.sin(theta_rad)
    
    # Calcular densidade média da mistura uma única vez
    rho_mistura_series = (1-alpha_series)*rho_liquido + alpha_series*rho_gas
    dP_dz_gravitacional = rho_mistura_series * g * sin_theta
    dP_dz_gravitacional_mean = np.mean(dP_dz_gravitacional)
    
    dP_F_df = pd.DataFrame()

    for i in indices_pdt:
        coluna_pdt_nome = colunas[i][0]
        
        # Determinar rho_tubbing e comprimento baseado no sensor
        if coluna_pdt_nome in [sensor_Yokogawa_mid]:
            rho_tubbing = rho_s
            comprimento = L
        elif coluna_pdt_nome in [sensor_Endress_top, sensor_Endress_bottom,sensor_Endress_mid]:
            rho_tubbing = rho_liquido
            if coluna_pdt_nome in [sensor_Endress_top, sensor_Endress_bottom]:
                comprimento = L2
            else:
                comprimento = L  # Padrão para outros sensores Endress (mid)
        
        delta_p_prime = df[coluna_pdt_nome].iloc[start_idx:end_idx]
        termo_gravitacional = (rho_mistura_series - rho_tubbing) * g * sin_theta

        if direction in ['Upward']:
            dP_F_over_dz_series = (delta_p_prime / comprimento) - termo_gravitacional
            dP_F_over_dz_RMS = np.sqrt(np.mean(np.square(dP_F_over_dz_series)))
            dP_dz_total = dP_F_over_dz_RMS + dP_dz_gravitacional_mean
        elif direction in ['Downward','Horizontal']:
            dP_F_over_dz_series = -(delta_p_prime / comprimento) + termo_gravitacional 
            dP_F_over_dz_RMS = np.sqrt(np.mean(np.square(dP_F_over_dz_series)))
            dP_dz_total = dP_F_over_dz_RMS - dP_dz_gravitacional_mean
        # print(f'dP_F_over_dz_RMS: {dP_F_over_dz_RMS:.3f}')
        # print(f'dP_dz_gravitacional_mean: {dP_dz_gravitacional_mean:.3f}')
        # print(f'dP_dz_total: {dP_dz_total:.3f}')
        # exit()
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
    Se houver texto antes do hífen '-', ignora esse texto e considera apenas após o hífen.
    Se começar com 'V', desloca a leitura em uma casa (ponto de validação).
    Formato esperado: [prefixo-][V]XXX##ID## onde:
    - prefixo: texto opcional antes do hífen (será ignorado)
    - X: letra indicando o fluido (A:Air, W:Water, O:Oil, S:SF6)
    - #: número indicando a inclinação em graus
    - ID: identificador do ponto experimental
    """
    fluid_map = {
        'A': 'Air',
        'W': 'Water',
        'O': 'Oil',
        'S': 'SF6',
        'D': 'Dense Liquid'
    }
    direction_map = {
        'H': 'Horizontal',
        'U': 'Upward',
        'D': 'Downward'
    }
    base_name = os.path.splitext(os.path.basename(filename))[0]
    
    # Se houver hífen, considerar apenas a parte após o primeiro hífen
    if '-' in base_name:
        base_name = base_name.split('-', 1)[1]
    
    offset = 1 if base_name[0] == 'V' else 0
    is_validation = base_name[0] == 'V'
    fluid_1 = fluid_map.get(base_name[0+offset], 'Unknown')
    fluid_2 = fluid_map.get(base_name[1+offset], 'Unknown')
    direction = direction_map.get(base_name[2+offset], 'Unknown')
    theta = int(base_name[3+offset:5+offset])
    ID = base_name[5+offset:]
    
    return fluid_1, fluid_2, direction, theta, ID, is_validation


def get_J_column_for_fluid(df: pd.DataFrame, fluid: str) -> str:
    """
    Retorna o nome da coluna de velocidade superficial J para o fluido,
    a partir da identificação do fluido (ex.: 'SF6', 'Oil', 'Air', 'Water').
    Procura primeiro 'J_<fluid>(m/s)'; se não existir, procura coluna que comece com 'J_' e contenha o nome do fluido.
    """
    if not fluid or fluid == 'Unknown':
        return None
    exact = f'J_{fluid}(m/s)'
    if exact in df.columns:
        return exact
    for c in df.columns:
        if isinstance(c, str) and c.startswith('J_') and fluid in c and '(m/s)' in c:
            return c
    return None


def check_required_columns(df: pd.DataFrame, colunas_analise: list):
    """
    Verifica se as colunas necessárias existem no DataFrame.
    Retorna True se todas existirem, False caso contrário.
    """
    colunas_faltantes = []
    
    colunas_calculadas = ['Alpha', 'rho_g', 'rho_g_parede', 'mu_g', 'mu_liquido', 'rho_liquido', 'Re_sg', 'Re_sl']
    
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

def plot_pressure_gradients(df: pd.DataFrame, dP_F_df: pd.DataFrame, start_idx: int, end_idx: int, output_dir: str, base_name: str, sensor_Yokogawa_mid: str, sensor_Endress_mid: str, sensor_Endress_top: str, sensor_Endress_bottom: str, direction: str):
    """
    Plota as séries temporais de gradientes de pressão e salva a imagem.
    
    Args:
        df (pandas.DataFrame): DataFrame com os dados
        dP_F_df (pandas.DataFrame): DataFrame com os gradientes de pressão
        start_idx (int): Índice inicial da janela
        end_idx (int): Índice final da janela
        output_dir (str): Diretório para salvar a imagem
        base_name (str): Nome base para o arquivo de saída
        sensor_Yokogawa_mid (str): Nome do sensor Yokogawa
        sensor_Endress_mid (str): Nome do sensor Endress mid
        sensor_Endress_top (str): Nome do sensor Endress left
        sensor_Endress_bottom (str): Nome do sensor Endress right
        direction (str): Direção do escoamento
    """
    if dP_F_df is not None and not dP_F_df.empty:
        plt.figure(figsize=(15, 8))

        # Série gravitacional
        grav_series = dP_F_df['dP_dz_gravitacional']
        grav_mean = grav_series.mean()
        grav_std = grav_series.std()
        plt.plot(
            df['X_Value'].iloc[start_idx:end_idx],
            grav_series,
            label=f'dP/dz gravitacional\nmean: {grav_mean:.2f} ± {grav_std:.2f}',
            color='black',
            linestyle='--',
        )

        # Guardar a cor usada para cada sensor (friccional) para reutilizar na série total
        sensor_colors = {}

        # Séries friccionais dP_F/dz
        for col in dP_F_df.columns:
            if not col.startswith('dP_dz_total') and col != 'dP_dz_gravitacional':
                series = dP_F_df[col]
                rms = np.sqrt(np.mean(np.square(series)))

                # Nome da série no formato usado no Excel
                label_excel = _gradient_column_name_for_excel(f'dP_F/dz {col}')

                line, = plt.plot(
                    df['X_Value'].iloc[start_idx:end_idx],
                    series,
                    label=f'{label_excel}\nRMS: {rms:.2f}',
                    alpha=0.7,
                )
                sensor_colors[col] = line.get_color()

        # Séries totais dP/dz_total com a mesma cor do friccional correspondente (apenas tracejado muda)
        for col in dP_F_df.columns:
            if col.startswith('dP_dz_total'):
                sensor_name = col.replace('dP_dz_total_', '')

                dP_dz_total = dP_F_df[col].iloc[0]
                series = pd.Series([dP_dz_total] * len(dP_F_df), index=dP_F_df.index)

                color = sensor_colors.get(sensor_name)

                label_excel = _gradient_column_name_for_excel(col)

                plot_kwargs = {
                    'linestyle': ':',
                    'linewidth': 2,
                    'label': f'{label_excel}\nValue: {dP_dz_total:.2f}',
                }
                if color is not None:
                    plot_kwargs['color'] = color

                plt.plot(
                    df['X_Value'].iloc[start_idx:end_idx],
                    series,
                    **plot_kwargs,
                )

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

    df, data_teste, i_g_extracted, i_l_extracted = read_file(file_path)
    
    # Atualizar as variáveis globais com os valores extraídos
    if i_g_extracted is not None:
        I_G = i_g_extracted
    if i_l_extracted is not None:
        I_L = i_l_extracted
    
    # Verificar se os valores foram extraídos corretamente
    if I_G is None or I_L is None:
        print("ERRO: Não foi possível extrair I_G e I_L da coluna Comment.")
        print("Verifique se a primeira linha da coluna Comment contém: I_G =XXXXX/I_L =XXXXX")
        sys.exit(1)

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

    # Colunas J (velocidade superficial) identificadas automaticamente a partir dos fluidos
    J_gas_col = get_J_column_for_fluid(df, fluid_1)
    J_liquido_col = get_J_column_for_fluid(df, fluid_2)
    if J_gas_col:
        print(f"Coluna J gás: {J_gas_col}")
    if J_liquido_col:
        print(f"Coluna J líquido: {J_liquido_col}")

    output_dir = os.path.dirname(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    if any(col[0] == 'Alpha' for col in colunas_analise):
        alpha_df_full = calc_alpha(df, 0, len(df))
        df['Alpha'] = alpha_df_full['Alpha'].values

    if any(col[0] == 'rho_g' for col in colunas_analise):
        try:
            pressao_gas_bar_full = df[pressao_mesa]
            temp_gas_celsius_full = df[temperatura_mesa]
            pressao_gas_pa_full = (pressao_gas_bar_full + 1) * 1e5
            temp_gas_k_full = temp_gas_celsius_full + 273.15
            rho_g_full = [PropsSI('D', 'P', p, 'T', t, fluid_1) for p, t in zip(pressao_gas_pa_full, temp_gas_k_full)]
            df['rho_g'] = rho_g_full
            # Viscosidade dinâmica do gás (CoolProp para Air, SF6) em Pa·s
            if fluid_1 in ['Air', 'SF6', 'Water']:
                mu_g_full = [PropsSI('V', 'P', p, 'T', t, fluid_1) for p, t in zip(pressao_gas_pa_full, temp_gas_k_full)]
                df['mu_g'] = mu_g_full
        except Exception as e:
            # Garantir que os dados são numéricos
            pressao_gas_pa_full = (df[pressao_mesa] + 1) * 1e5
            temp_gas_k_full = df[temperatura_mesa] + 273.15
            pressao_gas_pa_full_numeric = pd.to_numeric(pressao_gas_pa_full, errors='coerce')
            temp_gas_k_full_numeric = pd.to_numeric(temp_gas_k_full, errors='coerce')
            df['rho_g'] = (pressao_gas_pa_full_numeric) / (8.314 * temp_gas_k_full_numeric)
            if fluid_1 in ['Air', 'SF6', 'Water']:
                try:
                    df['mu_g'] = [PropsSI('V', 'P', p, 'T', t, fluid_1) for p, t in zip(pressao_gas_pa_full, temp_gas_k_full)]
                except Exception:
                    pass
    # Densidade e viscosidade do líquido (CoolProp para Water, equações para Oil) e número de Reynolds
    if temperatura_mesa in df.columns and fluid_2 in ['Water', 'Oil']:
        df['rho_liquido'] = calc_liquid_density(df[temperatura_mesa], fluid_2)
        df['mu_liquido'] = calc_liquid_viscosity(df[temperatura_mesa], fluid_2)

    # Número de Reynolds superficial: Re_s = rho * J * Di / mu (por fase)
    if 'rho_g' in df.columns and 'mu_g' in df.columns and J_gas_col:
        mu_g_safe = df['mu_g'].replace(0, np.nan)
        df['Re_sg'] = (df['rho_g'] * df[J_gas_col] * Di / mu_g_safe).fillna(np.nan)
    if 'rho_liquido' in df.columns and 'mu_liquido' in df.columns and J_liquido_col:
        mu_l_safe = df['mu_liquido'].replace(0, np.nan)
        df['Re_sl'] = (df['rho_liquido'] * df[J_liquido_col] * Di / mu_l_safe).fillna(np.nan)

    if any(col[0] == 'rho_g_parede' for col in colunas_analise):
        try:
            pressao_gas_parede_bar_full = df[pressao_parede]
            temp_gas_parede_celsius_full = df[temperatura_parede]
            pressao_gas_parede_pa_full = (pressao_gas_parede_bar_full + 1) * 1e5
            temp_gas_parede_k_full = temp_gas_parede_celsius_full + 273.15
            rho_g_parede_full = [PropsSI('D', 'P', p, 'T', t, fluid_1) for p, t in zip(pressao_gas_parede_pa_full, temp_gas_parede_k_full)]
            df['rho_g_parede'] = rho_g_parede_full
        except Exception as e:
            # Garantir que os dados são numéricos
            pressao_gas_parede_pa_full_numeric = pd.to_numeric(pressao_gas_parede_pa_full, errors='coerce')
            temp_gas_parede_k_full_numeric = pd.to_numeric(temp_gas_parede_k_full, errors='coerce')
            df['rho_g_parede'] = (pressao_gas_parede_pa_full_numeric) / (8.314 * temp_gas_parede_k_full_numeric)

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

    plot_windows(df, colunas_analise_filtradas, start_idx, end_idx, best_window_size, output_dir, base_name)
    
    alpha_df = plot_alpha(df, start_idx, end_idx, output_dir, base_name)
    
    alpha_series = alpha_df['Alpha'] if alpha_df is not None else None
    dP_F_df = calc_frictional_pressure_gradient(df, colunas_analise_filtradas, start_idx, end_idx, best_window_size, alpha_series=alpha_series)
    
    save_results(df, coluna_escolhida, start_idx, end_idx, min_std, media_janela,
                min_window_size, max_window_size, best_window_size, file_path, data_teste,
                fluid_1, fluid_2, direction, theta, ID, escolha_janela=escolha_janela)
    
    plot_pressure_gradients(df, dP_F_df, start_idx, end_idx, output_dir, base_name, sensor_Yokogawa_mid, sensor_Endress_mid, sensor_Endress_top, sensor_Endress_bottom, direction)
    print("Arquivo excel dos dados tratados criado...")



    print('##########################################################################')
    print('#AEEEE! Parabéns executado com sucesso, boa sorte com a análise de dados!#')
    print('##########################################################################')