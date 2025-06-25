import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from CoolProp.CoolProp import PropsSI
import tempfile

# Funções utilitárias copiadas/adaptadas do exp_unc.py
# (Inclua aqui as funções: read_file, check_required_columns, plot_time_series, plot_windows, calc_alpha, calc_frictional_pressure_gradient, uncert_propagation, extract_info_from_filename, etc)

# Exemplo de função adaptada para Streamlit (as demais devem ser adaptadas de forma semelhante)
def read_file(file_path: str):
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
    if 'J Ar' in df.columns:
        df['J Ar corrigido'] = df['J Ar'] * (1 - 0.06675)
    if 'J Agua' in df.columns:
        df['J Agua corrigido'] = df['J Agua'] * (1 - 0.06675)
    return df, data_teste

def extract_info_from_filename(filename: str):
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

# --- INÍCIO DO APP STREAMLIT ---
st.set_page_config(page_title="Análise Experimental LEMI", layout="wide")
st.title("Análise Experimental LEMI - exp_unc")

st.markdown("Faça upload do arquivo de dados (.dat ou .txt):")
uploaded_file = st.file_uploader("Escolha o arquivo de dados", type=["dat", "txt"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name
    df, data_teste = read_file(temp_path)
    st.success("Arquivo carregado com sucesso!")
    st.write(f"Data do teste experimental: {data_teste}")
    st.write(f"Dimensões do DataFrame: {df.shape}")
    st.write("Colunas disponíveis:", list(df.columns))
    # Extração de informações do nome do arquivo
    fluid_1, fluid_2, direction, theta, ID, is_validation = extract_info_from_filename(uploaded_file.name)
    st.info(f"Fluido 1: {fluid_1}\nFluido 2: {fluid_2}\nDireção: {direction}\nInclinação: {theta}°\nID: {ID}\nPonto de validação: {'Sim' if is_validation else 'Não'}")
    # Seleção de coluna para análise
    colunas_disponiveis = [col for col in df.columns if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]]
    coluna_escolhida = st.selectbox("Escolha a coluna para análise de série temporal e janela:", colunas_disponiveis)
    # Plot série temporal
    if st.button("Plotar série temporal"):
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df['X_Value'], df[coluna_escolhida], label=coluna_escolhida)
        ax.set_xlabel('Tempo (s)')
        ax.set_ylabel(coluna_escolhida)
        ax.set_title(f'Série temporal - {coluna_escolhida}')
        ax.grid(True)
        st.pyplot(fig)
    # (Aqui você pode adicionar mais opções: seleção de janela, cálculo de estatísticas, plot de janelas, etc)
else:
    st.warning("Faça upload de um arquivo para começar.") 