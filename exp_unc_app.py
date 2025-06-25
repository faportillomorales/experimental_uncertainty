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
    try:
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
        if header_end_idx == 0:
            st.error('Arquivo não possui dois cabeçalhos ***End_of_Header***. Formato inválido!')
            return None, None
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
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}\nVerifique se o arquivo está no formato correto do LEMI.")
        return None, None

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

def uncert_propagation(df, colunas, start_idx, end_idx):
    n = end_idx - start_idx
    medias = []
    desvios = []
    uAs = []
    for nome_coluna in colunas:
        dados_janela = df[nome_coluna].iloc[start_idx:end_idx]
        media = dados_janela.mean()
        desvio = dados_janela.std(ddof=1)
        uA = desvio / (n ** 0.5)
        medias.append(media)
        desvios.append(desvio)
        uAs.append(uA)
    return medias, desvios, uAs

def plot_time_series(df, colunas):
    fig, axs = plt.subplots(len(colunas), 1, figsize=(12, 3*len(colunas)))
    if len(colunas) == 1:
        axs = [axs]
    for i, col in enumerate(colunas):
        axs[i].plot(df['X_Value'], df[col], label=col)
        axs[i].set_xlabel('Tempo (s)')
        axs[i].set_ylabel(col)
        axs[i].set_title(f'Série temporal - {col}')
        axs[i].grid(True)
        axs[i].legend()
    plt.tight_layout()
    return fig

def plot_window(df, col, start_idx, end_idx):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df['X_Value'], df[col], 'b-', alpha=0.3, label='Série completa')
    ax.plot(df['X_Value'].iloc[start_idx:end_idx], df[col].iloc[start_idx:end_idx], 'r-', label='Janela selecionada')
    ax.set_xlabel('Tempo (s)')
    ax.set_ylabel(col)
    ax.set_title(f'Janela selecionada - {col}')
    ax.grid(True)
    ax.legend()
    return fig

def find_min_std_window(df, column_name, min_window_size, max_window_size):
    min_std = float('inf')
    best_start_idx = 0
    best_end_idx = 0
    for window_size in np.arange(min_window_size, max_window_size + 1, 1):
        for i in range(len(df)):
            start_time = df['X_Value'].iloc[i]
            end_time = start_time + window_size
            end_idx_candidate = df[df['X_Value'] <= end_time].index
            if len(end_idx_candidate) == 0:
                continue
            end_idx = end_idx_candidate[-1]
            if end_idx - i < 2:
                continue
            window_data = df[column_name].iloc[i:end_idx+1]
            current_std = window_data.std()
            if current_std < min_std:
                min_std = current_std
                best_start_idx = i
                best_end_idx = end_idx + 1
    return best_start_idx, best_end_idx

def calc_alpha(df, start_idx, end_idx, I_g, I_f):
    if 'Densitometro' not in df.columns:
        return None
    dados_densitometro = df['Densitometro'].iloc[start_idx:end_idx]
    alpha_series = np.log(dados_densitometro / I_f) / np.log(I_g / I_f)
    alpha_df = pd.DataFrame({
        'X_Value': df['X_Value'].iloc[start_idx:end_idx],
        'Alpha': alpha_series
    })
    return alpha_df

# --- INÍCIO DO APP STREAMLIT ---
st.set_page_config(page_title="Análise Experimental LEMI", layout="wide")
st.title("Análise Experimental LEMI - exp_unc (GUI)")

st.markdown("Faça upload do arquivo de dados (.dat ou .txt):")
uploaded_file = st.file_uploader("Escolha o arquivo de dados", type=["dat", "txt"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name
    df, data_teste = read_file(temp_path)
    if df is None:
        st.stop()
    st.success("Arquivo carregado com sucesso!")
    st.write(f"Data do teste experimental: {data_teste}")
    st.write(f"Dimensões do DataFrame: {df.shape}")
    # st.write("Colunas disponíveis:", list(df.columns))
    # Extração de informações do nome do arquivo
    fluid_1, fluid_2, direction, theta, ID, is_validation = extract_info_from_filename(uploaded_file.name)
    st.info(f"Fluido 1: {fluid_1}\nFluido 2: {fluid_2}\nDireção: {direction}\nInclinação: {theta}°\nID: {ID}\nPonto de validação: {'Sim' if is_validation else 'Não'}")

    # Seleção de colunas para análise
    colunas_numericas = [col for col in df.columns if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]]
    colunas_selecionadas = st.multiselect("Selecione as colunas para análise:", colunas_numericas, default=colunas_numericas[:2])

    # Plot série temporal
    if st.button("Plotar séries temporais"):
        fig = plot_time_series(df, colunas_selecionadas)
        st.pyplot(fig)

    # Seleção de janela
    st.subheader("Seleção de Janela")
    tipo_janela = st.radio("Tipo de seleção de janela:", ["Manual", "Automática"])
    if tipo_janela == "Manual":
        tempo_inicial = st.number_input("Tempo inicial (s):", float(df['X_Value'].min()), float(df['X_Value'].max()), float(df['X_Value'].min()))
        tempo_final = st.number_input("Tempo final (s):", float(df['X_Value'].min()), float(df['X_Value'].max()), float(df['X_Value'].max()))
        start_idx = df[df['X_Value'] >= tempo_inicial].index[0]
        end_idx = df[df['X_Value'] <= tempo_final].index[-1] + 1
    else:
        coluna_criterio = st.selectbox("Coluna critério para janela automática:", colunas_numericas)
        min_window = st.number_input("Tamanho mínimo da janela (s):", 1.0, float(df['X_Value'].max()), 10.0)
        max_window = st.number_input("Tamanho máximo da janela (s):", min_window, float(df['X_Value'].max()), 30.0)
        start_idx, end_idx = find_min_std_window(df, coluna_criterio, min_window, max_window)

    st.write(f"Janela selecionada: {df['X_Value'].iloc[start_idx]:.2f}s a {df['X_Value'].iloc[end_idx-1]:.2f}s")

    # Plot janela para cada coluna selecionada
    for col in colunas_selecionadas:
        fig = plot_window(df, col, start_idx, end_idx)
        st.pyplot(fig)

    # Estatísticas na janela
    st.subheader("Estatísticas na janela")
    medias, desvios, uAs = uncert_propagation(df, colunas_selecionadas, start_idx, end_idx)
    stats_df = pd.DataFrame({
        "Coluna": colunas_selecionadas,
        "Média": medias,
        "Desvio padrão": desvios,
        "Incerteza tipo A": uAs
    })
    st.dataframe(stats_df)

    # Cálculo e plot de Alpha (se disponível)
    st.subheader("Cálculo e plotagem de Alpha (fração de vazio)")
    I_g = st.number_input("Intensidade padrão para o gás (I_g):", value=252883)
    I_f = st.number_input("Intensidade padrão para o líquido (I_f):", value=151287)
    if st.button("Calcular e plotar Alpha"):
        alpha_df = calc_alpha(df, start_idx, end_idx, I_g, I_f)
        if alpha_df is not None:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(alpha_df['X_Value'], alpha_df['Alpha'], label='Alpha')
            ax.set_xlabel('Tempo (s)')
            ax.set_ylabel('Alpha')
            ax.set_title('Fração de vazio (Alpha)')
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)
            st.dataframe(alpha_df)
        else:
            st.warning("Coluna 'Densitometro' não encontrada no arquivo.")

    # Download dos resultados
    st.subheader("Download dos resultados")
    excel_buffer = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
    stats_df.to_excel(excel_buffer.name, index=False)
    with open(excel_buffer.name, "rb") as f:
        st.download_button("Baixar estatísticas em Excel", f, file_name="estatisticas_janela.xlsx")

else:
    st.warning("Faça upload de um arquivo para começar.") 