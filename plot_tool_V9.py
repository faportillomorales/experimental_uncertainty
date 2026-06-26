"""
Ferramentas de leitura de dados experimentais e geração de figuras (plot_tool V4).

O script segue um fluxo único: configurar ``file_path`` / ``NAS_file``, carregar o Excel,
padronizar colunas quando aplicável e gravar PDF/PNG por tipo de gráfico e por aba.
"""
from contextlib import redirect_stderr
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.legend import Legend
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np
import pandas as pd
from CoolProp.CoolProp import PropsSI

# --- Inputs globais (execução como script) ---
file_path = 'data_example/example/mean_sf6_v2/Mean_Experimental_Data_FSC2_SF6_Oil_v2.xlsx'  # Insira o caminho do arquivo a ser analisado NOTE: USE SEMPRE A BARRA NORMAL '/', SE ESTIVER INVERTIDA, MODIFIQUE-A

# Flag para indicar leitura de workbook NAS já processado (``processed_all_sheets_<nome>.xlsx``)
NAS_file = False

#insira o diâmetro da tubulação em metros
PIPE_DIAMETER_M = 0.05251

#Fix plot limits
FIX_PLOT_LIMITS = True
plot_limits_dpdz_f_y = [0, 1.5]
major_ticks_dpdz_f_y = 0.5      #Minor tick é automático (dividido em 5 partes)
plot_limits_dpdz_t_y = [-9,1]
major_ticks_dpdz_t_y = 1        #Minor tick é automático (dividido em 5 partes)

def dpdz_y_limits_for_plot(kind):
    """
    Limites Y [kPa/m] para plots de dp/dz. ``kind``: ``'f'`` (friccional) ou ``'t'`` (total).
    Retorna ``(bottom, top)`` ou ``(None, None)`` se ``FIX_PLOT_LIMITS`` for False.
    """
    if not FIX_PLOT_LIMITS:
        return None, None
    limits = plot_limits_dpdz_f_y if kind == 'f' else plot_limits_dpdz_t_y
    if not limits or len(limits) != 2:
        return None, None
    return float(limits[0]), float(limits[1])


def apply_fixed_dpdz_ylim(ax, kind):
    """Fixa o eixo Y conforme ``plot_limits_dpdz_*_y`` quando ``FIX_PLOT_LIMITS`` é True."""
    y0, y1 = dpdz_y_limits_for_plot(kind)
    if y0 is None and y1 is None:
        return
    kwargs = {}
    if y0 is not None:
        kwargs['bottom'] = y0
    if y1 is not None:
        kwargs['top'] = y1
    ax.set_ylim(**kwargs)


def dpdz_y_major_step_for_plot(kind):
    """
    Passo do major tick no eixo Y [kPa/m] quando ``FIX_PLOT_LIMITS`` é True.
    ``kind``: ``'f'`` (friccional) ou ``'t'`` (total). Caso contrário, ``None``.
    """
    if not FIX_PLOT_LIMITS:
        return None
    step = major_ticks_dpdz_f_y if kind == 'f' else major_ticks_dpdz_t_y
    try:
        s = float(step)
    except (TypeError, ValueError):
        return None
    return s if s > 0 else None


def apply_fixed_dpdz_y_axis(ax, kind):
    """Limites Y, major e minor ticks de dp/dz quando ``FIX_PLOT_LIMITS`` é True."""
    apply_fixed_dpdz_ylim(ax, kind)
    step = dpdz_y_major_step_for_plot(kind)
    if step is not None:
        ax.yaxis.set_major_locator(MultipleLocator(step))
    configure_dpdz_y_axis_minor_ticks(ax)


DP_DZ_Y_MINOR_TICK_WIDE_RANGE = 5.0
DP_DZ_Y_MINOR_TICK_WIDE = 0.2


def configure_dpdz_y_axis_minor_ticks(ax):
    """Minor ticks Y em plots dp/dz: 0,2 se ``ymax − ymin`` > 5."""
    y0, y1 = ax.get_ylim()
    if float(y1) - float(y0) > DP_DZ_Y_MINOR_TICK_WIDE_RANGE:
        ax.yaxis.set_minor_locator(MultipleLocator(DP_DZ_Y_MINOR_TICK_WIDE))


# Abas válidas para arquivos NAS (as demais serão ignoradas)
ALLOWED_SHEETS_NAS = {
    'AWH00', 'AWU05', 'AWU90',
    'AWD05', 'AWD15E', 'AWD30', 'AWD45', 'AWD60', 'AWD60E', 'AWD85', 'AWD90',
    'AOH00', 'AOU05', 'AOU90',
    'AOD05', 'AOD15E', 'AOD30', 'AOD45', 'AOD45E', 'AOD60', 'AOD60E', 'AOD85', 'AOD90',
    'SOH00', 'SOU05', 'SOU90',
    'SOD05', 'SOD15', 'SOD15E', 'SOD30', 'SOD45', 'SOD60', 'SOD85', 'SOD90',
    'ADH00', 'ADU05', 'ADU90',
    'ADD05', 'ADD15', 'ADD45', 'ADD60', 'ADD85', 'ADD90',
}

# Mapeamento de códigos curtos de flow pattern (formato NAS) para os nomes usados nos gráficos
NAS_FLOW_PATTERN_MAP = {
    'AN': 'Annular',
    'FF': 'Falling Film',
    'SS': 'Stratified Smooth',
    'SW': 'Stratified Wavy',
    'RW': 'Rolling Wave',
    'ST&MI': 'Stratified with Mixed Interface',
    'SL': 'Slug',
    'PSL': 'Pseudo-Slug',
    'CH': 'Churn',
    'DC': 'Dual-Continuous',
    'DB': 'Dispersed Bubbles',
    
    # Variantes com maiúscula/minúscula (normalizadas por strip().upper() na leitura)
    'ANNULAR': 'Annular',
    'FALLING FILM': 'Falling Film',
    'STRATIFIED SMOOTH': 'Stratified Smooth',
    'STRATIFIED WAVY': 'Stratified Wavy',
    'ROLLING WAVE': 'Rolling Wave',
    'STRATIFIED WITH MIXED INTERFACE': 'Stratified with Mixed Interface',
    'SLUG': 'Slug',
    'PSEUDO-SLUG': 'Pseudo-Slug',
    'CHURN': 'Churn',
    'DUAL-CONTINUOUS': 'Dual-Continuous',
    'DISPERSED BUBBLES': 'Dispersed Bubbles',
}

# Variantes de texto na planilha → chave de ``get_flow_pattern_symbols()``
_FLOW_PATTERN_CANONICAL_ALIASES = {
    'dispersed bubbles': 'Dispersed Bubbles',
    'pseudo-slug': 'Pseudo-Slug',
    'pseudo slug': 'Pseudo-Slug',
}


def canonical_flow_pattern_name(val):
    """Nome canónico para símbolo/cor/legenda (célula vazia → Unclassified)."""
    if _is_blank_excel_value(val):
        return FLOW_PATTERN_UNCLASSIFIED
    if isinstance(val, str) and val.strip() == '':
        return FLOW_PATTERN_UNCLASSIFIED
    s = str(val).strip()
    key = s.upper()
    if key in NAS_FLOW_PATTERN_MAP:
        name = NAS_FLOW_PATTERN_MAP[key]
    else:
        name = NAS_FLOW_PATTERN_MAP.get(key, s)
    alias = _FLOW_PATTERN_CANONICAL_ALIASES.get(str(name).strip().lower())
    if alias:
        return alias
    alias = _FLOW_PATTERN_CANONICAL_ALIASES.get(s.lower())
    if alias:
        return alias
    return str(name).strip()


def _build_flow_pattern_sigla_by_name():
    """Nome completo (gráficos) → sigla NAS (preferir códigos curtos do mapa)."""
    by_name = {}
    for sigla, name in NAS_FLOW_PATTERN_MAP.items():
        if len(sigla) > 6:
            continue
        cur = by_name.get(name)
        if cur is None or len(sigla) < len(cur):
            by_name[name] = sigla
    return by_name


FLOW_PATTERN_SIGLA_BY_NAME = _build_flow_pattern_sigla_by_name()


class LegendGridPlaceholder(Line2D):
    """
    Proxy só para identificação no ``handler_map``. Subclasse de Line2D para não colidir
    com patches usados em outros contextos.
    """

    def __init__(self):
        super().__init__(
            [],
            [],
            linestyle='none',
            marker='',
            linewidth=0,
            color=(1, 1, 1, 0),
            label=' ',
        )


class HandlerLegendGridPlaceholder(HandlerBase):
    """
    Desenha apenas uma linha com espessura 0 e cor RGBA (0,0,0,0); os backends AGG/PDF
    normalmente não rasterizam isso (ao contrário de Patch com ``edgecolor='none'``,
    que por vezes deixa um traço de 1 px por anti-aliasing).
    """

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        yc = (height - ydescent) * 0.5
        line = Line2D(
            (-xdescent, -xdescent + width),
            (yc, yc),
            linewidth=0,
            linestyle='solid',
            color=(0, 0, 0, 0),
            antialiased=False,
            solid_capstyle='butt',
        )
        line.set_transform(trans)
        return [line]


Legend.update_default_handler_map({
    LegendGridPlaceholder: HandlerLegendGridPlaceholder(),
})


# Suprimir avisos específicos do pandas
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')
warnings.filterwarnings('ignore', category=UserWarning, module='pandas')
# Suprimir avisos de timestamp ao salvar PDF/PNG (backend matplotlib/pdf)
warnings.filterwarnings('ignore', message=r".*timestamp seems very low.*")
warnings.filterwarnings('ignore', message=r".*regarding as unix timestamp.*")

# Legenda no topo; no máximo LEGEND_TOP_NCOL colunas; com menos entradas reduz-se ncol
LEGEND_TOP_KWARGS = {
    'loc': 'lower center',
    'bbox_to_anchor': (0.5, 1.02),
    'frameon': False,
    'fontsize': 20,
    'prop': {'family': 'serif'},
    'handler_map': {LegendGridPlaceholder: HandlerLegendGridPlaceholder()},
}
LEGEND_TOP_NCOL = 4
# Legenda de flow patterns nos plots de orientação (modo ``all``): ligeiramente menor que o padrão.
LEGEND_ORIENTATION_FLOW_PATTERN_FONTSIZE = 14
LEGEND_ORIENTATION_FLOW_PATTERN_MARKERSCALE = 0.75
# Duas legendas empilhadas: flow patterns mais perto do gráfico; J_l/Re_sl acima.
LEGEND_TOP_ANCHOR_Y_FLOW = 1.02
LEGEND_TOP_ANCHOR_Y_SERIES = 1.08


def legend_ncol_from_n_entries(n_entries, ncol_max=LEGEND_TOP_NCOL):
    """Colunas da legenda: no máximo ``ncol_max``; com menos entradas usa só as necessárias."""
    if n_entries <= 0:
        return 1
    return min(int(n_entries), int(ncol_max))


def legend_ncol_flow_patterns_only(n_patterns):
    """Segunda linha da legenda: uma coluna por padrão de escoamento (uma linha só)."""
    return max(int(n_patterns), 1)


def apply_two_row_top_legend(
    ax,
    head_handles,
    tail_handles,
    *,
    head_ncol_max=LEGEND_TOP_NCOL,
    tail_fontsize=None,
    tail_markerscale=1.0,
):
    """
    Duas legendas no topo, cada uma com ``ncol`` próprio — sem placeholders nem células vazias.
    Linha superior: condições (J_l, Re_sl, …); linha inferior: siglas de flow pattern.
    """
    head = list(head_handles or [])
    tail = list(tail_handles or [])
    base = {k: v for k, v in LEGEND_TOP_KWARGS.items() if k != 'bbox_to_anchor'}
    tail_base = dict(base)
    if tail_fontsize is not None:
        tail_base['fontsize'] = tail_fontsize
        tail_base['prop'] = {'family': 'serif', 'size': tail_fontsize}
    if tail_markerscale != 1.0:
        tail_base['markerscale'] = tail_markerscale

    if head and tail:
        leg_flow = ax.legend(
            handles=tail,
            ncol=legend_ncol_flow_patterns_only(len(tail)),
            bbox_to_anchor=(0.5, LEGEND_TOP_ANCHOR_Y_FLOW),
            **tail_base,
        )
        ax.add_artist(leg_flow)
        ax.legend(
            handles=head,
            ncol=legend_ncol_from_n_entries(len(head), ncol_max=head_ncol_max),
            bbox_to_anchor=(0.5, LEGEND_TOP_ANCHOR_Y_SERIES),
            **base,
        )
    elif head:
        ax.legend(
            handles=head,
            ncol=legend_ncol_from_n_entries(len(head), ncol_max=head_ncol_max),
            bbox_to_anchor=LEGEND_TOP_KWARGS['bbox_to_anchor'],
            **base,
        )
    elif tail:
        ax.legend(
            handles=tail,
            ncol=legend_ncol_flow_patterns_only(len(tail)),
            bbox_to_anchor=LEGEND_TOP_KWARGS['bbox_to_anchor'],
            **tail_base,
        )




def _safe_processed_filename_segment(name: str) -> str:
    """Parte de nome de ficheiro segura no Windows (substitui caracteres reservados)."""
    s = str(name).strip() if name else 'Sheet1'
    for c in '\\/:*?"<>|':
        s = s.replace(c, '_')
    return s or 'Sheet1'


def _base_strip_column_mapping(df):
    """
    Nome de coluna sem espaços externos → objeto coluna original no ``DataFrame``.
    Sem aliases experimentais (contrasta com ``build_column_mapping``).
    """
    return {str(col).strip(): col for col in df.columns if pd.notna(col)}


def _density_viscosity_liquid_25c(fluid_2):
    """
    Densidade [kg/m³] e viscosidade dinâmica [Pa·s] do líquido a 25 °C e 1 atm.
    Mesma lógica que ``standardize_liquid_conditions`` (Water / Oil / CoolProp / fallback).
    """
    temp_c = 25.0
    temp_k = temp_c + 273.15
    if fluid_2 == 'Water':
        rho = PropsSI('D', 'P', 101325, 'T', temp_k, 'Water')
        mu = PropsSI('V', 'P', 101325, 'T', temp_k, 'Water')
    elif fluid_2 == 'Oil':
        rho = 0.0008 * temp_c**2 - 0.698 * temp_c + 879.154
        mu_cp = 0.031267 * (temp_c**2) - 3.2050 * temp_c + 97.6594
        mu = mu_cp * 1e-3
    else:
        try:
            rho = PropsSI('D', 'P', 101325, 'T', temp_k, fluid_2)
            mu = PropsSI('V', 'P', 101325, 'T', temp_k, fluid_2)
        except Exception:
            rho = PropsSI('D', 'P', 101325, 'T', temp_k, 'Water')
            mu = PropsSI('V', 'P', 101325, 'T', temp_k, 'Water')
    return rho, mu


def _set_ticklabels_font_serif(ax):
    """Aplica fonte serif aos rótulos numéricos dos eixos."""
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily('serif')


def get_series_line_and_marker_styles():
    """
    Ciclo fixo de estilo de linha e símbolo de marcador para séries (jL, Re_sl, point_id, etc.).
    Mesma ordem em todos os plots para manter coerência visual.

    Returns
    -------
    tuple[tuple[str, ...], tuple[str, ...]]
        (estilos de linha matplotlib, símbolos de marcador nas linhas).
    """
    line_styles = (
        ':',
        '-.',
        '--',
        '-',
        ':',
        '-.',
        '--',
        '-',
        ':',
        '-.',
    )
    marker_symbols = ('^', 'v', 's', 'D', 'd', 'p', 'o', 'h', '8', '*', 'X')
    return line_styles, marker_symbols


def _jl_proximity_cluster_assignments(jl_raw: pd.Series, tolerance_abs: float):
    """
    Agrupa valores distintos de j_L por proximidade (greedy sobre únicos ordenados).
    Devolve, por linha: média do cluster e id inteiro do cluster (0..G-1, por média crescente).
    """
    idx = jl_raw.index
    arr = jl_raw.to_numpy(dtype=float)
    n = len(arr)
    u = np.sort(np.unique(arr[np.isfinite(arr)]))
    if u.size == 0:
        nan_s = pd.Series(np.nan, index=idx, dtype=float)
        return nan_s, nan_s.copy()

    groups = []
    cur = [float(u[0])]
    for i in range(1, len(u)):
        v = float(u[i])
        if abs(v - float(np.mean(cur))) <= tolerance_abs:
            cur.append(v)
        else:
            groups.append(cur)
            cur = [v]
    groups.append(cur)

    groups_with_mean = [(float(np.mean(g)), g) for g in groups]
    groups_with_mean.sort(key=lambda t: t[0])

    val_to_mean = {}
    val_to_gid = {}
    for gid, (m, g) in enumerate(groups_with_mean):
        for x in g:
            xf = float(x)
            val_to_mean[xf] = m
            val_to_gid[xf] = gid

    jl_mean_row = np.full(n, np.nan)
    jl_gid_row = np.full(n, np.nan)
    for i in range(n):
        x = arr[i]
        if not np.isfinite(x):
            continue
        xf = float(x)
        best_key = None
        best_d = np.inf
        for k in val_to_mean:
            d = abs(float(k) - xf)
            if d < best_d:
                best_d = d
                best_key = k
        if best_key is not None and np.isclose(best_key, xf, rtol=1e-9, atol=1e-7):
            jl_mean_row[i] = val_to_mean[best_key]
            jl_gid_row[i] = float(val_to_gid[best_key])

    return (
        pd.Series(jl_mean_row, index=idx),
        pd.Series(jl_gid_row, index=idx),
    )


def standardize_liquid_conditions(
    all_dataframes: dict,
    *,
    jl_tolerance=0.05,
    D=PIPE_DIAMETER_M,
):
    """
    Padroniza a condição do líquido para todas as inclinações/abas.

    - Agrupa os j_L medidos por proximidade (`jl_tolerance` em m/s), substitui a coluna
      jL pela média de cada grupo (para Reynolds e gráficos que usam jL pós-processado).
    - Mantém jL_raw com os valores da planilha (inalterados após a primeira cópia).
    - Usa propriedades termofísicas do fluido líquido (Water/Oil/SF6, etc.)
      em 25°C e 1 atm para calcular Re_sl.
    - Grava/atualiza colunas:
      - jL_raw: cópia do j_L medido antes da substituição por médias de cluster
      - Re_sl_raw: Re_sl com jL = média do cluster de proximidade
      - Re_sl_group: média de Re_sl_raw por jL_group_id (coerente com o cluster)
      - jL_group_id: id do cluster (ordenado por média de j_L crescente)
    """
    if not all_dataframes:
        return all_dataframes

    for sheet_name, df in all_dataframes.items():
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            continue

        # Encontrar coluna de jL (com tolerância a espaços); sem aliases — igual ao histórico
        col_mapping = _base_strip_column_mapping(df)
        if 'jL' not in col_mapping:
            print(f"Aviso: coluna 'jL' não encontrada em {sheet_name}; padronização de líquido ignorada.")
            continue

        jl_col = col_mapping['jL']
        # Valores medidos antes da substituição por médias de cluster (matrizes / paridade)
        df['jL_raw'] = pd.to_numeric(df[jl_col], errors='coerce')

        jl_cluster_mean, jl_gid = _jl_proximity_cluster_assignments(
            df['jL_raw'], jl_tolerance
        )
        df[jl_col] = jl_cluster_mean
        df['jL_group_id'] = jl_gid

        # Identificar fluido líquido a partir do nome da aba (mesma convenção de extract_info_from_filename)
        try:
            _, fluid_2, _, _, _, _ = extract_info_from_filename(sheet_name)
        except Exception:
            fluid_2 = 'Water'

        rho_L_fixed, mu_L_fixed = _density_viscosity_liquid_25c(fluid_2)

        # Re_sl com jL = média do cluster de proximidade (coluna jL já substituída acima)
        jl_numeric = pd.to_numeric(df[jl_col], errors='coerce')
        Re_sl_raw = rho_L_fixed * jl_numeric * D / mu_L_fixed
        df['Re_sl_raw'] = Re_sl_raw

        # Média de Re_sl por jL_group_id (coerente com o agrupamento por proximidade)
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
        'D': 'Dense Liquid'
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

        if df is None:
            print("Falha ao carregar a aba selecionada (dados não disponíveis).")
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
            theta = -theta
        
        return df, units, sheet_name, fluid_1, fluid_2, theta, False  # False indica que não foi escolhido 'all'
        
    except Exception as e:
        print(f"Erro inesperado: {e}")
        return None, None


def _is_blank_excel_value(x):
    """True se a célula não deve ser usada como dado (vazio, só espaços, NaN/NA/None)."""
    if x is None:
        return True
    if isinstance(x, str) and x.strip() == '':
        return True
    try:
        if pd.isna(x):
            return True
    except (TypeError, ValueError):
        pass
    return False


# Placeholders frequentes no Excel para “sem medição” (α, -dP/dz F, -dP/dz T)
_MEASUREMENT_PLACEHOLDER_STRINGS = frozenset(
    {
        '-',
        '—',
        '–',
        '−',  # minus Unicode
        '--',
        'n/a',
        'na',
        '#n/a',
        '#na',
        'none',
        'nan',
        'null',
    }
)


def _is_empty_measurement_cell(x):
    """
    True se a célula de α / dp/dz_F / dp/dz_T não deve ser tratada como dado numérico.
    Inclui vazio, NaN e strings-tipo (“-”, “N/A”, etc.).
    """
    if _is_blank_excel_value(x):
        return True
    if isinstance(x, str):
        t = x.strip()
        if not t:
            return True
        if t.lower() in _MEASUREMENT_PLACEHOLDER_STRINGS:
            return True
    return False


def normalize_excel_empty_cells(df):
    """
    Converte células em branco (incl. string vazia ou só espaços do Excel) em NaN.
    Assim gráficos e cálculos ignoram esses valores como já fazem com NaN.
    """
    if df is None:
        return None
    out = df.copy()
    # Por índice de coluna: com cabeçalhos duplicados, df[nome] é DataFrame e
    # blank.any() deixa de ser bool → "truth value of a Series is ambiguous".
    for j in range(out.shape[1]):
        ser = out.iloc[:, j]
        try:
            blank = ser.map(_is_blank_excel_value)
        except (TypeError, AttributeError):
            blank = ser.apply(_is_blank_excel_value)
        if bool(blank.any()):
            out.iloc[:, j] = ser.mask(blank).values
    return out


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

        df = normalize_excel_empty_cells(df)
        df = coerce_measurement_columns_to_nan(df)

        # Remover linhas vazias no início
        df = df.dropna(how='all')
        
        # Resetar o índice
        df = df.reset_index(drop=True)
        
        return df, units_dict
        
    except Exception as e:
        print(f"Erro ao ler aba {sheet_name}: {e}")
        return None, None


# NAS: ``B:Z`` pede colunas que muitas abas não têm → openpyxl/pandas rejeita usecols fora da
# folha (ex.: índices 24–25). Manter ``B:W`` como em ``plot_tool_V6`` (formato NAS original).
_NAS_SHEET_USECOLS = "B:W"


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
            nrows=8,
            usecols=_NAS_SHEET_USECOLS,
        )

        # Nomes das colunas na linha 4 (índice 3, 0-indexado)
        def _nas_clean(s):
            if not isinstance(s, str):
                return ""
            return s.strip().replace('\n', '').replace('\r', '').strip()

        def _looks_like_nas_header(row):
            cleaned = {_nas_clean(v).upper() for v in row if _nas_clean(v)}
            return (
                ('JG' in cleaned and 'JL' in cleaned)
                or ('FLOW PATTERN' in cleaned and any('DP/DZ' in c for c in cleaned))
            )

        header_idx = None
        for idx in range(len(df_header)):
            if _looks_like_nas_header(df_header.iloc[idx].tolist()):
                header_idx = idx
                break

        if header_idx is None:
            raise ValueError("linha de cabeçalho NAS não encontrada nas primeiras 8 linhas")

        units_idx = header_idx + 1
        if units_idx >= len(df_header):
            raise ValueError("linha de unidades NAS não encontrada após o cabeçalho")

        column_names = df_header.iloc[header_idx].tolist()

        # Unidades na linha 5 (índice 4)
        units = df_header.iloc[units_idx].tolist()

        _nas_ncols = df_header.shape[1]
        valid_col_positions = [
            i
            for i, col_name in enumerate(column_names)
            if not _is_blank_excel_value(col_name) and i < _nas_ncols
        ]
        column_names = [column_names[i] for i in valid_col_positions]
        units = [units[i] for i in valid_col_positions]

        # Criar dicionário de unidades
        units_dict = {}
        for col_name, unit in zip(column_names, units):
            if pd.notna(col_name) and pd.notna(unit):
                units_dict[col_name] = unit
            elif pd.notna(col_name):
                units_dict[col_name] = ""

        # Ler dados (16 linhas), mesma largura que o cabeçalho (B:W).
        df_raw = pd.read_excel(
            file_path,
            sheet_name=sheet_name,
            header=None,
            skiprows=units_idx + 1,
            nrows=16,     # 6–21
            usecols=_NAS_SHEET_USECOLS,
        )
        df = df_raw.iloc[:, valid_col_positions].copy()

        # Definir nomes das colunas
        df.columns = column_names

        df = normalize_excel_empty_cells(df)

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
                clean_upper = clean.upper()
                if clean_upper == 'JL':
                    rename_map[col] = 'jL'
                elif clean_upper == 'JG':
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
                elif clean_upper == 'JG':
                    rename_map[col] = 'jG'
                elif clean_upper.startswith('TEMP') or clean == 'T':
                    rename_map[col] = 'Temp.'
                elif clean_upper == 'PRESSURE' or (clean_upper.startswith('GAUGE') and 'P' in clean_upper):
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
                if _is_blank_excel_value(val):
                    return np.nan
                s = str(val).strip()
                if s == '':
                    return np.nan
                key = s.upper()
                return NAS_FLOW_PATTERN_MAP.get(key, NAS_FLOW_PATTERN_MAP.get(s, val))
            df[fp_col] = df[fp_col].apply(_nas_fp_to_full)

        df = coerce_measurement_columns_to_nan(df)

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
        'font.size': 22,
        'axes.labelsize': 26,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 19,
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


def apply_subtle_gray_grid(ax):
    """
    Grade cinza discreta por trás dos dados: linhas dos ticks principais um pouco
    mais escuras e visíveis que as dos ticks secundários.
    """
    ax.set_axisbelow(True)
    ax.grid(
        True,
        which='major',
        color='0.52',
        alpha=0.5,
        linestyle='-',
        linewidth=0.65,
        zorder=0,
    )
    ax.grid(
        True,
        which='minor',
        color='0.52',
        alpha=0.5,
        linestyle=':',
        linewidth=0.6,
        zorder=0,
    )


FLOW_PATTERN_UNCLASSIFIED = 'Unclassified'


def get_flow_pattern_symbols():
    """Retorna o dicionário de símbolos para Flow Patterns baseado nos dados atuais do Excel"""
    return {
        FLOW_PATTERN_UNCLASSIFIED: {'symbol': 'X', 'color': 'gray'},
        # Padrões encontrados no arquivo Excel atual (análise detalhada)
        'Annular': {'symbol': '^', 'color': 'green'},                            # Annular
        'Falling Film': {'symbol': 'v', 'color': 'lime'},                        # Falling Film
        'Stratified Smooth': {'symbol': 's', 'color': 'purple'},                 # Stratified Smooth
        'Stratified Wavy': {'symbol': 'D', 'color': 'darkorchid'},               # Stratified Wavy
        'Rolling Wave':  {'symbol': 'd', 'color': 'violet'},                     # Rolling Wave
        'Stratified with Mixed Interface': {'symbol': 'p', 'color': 'magenta'},  # ST&MI
        'Slug': {'symbol': 'o', 'color': 'red'},                                 # Slug
        'Pseudo-Slug': {'symbol': 'h', 'color': 'firebrick'},                    # Pseudo-slug
        'Churn': {'symbol': 'X', 'color': 'gold'},                               # Churn
        'Dual-Continuous': {'symbol': '*', 'color': 'black'},                    # Dual-continuous
        'Dispersed Bubbles': {'symbol': 'P', 'color': 'blue'},                   #Dispersed Bubbles
    }


def flow_pattern_display_label(val):
    """Rótulo para legenda/marcador; célula de flow pattern vazia → Unclassified."""
    return canonical_flow_pattern_name(val)


def flow_pattern_legend_label(val):
    """Sigla NAS para rótulo de legenda; célula vazia → Unclassified."""
    display = flow_pattern_display_label(val)
    if display == FLOW_PATTERN_UNCLASSIFIED:
        return FLOW_PATTERN_UNCLASSIFIED
    return FLOW_PATTERN_SIGLA_BY_NAME.get(display, display)


def style_for_flow_pattern_cell(val, flow_pattern_symbols=None):
    """Estilo (marker, color) para um valor de célula de Flow Pattern, incl. vazios."""
    if flow_pattern_symbols is None:
        flow_pattern_symbols = get_flow_pattern_symbols()
    label = flow_pattern_display_label(val)
    if label == FLOW_PATTERN_UNCLASSIFIED:
        return flow_pattern_symbols[FLOW_PATTERN_UNCLASSIFIED]
    return flow_pattern_symbols.get(
        label,
        {'symbol': 'o', 'color': 'gray'},
    )


def _one_col_series(df, col_name):
    """
    Retorna uma Series para df[col_name].
    Se o Excel tiver cabeçalhos duplicados, df[col_name] é um DataFrame; usa-se a 1ª coluna.
    """
    block = df[col_name]
    if isinstance(block, pd.DataFrame):
        return block.iloc[:, 0]
    return block


def _one_col_series_masked(df, mask, col_name):
    """Igual a _one_col_series para df.loc[mask, col_name]."""
    block = df.loc[mask, col_name]
    if isinstance(block, pd.DataFrame):
        return block.iloc[:, 0]
    return block


def _measured_jl_for_legend_series(df_plot, jl_col):
    """Valores de j_L da planilha: ``jL_raw`` se existir, senão a coluna canónica ``jl_col``."""
    if 'jL_raw' in df_plot.columns:
        return pd.to_numeric(_one_col_series(df_plot, 'jL_raw'), errors='coerce')
    return pd.to_numeric(_one_col_series(df_plot, jl_col), errors='coerce')


def _cluster_measured_jl_legend_labels(df_plot, jl_col, tolerance_abs=None):
    """
    Agrupa valores distintos de j_L medidos «próximos» (|v − média do grupo| ≤ tolerância),
    sobre valores únicos ordenados. Por linha define ``_jl_legend_group_mean``: média do grupo,
    arredondada a 2 casas decimais — sem usar níveis impostos pela padronização.
    """
    if tolerance_abs is None:
        tolerance_abs = JL_LEGEND_CLUSTER_TOLERANCE_ABS
    jl_meas = _measured_jl_for_legend_series(df_plot, jl_col)
    arr = jl_meas.to_numpy(dtype=float)
    u = np.sort(np.unique(arr[np.isfinite(arr)]))
    if u.size == 0:
        df_plot['_jl_legend_group_mean'] = np.nan
        return
    groups = []
    cur = [float(u[0])]
    for i in range(1, len(u)):
        v = float(u[i])
        if abs(v - float(np.mean(cur))) <= tolerance_abs:
            cur.append(v)
        else:
            groups.append(cur)
            cur = [v]
    groups.append(cur)
    val_to_label = {}
    for g in groups:
        lbl = round(float(np.mean(g)), 2)
        for x in g:
            val_to_label[float(x)] = lbl
    out = []
    for x in jl_meas:
        if not np.isfinite(x):
            out.append(np.nan)
            continue
        xf = float(x)
        lbl = None
        for k, lab in val_to_label.items():
            if abs(k - xf) <= 1e-9:
                lbl = lab
                break
        if lbl is None:
            lbl = round(xf, 2)
        out.append(lbl)
    df_plot['_jl_legend_group_mean'] = out


def _assign_numeric_to_first_named_column(df, col_name, numeric_series):
    """Escreve numeric_series na primeira coluna de df cujo nome é col_name (evita duplicados)."""
    idxs = [i for i, c in enumerate(df.columns) if c == col_name]
    if idxs:
        df.iloc[:, idxs[0]] = pd.to_numeric(numeric_series, errors='coerce').values
    else:
        df[col_name] = pd.to_numeric(numeric_series, errors='coerce')


# kwargs comuns para savefig dos plots por aba
PLOT_SAVEFIG_KWARGS = {
    'dpi': 300,
    'bbox_inches': 'tight',
    'pad_inches': 0.1,
    'facecolor': 'white',
    'edgecolor': 'none',
}

# Proporção comum a todos os gráficos (largura : altura = 12 : 9)
PLOT_FIGSIZE = (11, 8)

# Tolerância para agrupar j_L medidos próximos nas legendas [m/s] (sem níveis impostos)
JL_LEGEND_CLUSTER_TOLERANCE_ABS = 0.05

# Fallback de limites log (matriz j_L vs j_G) só quando não há dados positivos finitos
JL_JG_FLOW_MATRIX_LOG_FALLBACK = (1e-2, 1e2)

# Se o máximo de j_L nos dados for inferior a este valor [m/s], o limite superior do eixo Y
# (escala log) não fica abaixo deste valor — evita eixo Y demasiado “curto” na matriz log.
JL_JG_FLOW_MATRIX_LOG_Y_CAP_MIN = 3.0

# Margem relativa ao máximo antes de subir ao próximo tick «nice» (1–2–5–10) no modo compacto
LOG_AXIS_COMPACT_MARGIN_HI_REL = 0.03

# Lado de cada painel do mosaico (polegadas), para caixas de eixos quadradas
MOSAIC_JL_JG_PANEL_IN = 4.25

# Eixo Y — gradiente friccional / total [kPa/m] (mesmo padrão em todos os plots).
# Matplotlib mathtext não suporta \Big; \left/\right são equivalentes e válidos.
YLABEL_DP_DZ_F = (
    r'$- \left(\frac{\text{dP}}{\text{dz}}\right)_\text{f} \; \left[\frac{\mathrm{kPa}}{\mathrm{m}}\right]$'
)
YLABEL_DP_DZ_T = (
    r'$- \left(\frac{\text{dP}}{\text{dz}}\right)_\text{t} \; \left[\frac{\mathrm{kPa}}{\mathrm{m}}\right]$'
)

ALPHA_Y_AXIS_MAJOR_STEP = 0.1
ALPHA_Y_AXIS_MINOR_STEP = 0.05


def configure_alpha_y_axis_ticks(ax, *, major_step=None):
    """Eixo Y de α ∈ [0, 1]: major (default 0,1) e minor (0,05) padronizados."""
    if major_step is None:
        major_step = ALPHA_Y_AXIS_MAJOR_STEP
    ax.yaxis.set_major_locator(MultipleLocator(major_step))
    ax.yaxis.set_minor_locator(MultipleLocator(ALPHA_Y_AXIS_MINOR_STEP))


def _apply_mean_experimental_column_aliases(col_mapping):
    """
    Planilhas Mean_Experimental_Data_* (saída estilo exp_unc) usam -dpdz_F / -dpdz_T
    em vez de dp/dz_F / dp/dz_T. Garante chaves canónicas para os plots.
    """
    if not col_mapping:
        return
    for canonical, prefixes in (
        ('dp/dz_F', ('-dpdz_F', '-dpdz_f')),
        ('dp/dz_T', ('-dpdz_T', '-dpdz_t')),
    ):
        if canonical in col_mapping:
            continue
        for p in prefixes:
            if p in col_mapping:
                col_mapping[canonical] = col_mapping[p]
                break
        if canonical in col_mapping:
            continue
        for key in col_mapping:
            if key.startswith(prefixes[0]) or key.startswith(prefixes[1]):
                col_mapping[canonical] = col_mapping[key]
                break


def build_column_mapping(df):
    """Mapeia nome de coluna normalizado (strip) → coluna original do DataFrame."""
    col_mapping = _base_strip_column_mapping(df)
    _apply_mean_experimental_column_aliases(col_mapping)
    return col_mapping


def missing_column_keys(col_mapping, required_keys):
    """Lista de chaves em required_keys que não existem em col_mapping."""
    return [k for k in required_keys if k not in col_mapping]


def ensure_alpha_in_column_mapping(col_mapping, columns_iterable):
    """
    Garante col_mapping['α'] a partir de nomes alternativos (NAS / Excel).
    Retorna True se encontrou; caso contrário False.
    """
    for name in ('α', 'Alpha', 'alpha', 'Void fraction', 'Void Fraction', 'void fraction'):
        if name in col_mapping:
            if name != 'α':
                col_mapping['α'] = col_mapping[name]
            return True
    for col in columns_iterable:
        if col is None or pd.isna(col):
            continue
        c = str(col).strip().lower()
        if c in ('α', 'alpha', 'void fraction') or c.replace(' ', '') == 'voidfraction':
            key = str(col).strip()
            col_mapping['α'] = col_mapping[key]
            return True
    return False


# Colunas de incerteza experimental nas planilhas (Mean / exp_unc); ordem = preferência.
_UNCERTAINTY_COLUMN_CANDIDATES = {
    'U_alpha': (
        'U(?)',
        # Mean / exp_unc
        'U(alpha)',
        'U(α)',
        'U(Alpha)',
        'u(alpha)',
        'U (alpha)',
        # NAS: U seguido de α grego (sem parênteses)
        'Uα',
        'uα',
    ),
    'U_dpdz_F': (
        'U(-dP/dz F)',
        'U(-dP/dz f)',
        'U(-dpdz_F)',
        'U(-dpdz_f)',
    ),
    'U_dpdz_T': (
        'U(-dP/dz T)',
        'U(-dP/dz t)',
        'U(-dpdz_T)',
        'U(-dpdz_t)',
    ),
}

_UNCERTAINTY_TEXT_BBOX = {
    'boxstyle': 'round,pad=0.38',
    'facecolor': 'white',
    'alpha': 0.88,
    'edgecolor': '0.72',
    'linewidth': 0.7,
}


def _add_theta_annotation_box(ax, theta, *, fontsize=17):
    """Caixa com $\\theta$ no canto inferior direito (mesmo estilo que paridade / incerteza)."""
    ax.text(
        0.97,
        0.03,
        rf'$\theta = {theta}^\circ$',
        transform=ax.transAxes,
        fontsize=fontsize,
        fontfamily='serif',
        ha='right',
        va='bottom',
        multialignment='center',
        zorder=6,
        bbox=_UNCERTAINTY_TEXT_BBOX,
        linespacing=1.22,
    )


def _normalize_col_name(s):
    return str(s).strip().lower().replace(' ', '')


def _filter_dataframe_by_theta(df, theta_deg):
    """
    Se existir coluna de inclinação nos dados, restringe às linhas com esse θ;
    caso contrário devolve ``df`` (uma aba = uma inclinação).
    """
    if df is None or getattr(df, 'empty', True):
        return df
    theta_col = None
    for col in df.columns:
        cl = str(col).strip().lower()
        if cl in ('theta', 'θ', 'inclination', 'angle (deg)', 'angle', 'tilt'):
            theta_col = col
            break
    if theta_col is None:
        return df
    t = pd.to_numeric(df[theta_col], errors='coerce')
    mask = np.isfinite(t) & np.isclose(t, float(theta_deg), atol=0.55)
    if not mask.any():
        return df
    return df.loc[mask]


def _matches_u_alpha_column_header(name) -> bool:
    """
    Reconhece incerteza de α nas convenções:
    Mean — ``U(α)``, ``U(alpha)``; NAS — ``Uα`` (U + letra grega α U+03B1).
    """
    sc = str(name).strip().casefold().replace(' ', '')
    α = '\u03b1'
    if sc in (f'u{α}', f'u({α})', 'u(alpha)'):
        return True
    return False


def _find_uncertainty_actual_column(df, col_mapping, logical_key):
    """Resolve coluna original no ``df`` para U(α), U(-dP/dz F) ou U(-dP/dz T)."""
    candidates = _UNCERTAINTY_COLUMN_CANDIDATES.get(logical_key, ())
    for c in candidates:
        if c in col_mapping:
            return col_mapping[c]
    norm_to_key = {_normalize_col_name(k): k for k in col_mapping}
    for c in candidates:
        nk = _normalize_col_name(c)
        if nk in norm_to_key:
            return col_mapping[norm_to_key[nk]]
    # Fallback: igualdade nas colunas do DataFrame
    for c in candidates:
        nk = _normalize_col_name(c)
        for col in df.columns:
            if _normalize_col_name(col) == nk:
                return col
    if logical_key == 'U_alpha':
        for col in df.columns:
            if _matches_u_alpha_column_header(col):
                return col
    return None


def _find_point_uncertainty_column(df, col_mapping, logical_key):
    """Resolve a uncertainty column that starts with U( for the plotted y property."""
    actual = _find_uncertainty_actual_column(df, col_mapping, logical_key)
    if actual is not None:
        return actual

    expected_tokens = {
        'U_alpha': ('?', 'alpha', 'α', 'voidfraction'),
        'U_dpdz_F': ('dpdzf', 'dp/dzf', 'dp/dz_f', '-dpdz_f'),
        'U_dpdz_T': ('dpdzt', 'dp/dzt', 'dp/dz_t', '-dpdz_t'),
    }.get(logical_key, ())

    for col in df.columns:
        raw = str(col).strip().replace('\n', '').replace('\r', '')
        compact = raw.casefold().replace(' ', '').replace('-', '')
        if not raw.casefold().startswith('u('):
            continue
        if any(tok.casefold().replace(' ', '').replace('-', '') in compact for tok in expected_tokens):
            return col
    return None


def _indices_for_lowest_jl_series(df_plot, jl_col, x_series, y_series):
    """Índices de todos os pontos da série de menor j_L (``_jl_legend_group_mean``)."""
    if '_jl_legend_group_mean' not in df_plot.columns:
        _cluster_measured_jl_legend_labels(df_plot, jl_col)
    grp = pd.to_numeric(df_plot['_jl_legend_group_mean'], errors='coerce')
    x = pd.to_numeric(x_series, errors='coerce')
    y = pd.to_numeric(y_series, errors='coerce')
    valid = grp.notna() & x.notna() & y.notna()
    if not bool(valid.any()):
        return []
    min_grp = float(grp[valid].min())
    sel = valid & np.isclose(grp, min_grp, rtol=0, atol=1e-6)
    return list(df_plot.index[sel])


def _indices_for_lowest_resl_series(df_plot, x_series, y_series):
    """Índices de todos os pontos da série de menor ``Re_sl_group``."""
    if 'Re_sl_group' not in df_plot.columns:
        return []
    re = pd.to_numeric(df_plot['Re_sl_group'], errors='coerce')
    x = pd.to_numeric(x_series, errors='coerce')
    y = pd.to_numeric(y_series, errors='coerce')
    valid = re.notna() & x.notna() & y.notna()
    if not bool(valid.any()):
        return []
    min_re = float(re[valid].min())
    sel = valid & np.isclose(re, min_re, rtol=0, atol=0.5)
    return list(df_plot.index[sel])


def _selected_lowest_condition_indices(selector_series, x_series, y_series):
    """Retrocompat.: preferir ``_indices_for_lowest_jl_series`` / ``_indices_for_lowest_resl_series``."""
    selector = pd.to_numeric(selector_series, errors='coerce')
    x = pd.to_numeric(x_series, errors='coerce')
    y = pd.to_numeric(y_series, errors='coerce')
    valid = selector.notna() & x.notna() & y.notna()
    if not bool(valid.any()):
        return []
    min_val = float(selector[valid].min())
    in_min = valid & np.isclose(selector, min_val, rtol=0, atol=1e-6)
    return list(selector.index[in_min])


def _selected_highest_indices(selector_series, x_series, y_series, n=4):
    """Índices dos n pontos válidos com maior valor do selector (ordem estável)."""
    selector = pd.to_numeric(selector_series, errors='coerce')
    x = pd.to_numeric(x_series, errors='coerce')
    y = pd.to_numeric(y_series, errors='coerce')
    valid = selector.notna() & x.notna() & y.notna()
    if not bool(valid.any()):
        return []
    return list(selector[valid].sort_values(kind='mergesort', ascending=False).head(n).index)


def _plot_selected_y_uncertainty(ax, df_plot, indices, x_col, y_col, y_unc_col, y_scale=1.0):
    """Draw vertical uncertainty bars only on selected points."""
    if not indices or y_unc_col is None:
        return 0

    x = pd.to_numeric(_one_col_series(df_plot, x_col), errors='coerce')
    y = pd.to_numeric(_one_col_series(df_plot, y_col), errors='coerce') * y_scale
    yerr = pd.to_numeric(_one_col_series(df_plot, y_unc_col), errors='coerce').abs() * abs(y_scale)

    idx = [
        i for i in indices
        if i in df_plot.index and pd.notna(x.loc[i]) and pd.notna(y.loc[i]) and pd.notna(yerr.loc[i])
    ]
    if not idx:
        return 0

    ax.errorbar(
        x.loc[idx],
        y.loc[idx],
        yerr=yerr.loc[idx],
        fmt='none',
        ecolor='black',
        elinewidth=1.2,
        capsize=4,
        capthick=1.2,
        alpha=0.85,
        zorder=3,
    )
    return len(idx)


def _format_uncertainty_mean_display(mean_val, *, kind):
    """Médias de incerteza nos gráficos: α e gradiente (kPa/m a partir de Pa/m) com 3 casas decimais."""
    if not np.isfinite(mean_val):
        return None
    if kind == 'alpha':
        return f'{mean_val:.3f}'
    if kind == 'kpa_per_m':
        return f'{mean_val / 1000.0:.3f}'
    return f'{mean_val:.3f}'


def add_theta_and_mean_uncertainty_text(
    ax,
    df,
    theta,
    col_mapping,
    *,
    uncertainty_key,
    unit_latex=None,
    placement='bottom',
    fontsize=17,
    kind='alpha',
):
    """
    Quadro nos cantos do eixo (inferior direito ou superior esquerdo): θ na primeira linha;
    na segunda, ``$U_{\\mathrm{exp}} = \\pm ...$`` quando disponível.
    O texto multi-linha é centralizado **dentro** da caixa via ``multialignment``.
    ``placement``: ``'bottom'`` — canto inferior direito; ``'top'`` — canto superior esquerdo;
    ``'top_right'`` — canto superior direito.
    """
    lines = [rf'$\theta = {theta}^\circ$']
    try:
        sub = _filter_dataframe_by_theta(df, theta)
        if sub is not None and not getattr(sub, 'empty', True):
            actual = _find_uncertainty_actual_column(sub, col_mapping, uncertainty_key)
            if actual is not None:
                ser = pd.to_numeric(_one_col_series(sub, actual), errors='coerce').dropna()
                if not ser.empty:
                    mean_u = float(ser.mean())
                    if np.isfinite(mean_u):
                        val_str = _format_uncertainty_mean_display(mean_u, kind=kind)
                        if val_str is not None:
                            if unit_latex:
                                lines.append(
                                    rf'$U_{{\mathrm{{exp}}}} = \pm {val_str}\,{unit_latex}$'
                                )
                            else:
                                lines.append(rf'$U_{{\mathrm{{exp}}}} = \pm {val_str}$')
    except Exception:
        pass
    txt = '\n'.join(lines)
    if placement == 'top':
        xy = (0.03, 0.92)
        ha = 'left'
        va = 'top'
    elif placement == 'top_right':
        xy = (0.97, 0.92)
        ha = 'right'
        va = 'top'
    elif placement == 'top_left':
        xy = (0.03, 0.92)
        ha = 'left'
        va = 'top'
    elif placement == 'bottom_right':
        xy = (0.97, 0.03)
        ha = 'right'
        va = 'bottom'
    elif placement == 'bottom_left':
        xy = (0.03, 0.03)
        ha = 'left'
        va = 'bottom'
    else:
        xy = (0.97, 0.03)
        ha = 'right'
        va = 'bottom'
    ax.text(
        xy[0],
        xy[1],
        txt,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontfamily='serif',
        ha=ha,
        va=va,
        multialignment='center',
        zorder=6,
        bbox=_UNCERTAINTY_TEXT_BBOX,
        linespacing=1.22,
    )


def coerce_measurement_columns_to_nan(df):
    """
    Força NaN em células vazias ou placeholder nas colunas α, dp/dz_F e dp/dz_T
    (nomes canónicos ou equivalentes do Excel/NAS). Assim gráficos e resumos ignoram esses pontos.
    """
    if df is None or df.empty:
        return df
    col_mapping = build_column_mapping(df)
    ensure_alpha_in_column_mapping(col_mapping, df.columns)

    aliases_by_quantity = {
        'α': (
            'α',
            'Alpha',
            'alpha',
            'Void fraction',
            'Void Fraction',
            'void fraction',
        ),
        'dp/dz_F': ('dp/dz_F', '-dP/dz F', '-dP/dz f', '-dpdz_F', '-dpdz_f'),
        'dp/dz_T': ('dp/dz_T', '-dP/dz T', '-dP/dz t', '-dpdz_T', '-dpdz_t'),
    }

    for _logical, aliases in aliases_by_quantity.items():
        actual_col = None
        for a in aliases:
            if a in col_mapping:
                actual_col = col_mapping[a]
                break
        if actual_col is None:
            continue
        ser = _one_col_series(df, actual_col)
        mask_invalid = ser.map(_is_empty_measurement_cell)
        numeric = pd.to_numeric(ser.mask(mask_invalid), errors='coerce')
        _assign_numeric_to_first_named_column(df, actual_col, numeric)

    return df


def _resolve_column_from_mapping(col_mapping, *names):
    """Resolve coluna original a partir de candidatos (match strip, case-insensitive)."""
    if not col_mapping:
        return None
    norm_map = {str(k).strip().lower(): v for k, v in col_mapping.items()}
    for name in names:
        key = str(name).strip().lower()
        if key in norm_map:
            return norm_map[key]
    compact = {
        str(k).strip().lower().replace('_', '').replace(' ', ''): v
        for k, v in col_mapping.items()
    }
    for name in names:
        key = str(name).strip().lower().replace('_', '').replace(' ', '')
        if key in compact:
            return compact[key]
    return None


def _unit_for_column(units_dict, col_obj):
    """Unidade declarada na linha 4 do Excel para a coluna ``col_obj``."""
    if not units_dict:
        return ''
    target = str(col_obj).strip().lower()
    for k, u in units_dict.items():
        if str(k).strip().lower() == target:
            return '' if u is None or (isinstance(u, float) and pd.isna(u)) else str(u)
    return ''


def _absolute_pressure_pa_from_gauge(gauge_values, unit_text=''):
    """
    Converte pressão manométrica para absoluta [Pa].
    Mean/exp_unc: bar gauge → ``(P+1)*1e5``; kPa ou Pa gauge conforme unidade na planilha.
    """
    g = pd.to_numeric(gauge_values, errors='coerce')
    u = (unit_text or '').strip().lower()
    if 'bar' in u:
        return (g + 1.0) * 1e5
    if 'kpa' in u:
        return g * 1000.0 + 101325.0
    return g + 101325.0


def _gas_density_viscosity_arrays(P_Pa, T_K, fluid_1):
    """
    Densidade e viscosidade do gás (CoolProp) linha a linha.
    Não chama PropsSI se P ou T não forem finitos; nesses índices devolve NaN.
    """
    p_arr = np.asarray(pd.Series(P_Pa).to_numpy(), dtype=float)
    t_arr = np.asarray(pd.Series(T_K).to_numpy(), dtype=float)
    n = p_arr.shape[0]
    rho_G = np.full(n, np.nan, dtype=float)
    mu_G = np.full(n, np.nan, dtype=float)
    for i in range(n):
        p, t = p_arr[i], t_arr[i]
        if not np.isfinite(p) or not np.isfinite(t):
            continue
        try:
            rho = PropsSI('D', 'P', p, 'T', t, fluid_1)
            mu = PropsSI('V', 'P', p, 'T', t, fluid_1)
            if np.isfinite(rho) and rho > 0:
                rho_G[i] = rho
            if np.isfinite(mu) and mu > 0:
                mu_G[i] = mu
        except Exception:
            continue
    return rho_G, mu_G


def compute_Re_sg_column(df, col_mapping, fluid_1, D=PIPE_DIAMETER_M, units_dict=None):
    """
    Calcula ``Re_sg = ρ_g j_g D / μ_g`` ponto a ponto e grava em ``df['Re_sg']``.

    Planilhas Mean: usa ``Rho_gas`` e ``Mu_gas`` quando válidos (> 0). Se μ_g (ou ρ_g)
    faltar ou for ≤ 0 numa linha (ex. SOU90 com Mu_gas = 0 na condição de maior j_L),
    recalcula só essa propriedade com CoolProp (P, T) — evita série ausente em *_vs_Re_g.
    Sem colunas de propriedades do gás, usa CoolProp em todas as linhas (ex.: NAS).
    """
    jg_vals = pd.to_numeric(_one_col_series(df, col_mapping['jG']), errors='coerce').to_numpy()
    n = len(jg_vals)
    rho = np.full(n, np.nan, dtype=float)
    mu = np.full(n, np.nan, dtype=float)

    rho_col = _resolve_column_from_mapping(col_mapping, 'Rho_gas', 'rho_g', 'Rho_g')
    mu_col = _resolve_column_from_mapping(col_mapping, 'Mu_gas', 'mu_g', 'Mu_g')
    if rho_col is not None:
        rho = pd.to_numeric(_one_col_series(df, rho_col), errors='coerce').to_numpy(dtype=float)
    if mu_col is not None:
        mu = pd.to_numeric(_one_col_series(df, mu_col), errors='coerce').to_numpy(dtype=float)

    P_col = col_mapping.get('Gauge Pressure')
    T_col = col_mapping.get('Temp.')
    if P_col is not None and T_col is not None:
        unit_p = _unit_for_column(units_dict, P_col)
        P_Pa = _absolute_pressure_pa_from_gauge(_one_col_series(df, P_col), unit_p)
        T_K = pd.to_numeric(_one_col_series(df, T_col), errors='coerce').to_numpy(dtype=float) + 273.15
        rho_cp, mu_cp = _gas_density_viscosity_arrays(P_Pa, T_K, fluid_1)
        need_rho = ~np.isfinite(rho) | (rho <= 0)
        need_mu = ~np.isfinite(mu) | (mu <= 0)
        if rho_col is None:
            need_rho = np.isfinite(T_K) & np.isfinite(P_Pa)
        if mu_col is None:
            need_mu = np.isfinite(T_K) & np.isfinite(P_Pa)
        rho = np.where(
            need_rho & np.isfinite(rho_cp) & (rho_cp > 0),
            rho_cp,
            rho,
        )
        mu = np.where(
            need_mu & np.isfinite(mu_cp) & (mu_cp > 0),
            mu_cp,
            mu,
        )

    mu = np.where(np.isfinite(mu) & (mu > 0), mu, np.nan)
    rho = np.where(np.isfinite(rho) & (rho > 0), rho, np.nan)
    df['Re_sg'] = pd.Series(rho * jg_vals * D / mu, index=df.index)


def save_figure_to_sheet_dir(base_name, sheet_name):
    """Guarda PDF e PNG em dirname(file_path)/sheet_name/."""
    sheet_dir = os.path.join(os.path.dirname(file_path), sheet_name)
    os.makedirs(sheet_dir, exist_ok=True)
    pdf_file = os.path.join(sheet_dir, f'{base_name}.pdf')
    png_file = os.path.join(sheet_dir, f'{base_name}.png')
    plt.savefig(pdf_file, format='pdf', **PLOT_SAVEFIG_KWARGS)
    print(f'Plot PDF salvo: {pdf_file}')
    plt.savefig(png_file, format='png', **PLOT_SAVEFIG_KWARGS)
    print(f'Plot PNG salvo: {png_file}')


def flow_pattern_legend_handles(df_plot, flow_pattern_col, flow_pattern_symbols=None):
    """Handles matplotlib para legenda dos flow patterns presentes em df_plot."""
    if flow_pattern_symbols is None:
        flow_pattern_symbols = get_flow_pattern_symbols()
    used = set()
    for val in _one_col_series(df_plot, flow_pattern_col):
        used.add(flow_pattern_display_label(val))
    handles = []
    for pattern in sorted(used, key=str):
        pd_ = style_for_flow_pattern_cell(pattern, flow_pattern_symbols)
        symbol, color = pd_['symbol'], pd_['color']
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker=symbol,
                color='w',
                markerfacecolor=color,
                markersize=8,
                markeredgecolor=color,
                markeredgewidth=1,
                label=flow_pattern_legend_label(pattern),
            )
        )
    return handles


def _x_values_from_ax_artists(ax, *, positive_only=False):
    """Recolhe coordenadas X de linhas e scatters no eixo (apenas finitas)."""
    xs = []
    for line in ax.get_lines():
        xd = np.asarray(line.get_xdata(), dtype=float)
        m = np.isfinite(xd)
        if positive_only:
            m = m & (xd > 0)
        else:
            m = m & (xd >= 0)
        xd = xd[m]
        if xd.size:
            xs.extend(xd.tolist())
    for coll in ax.collections:
        off = coll.get_offsets()
        if len(off):
            xcol = np.asarray(off[:, 0], dtype=float)
            m = np.isfinite(xcol)
            if positive_only:
                m = m & (xcol > 0)
            else:
                m = m & (xcol >= 0)
            xcol = xcol[m]
            if xcol.size:
                xs.extend(xcol.tolist())
    return xs


def finalize_jg_plot_xlim(ax):
    """
    Eixo X (j_g): origem em 0; limite superior = próxima unidade inteira acima de j_g máximo
    (teto em 1 m/s, e se o máximo coincidir com um inteiro, usa o inteiro seguinte).
    Ex.: j_g_max = 1,2 → limite 2; j_g_max = 2 → limite 3; j_g_max = 0,8 → limite 1.

    Caso especial: se ``1 < j_{g,\\max} < 1{,}5`` m/s (máximo estritamente entre 1 e 1,5),
    impõe-se ``x_{\\max} = 1{,}5`` (faixa típica abaixo de 2 m/s porém acima de 1 m/s).
    """
    xs = _x_values_from_ax_artists(ax, positive_only=False)
    if not xs:
        ax.set_xlim(0.0, 1.0)
        return
    jg_max = float(max(xs))
    if jg_max > 1.0 and jg_max < 1.5:
        ax.set_xlim(0.0, 1.5)
        return
    x_hi = int(np.ceil(jg_max - 1e-12))
    if x_hi <= jg_max:
        x_hi += 1
    ax.set_xlim(0.0, float(x_hi))


def configure_linear_jg_axis_tick_locators(ax, jg_values):
    """
    Eixo X em j_g (m/s), escala linear. Se o maior j_g for positivo e ≤ 2 (inclui máximos
    abaixo de 1 m/s e entre 1 e 2 m/s), major = 0,5 e minor = 0,25; caso contrário
    major = 1 e minor = 0,5.
    """
    arr = np.asarray(jg_values, dtype=float)
    arr = arr[np.isfinite(arr)]
    jg_max = float(np.max(arr)) if arr.size > 0 else float('nan')
    if np.isfinite(jg_max) and jg_max > 0 and jg_max <= 2.0:
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(MultipleLocator(0.25))
    else:
        ax.xaxis.set_major_locator(MultipleLocator(1.0))
        ax.xaxis.set_minor_locator(MultipleLocator(0.5))


def _re_sg_log_x_limits_from_artists(ax, *, fallback=(1000.0, 250000.0)):
    """
    Limites do eixo X em escala log(Re_sg): mesma lógica compacta que
    ``_log_axis_compact_limits`` (inferior = potência de 10 abaixo do mínimo;
    superior ponderado em relação à década), só com dados > 0.
    """
    xs = _x_values_from_ax_artists(ax, positive_only=True)
    if not xs:
        return fallback
    rmin, rmax = min(xs), max(xs)
    if rmin <= 0 or rmax <= 0:
        return fallback
    return _log_axis_compact_limits(xs, fallback=fallback)


def style_axes_re_g_log_x(
    ax,
    *,
    y_major_step=0.5,
    y_lim_bottom=None,
    y_lim_top=None,
):
    """
    Eixo X em log(Re_sg): limites compactos (``_log_axis_compact_limits``): inferior =
    potência de 10 abaixo do mínimo; superior conforme proximidade à próxima década.
    Só pontos com Re_sg > 0; sem dados mantém o fallback 10³–2,5×10⁵. Eixo Y linear com
    rótulos numéricos a 1 casa decimal.

    y_lim_bottom / y_lim_top: None = não alterar esse limite (mantém autoscale após o plot).
    Para dp/dz total, não fixar y inferior: valores podem ser negativos (ex. tubos inclinados).
    y_major_step: None = localizador major padrão do matplotlib (melhor para grandes intervalos em Y).
    """
    ax.set_axisbelow(True)
    ax.minorticks_on()
    if y_major_step is not None:
        ax.yaxis.set_major_locator(MultipleLocator(y_major_step))
    if y_lim_bottom is not None or y_lim_top is not None:
        y0, y1 = ax.get_ylim()
        if y_lim_bottom is not None:
            y0 = y_lim_bottom
        if y_lim_top is not None:
            y1 = y_lim_top
        ax.set_ylim(y0, y1)
    ax.set_xscale('log')
    ax.yaxis.set_major_formatter(FuncFormatter(_axis_tick_decimal_linear))
    x_lo, x_hi = _re_sg_log_x_limits_from_artists(ax)
    ax.set_xlim(x_lo, x_hi)
    ax.tick_params(axis='both', which='major', labelsize=20, size=8)
    ax.tick_params(axis='both', which='minor', labelsize=20, size=6)
    _set_ticklabels_font_serif(ax)
    apply_subtle_gray_grid(ax)


def round_re_sl_for_display(re_sl):
    """
    Arredonda Re_sl à centena (igual ao rótulo $Re_{sl} \\approx$ nos plots de orientação).
    """
    try:
        x = float(re_sl)
    except (TypeError, ValueError):
        return 0
    if not np.isfinite(x):
        return 0
    return int(round(x / 100.0) * 100)


def _mask_re_sl_group(series, re_l):
    """Máscara booleana para pontos da série ``Re_sl_group`` (tolerância em Re_sl)."""
    re = pd.to_numeric(series, errors='coerce')
    return np.isclose(re, float(re_l), rtol=0, atol=0.5)


def re_sl_legend_handles_from_meta(legend_series_meta):
    """legend_series_meta: sequência de (re_l, linestyle)."""
    out = []
    for re_l, line_style in legend_series_meta:
        r_disp = round_re_sl_for_display(re_l)
        out.append(
            plt.Line2D(
                [0],
                [0],
                color='black',
                linestyle=line_style,
                linewidth=1.5,
                label=rf'$Re_{{sl}} \approx {r_disp}$',
            )
        )
    return out


def _legend_grid_placeholder_handle():
    """Handle invisível para ocupar células na grelha da legenda (alinhamento por ncol)."""
    return LegendGridPlaceholder()


def combine_legend_handles_reynolds_block_first(
    reynolds_handles,
    tail_handles,
    *,
    ncol=LEGEND_TOP_NCOL,
    tail_ncol=None,
):
    """
    Coloca os handles da primeira lista (``Re_{sl}``, ``Re_{sg}``, ``J_l``, etc.)
    nas primeira(s) linha(s) da grelha da legenda; as entradas seguintes (ex.: flow
    patterns) ocupam as linhas posteriores.

    ``ncol`` regula o bloco superior; ``tail_ncol`` (por omissão = ``ncol``) regula o
    bloco inferior — tipicamente ``tail_ncol`` = número de padrões de escoamento para
    uma única linha. A largura da grelha é ``max(ncol, tail_ncol)``.

    O matplotlib constrói colunas empilhando blocos contíguos da lista linear de handles;
    por isso é necessário converter uma grelha **linha a linha** para a ordem linear que
    o legend usa (preencimento por colunas / ordem Fortran da matriz).
    """
    head_ncol = ncol
    if tail_ncol is None:
        tail_ncol = head_ncol
    grid_ncol = max(head_ncol, tail_ncol)

    def _rows_from_handles(handles, block_ncol):
        if not handles:
            return []
        h = list(handles)
        rows = []
        for i in range(0, len(h), block_ncol):
            chunk = h[i : i + block_ncol]
            if len(chunk) < grid_ncol:
                chunk = chunk + [_legend_grid_placeholder_handle()] * (grid_ncol - len(chunk))
            rows.append(chunk)
        return rows

    r_rows = _rows_from_handles(reynolds_handles, head_ncol)
    t_rows = _rows_from_handles(tail_handles, tail_ncol)
    if not r_rows:
        all_rows = t_rows
    elif not t_rows:
        all_rows = r_rows
    else:
        all_rows = r_rows + t_rows

    if not all_rows:
        return []

    linear = []
    for c in range(grid_ncol):
        for r in range(len(all_rows)):
            linear.append(all_rows[r][c])
    return linear


def generate_alpha_vs_jg_plot(df, sheet_name, fluid_1, fluid_2, theta):
    """
    Gera um plot científico de jG vs α, onde cada jL é uma série diferente.
    Inclui símbolos diferentes para cada Flow Pattern.
    
    Args:
        df (pd.DataFrame): DataFrame com os dados
        sheet_name (str): Nome da aba/sheet
    """
    try:
        available_cols = list(df.columns)
        col_mapping = build_column_mapping(df)
        if not ensure_alpha_in_column_mapping(col_mapping, available_cols):
            print('Colunas ausentes para plot jG vs α: α (ou Alpha / Void fraction)')
            print(f'Colunas disponíveis: {list(col_mapping.keys())}')
            return
        miss = missing_column_keys(col_mapping, ['jG', 'jL', 'Flow Pattern'])
        if miss:
            print(f'Colunas ausentes para plot jG vs α: {miss}')
            print(f'Colunas disponíveis: {list(col_mapping.keys())}')
            return

        setup_plot_style()
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)

        # Obter valores únicos de jL e agrupar por séries
        jl_col = col_mapping['jL']
        jg_col = col_mapping['jG']
        alpha_col = col_mapping['α']
        flow_pattern_col = col_mapping['Flow Pattern']
        
        # Agrupar séries por j_L medido (valores próximos → média com 2 d.p.; ver _cluster_measured_jl_legend_labels)
        df_plot = df.copy().reset_index(drop=True)
        # Garantir α numérico (NAS pode trazer como string; colunas duplicadas → 1ª coluna)
        _assign_numeric_to_first_named_column(df_plot, alpha_col, _one_col_series(df_plot, alpha_col))
        _cluster_measured_jl_legend_labels(df_plot, jl_col)
        jl_series = sorted(df_plot['_jl_legend_group_mean'].dropna().unique())
        jl_series = [jl for jl in jl_series if pd.notna(jl)]
        
        print(f"Séries de jL encontradas (médias por grupo de valores medidos próximos): {jl_series}")

        line_styles, _ = get_series_line_and_marker_styles()
        flow_pattern_symbols = get_flow_pattern_symbols()
        alpha_unc_col = _find_point_uncertainty_column(df_plot, col_mapping, 'U_alpha')
        lowest_jl_uncertainty_indices = _indices_for_lowest_jl_series(
            df_plot,
            jl_col,
            _one_col_series(df_plot, jg_col),
            _one_col_series(df_plot, alpha_col),
        )

        for i, jl_lbl in enumerate(jl_series):
            mask = df_plot['_jl_legend_group_mean'] == jl_lbl
            jl_disp = float(jl_lbl)

            jg_data = _one_col_series_masked(df_plot, mask, jg_col)
            alpha_data = _one_col_series_masked(df_plot, mask, alpha_col)
            flow_pattern_data = _one_col_series_masked(df_plot, mask, flow_pattern_col)
            
            # Remover valores nulos
            valid_mask = pd.notna(jg_data) & pd.notna(alpha_data)
            jg_clean = jg_data[valid_mask]
            alpha_clean = alpha_data[valid_mask]
            flow_pattern_clean = flow_pattern_data[valid_mask]
            if len(jg_clean) > 0:
                line_style = line_styles[i % len(line_styles)]
                sorted_data = sorted(zip(jg_clean, alpha_clean, flow_pattern_clean))
                jg_sorted = [x[0] for x in sorted_data]
                alpha_sorted = [x[1] for x in sorted_data]
                flow_pattern_sorted = [x[2] for x in sorted_data]

                ax.plot(
                    jg_sorted,
                    alpha_sorted,
                    line_style,
                    color='black',
                    linewidth=1.5,
                    zorder=1,
                )

                for j, (jg_val, alpha_val, flow_pattern) in enumerate(zip(jg_sorted, alpha_sorted, flow_pattern_sorted)):
                    pattern_data = style_for_flow_pattern_cell(
                        flow_pattern, flow_pattern_symbols
                    )
                    symbol = pattern_data['symbol']
                    color = pattern_data['color']
                    ax.scatter(jg_val, alpha_val, c=color, marker=symbol, s=100,
                             edgecolors=color, linewidth=1, zorder=2)
                    
                
                
                print(f"Plotando série jL = {jl_disp:.2f} m/s com {len(jg_clean)} pontos")

        n_unc = _plot_selected_y_uncertainty(
            ax,
            df_plot,
            lowest_jl_uncertainty_indices,
            jg_col,
            alpha_col,
            alpha_unc_col,
        )
        if n_unc:
            print(f"Incerteza de alpha incluida nos {n_unc} pontos de menor jL.")

        legend_elements = flow_pattern_legend_handles(
            df_plot, flow_pattern_col, flow_pattern_symbols
        )

        x_label = r'$J_{g} [m/s]$'
        y_label = r'$\alpha$ [-]'
        # Configurar eixos com fonte acadêmica
        ax.set_xlabel(x_label, fontsize=26, fontfamily='serif')
        ax.set_ylabel(y_label, fontsize=26, fontfamily='serif')
        # Remover título conforme solicitado
        
        ax.set_axisbelow(True)
        
        # Configurar ticks menores para grade mais detalhada
        ax.minorticks_on()

        jg_vals = pd.to_numeric(_one_col_series(df_plot, jg_col), errors='coerce').to_numpy()
        configure_linear_jg_axis_tick_locators(ax, jg_vals)

        configure_alpha_y_axis_ticks(ax)

        ax.set_ylim(bottom=0, top=1)
        
        # Configurar tamanho dos ticks com fonte acadêmica
        ax.tick_params(axis='both', which='major', labelsize=20)
        _set_ticklabels_font_serif(ax)
        apply_subtle_gray_grid(ax)
        _apply_linear_axes_one_decimal_format(ax)
        finalize_jg_plot_xlim(ax)

        jl_legend_elements = []
        for i, jl_lbl in enumerate(jl_series):
            jl_disp = float(jl_lbl)
            line_style = line_styles[i % len(line_styles)]
            jl_legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    color='black',
                    linestyle=line_style,
                    linewidth=1.5,
                    label=rf'$J_{{l}}$ = {jl_disp:.2f} m/s',
                )
            )
        apply_two_row_top_legend(ax, jl_legend_elements, legend_elements)
        add_theta_and_mean_uncertainty_text(
            ax,
            df_plot,
            theta,
            col_mapping,
            uncertainty_key='U_alpha',
            placement='bottom',
            fontsize=17,
            kind='alpha',
        )

        plt.tight_layout()
        save_figure_to_sheet_dir(f'{sheet_name}_alpha_vs_jg', sheet_name)

    except Exception as e:
        print(f"Erro ao gerar plot: {e}")
        import traceback
        traceback.print_exc()


def generate_alpha_vs_beta_homogeneous_parity_plot(df, sheet_name, fluid_1, fluid_2, theta):
    """
    Gráfico de paridade: fração de vazio experimental (α) vs modelo homogêneo
    β = j_g / (j_g + j_l). Inclui a linha α = β e marcadores por flow pattern.

    Destina-se ao caso de **uma única inclinação** (uma aba); não é usado no modo
    multi-abas ou com a opção 'all'.
    """
    try:
        available_cols = list(df.columns)
        col_mapping = build_column_mapping(df)
        if not ensure_alpha_in_column_mapping(col_mapping, available_cols):
            print('Paridade α vs β: coluna α (void fraction) ausente.')
            print(f'Colunas disponíveis: {list(col_mapping.keys())}')
            return
        miss = missing_column_keys(col_mapping, ['jG', 'jL', 'Flow Pattern'])
        if miss:
            print(f'Paridade α vs β: colunas ausentes {miss}')
            print(f'Colunas disponíveis: {list(col_mapping.keys())}')
            return

        setup_plot_style()
        df_plot = df.copy().reset_index(drop=True)
        jl_col = col_mapping['jL']
        jg_col = col_mapping['jG']
        alpha_col = col_mapping['α']
        flow_pattern_col = col_mapping['Flow Pattern']

        _assign_numeric_to_first_named_column(
            df_plot, alpha_col, _one_col_series(df_plot, alpha_col)
        )
        jg_arr = pd.to_numeric(_one_col_series(df_plot, jg_col), errors='coerce').to_numpy(
            dtype=float
        )
        jl_src = 'jL_raw' if 'jL_raw' in df_plot.columns else jl_col
        jl_arr = pd.to_numeric(_one_col_series(df_plot, jl_src), errors='coerce').to_numpy(
            dtype=float
        )
        alpha_arr = pd.to_numeric(_one_col_series(df_plot, alpha_col), errors='coerce').to_numpy(
            dtype=float
        )
        fp_series = _one_col_series(df_plot, flow_pattern_col)

        denom = jg_arr + jl_arr
        beta_arr = np.divide(
            jg_arr,
            denom,
            out=np.full_like(jg_arr, np.nan, dtype=float),
            where=(denom > 0) & np.isfinite(denom),
        )

        valid = (
            np.isfinite(beta_arr)
            & np.isfinite(alpha_arr)
            & np.isfinite(jg_arr)
            & np.isfinite(jl_arr)
            & (beta_arr >= 0.0)
            & (beta_arr <= 1.0)
        )
        if not np.any(valid):
            print(f'Paridade α vs β ({sheet_name}): nenhum ponto válido.')
            return

        flow_pattern_symbols = get_flow_pattern_symbols()
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)

        ax.plot(
            [0.0, 1.0],
            [0.0, 1.0],
            linestyle='--',
            color='0.35',
            linewidth=1.2,
            zorder=1,
        )

        for i in range(len(df_plot)):
            if not valid[i]:
                continue
            pd_ = style_for_flow_pattern_cell(fp_series.iloc[i], flow_pattern_symbols)
            ax.scatter(
                beta_arr[i],
                alpha_arr[i],
                c=pd_['color'],
                marker=pd_['symbol'],
                s=100,
                edgecolors=pd_['color'],
                linewidth=1,
                zorder=2,
            )

        df_valid = df_plot.loc[valid].copy()
        legend_fp = flow_pattern_legend_handles(
            df_valid, flow_pattern_col, flow_pattern_symbols
        )

        _ncol = legend_ncol_flow_patterns_only(len(legend_fp))
        ax.legend(
            handles=legend_fp,
            ncol=_ncol,
            **LEGEND_TOP_KWARGS,
        )

        ax.set_xlabel(
            r'$\beta$ [-]',
            fontsize=26,
            fontfamily='serif',
        )
        ax.set_ylabel(r'$\alpha$ [-]', fontsize=26, fontfamily='serif')
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect('equal', adjustable='box')

        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        configure_alpha_y_axis_ticks(ax, major_step=0.2)
        ax.minorticks_on()

        ax.tick_params(axis='both', which='major', labelsize=20)
        _set_ticklabels_font_serif(ax)
        apply_subtle_gray_grid(ax)

        _apply_linear_axes_one_decimal_format(ax)

        _add_theta_annotation_box(ax, theta)

        plt.tight_layout()
        save_figure_to_sheet_dir(f'{sheet_name}_alpha_vs_beta_homogeneous_parity', sheet_name)
        print(f'Paridade α vs β ({sheet_name}): {int(np.sum(valid))} pontos.')

    except Exception as e:
        print(f'Erro ao gerar paridade α vs β: {e}')
        import traceback
        traceback.print_exc()


def _jl_jg_linear_axis_limits(arr, margin_frac=0.05):
    """
    Limites lineares: faixa dos dados + margem, arredondada para fora em passos
    “redondos” (mesma ordem de grandeza que o intervalo).
    """
    v = np.asarray(arr, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None, None
    lo, hi = float(v.min()), float(v.max())
    span = hi - lo
    if span <= 0:
        eps = max(abs(lo), 1e-12) * 0.1
        lo, hi = lo - eps, hi + eps
        span = hi - lo
    a, b = lo - margin_frac * span, hi + margin_frac * span
    rng = b - a
    exp = np.floor(np.log10(max(rng, 1e-30)))
    base_step = 10.0**exp
    for mult in (0.2, 0.5, 1.0, 2.0, 5.0, 10.0):
        step = base_step * mult
        if rng / step <= 14:
            break
    else:
        step = base_step
    lower = np.floor(a / step) * step
    upper = np.ceil(b / step) * step
    if upper <= lower:
        upper = lower + step
    return lower, upper


def _log_axis_compact_hi(rmax: float, *, margin_hi_rel: float) -> float:
    """
    Limite superior compacto para escala log: dentro de cada década [L, 10L),
    usa o menor entre {2L, 5L, 10L} que ainda cobre rmax com margem relativa.

    Isto reproduz o comportamento desejado (ex.: 1.2→2, 3.5→5, 7.5→10, 12→20, 85→100)
    sem estender sempre até à próxima potência de 10 inteira.
    """
    r_eff = rmax * (1.0 + margin_hi_rel)
    if not np.isfinite(r_eff) or r_eff <= 0:
        return float('nan')
    L = 10.0 ** np.floor(np.log10(r_eff))
    for mult in (3.0, 5.0, 10.0):
        cand = mult * L
        if cand + 1e-15 * max(cand, 1.0) >= r_eff:
            return float(cand)
    return float(10.0 * L)


def _log_axis_compact_limits(
    values,
    *,
    margin_hi_rel=None,
    fallback=None,
):
    """
    Limites em escala logarítmica compactos para matplotlib (loglog / eixo X log).

    Inferior: ``10**floor(log10(min))`` — potência de 10 imediatamente abaixo do menor
    valor > 0 (ex.: min 0,15 → 0,1; min 0,05 → 0,01).

    Superior: menor valor da forma k·L com k ∈ {2, 5, 10} e
    ``L = 10**floor(log10(rmax*(1+margin)))`` tal que ainda cubra o máximo com margem;
    assim valores longe do topo da década não abrem o eixo até à década seguinte.
    """
    if margin_hi_rel is None:
        margin_hi_rel = LOG_AXIS_COMPACT_MARGIN_HI_REL
    if fallback is None:
        fallback = JL_JG_FLOW_MATRIX_LOG_FALLBACK

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size == 0:
        return float(fallback[0]), float(fallback[1])

    rmin = float(arr.min())
    rmax = float(arr.max())
    lo = 10.0 ** np.floor(np.log10(rmin))
    hi = _log_axis_compact_hi(rmax, margin_hi_rel=margin_hi_rel)

    if not np.isfinite(hi) or hi <= lo:
        hi = lo * 10.0
    return float(lo), float(hi)


def _jl_jg_flow_pattern_matrix_log_ylim_cap(jl_v: np.ndarray, y_lo: float, y_hi: float):
    """
    Matriz j_L vs j_G em escala log: se max(j_L) < ``JL_JG_FLOW_MATRIX_LOG_Y_CAP_MIN``,
    garante que o limite superior em Y seja pelo menos esse valor.
    """
    if jl_v.size == 0:
        return y_lo, y_hi
    jl_max = float(np.max(jl_v))
    if np.isfinite(jl_max) and jl_max < JL_JG_FLOW_MATRIX_LOG_Y_CAP_MIN:
        return y_lo, max(float(y_hi), JL_JG_FLOW_MATRIX_LOG_Y_CAP_MIN)
    return y_lo, y_hi


def _axis_tick_decimal_log(x, pos):
    """Ticks em escala log com valores numéricos decimais (sem potências no eixo)."""
    if not np.isfinite(x) or x <= 0:
        return ''
    return f'{x:g}'


def _axis_tick_decimal_linear(x, pos):
    """Ticks em escala linear com 1 casa decimal."""
    if not np.isfinite(x):
        return ''
    return f'{x:.1f}'


def _axis_tick_integer_linear(x, pos):
    """Ticks em escala linear sem casas decimais (p.ex. inclinação θ [°])."""
    if not np.isfinite(x):
        return ''
    return f'{int(round(x))}'


def _apply_linear_axes_one_decimal_format(ax, *, integer_x=False):
    """Formata eixo Y com 1 casa decimal; eixo X idem ou inteiro se ``integer_x``."""
    ax.yaxis.set_major_formatter(FuncFormatter(_axis_tick_decimal_linear))
    if integer_x:
        ax.xaxis.set_major_formatter(FuncFormatter(_axis_tick_integer_linear))
    else:
        ax.xaxis.set_major_formatter(FuncFormatter(_axis_tick_decimal_linear))


def _jl_jg_matrix_load_valid_series(df):
    """
    Extrai j_G, j_L e flow pattern por ponto (apenas j_G, j_L > 0).
    Usa j_L direto da planilha: coluna ``jL_raw`` se existir (antes de médias de cluster),
    senão a coluna canónica de jL.
    Retorna dict com arrays e df para legenda, ou (None, lista de chaves em falta).
    """
    col_mapping = build_column_mapping(df)
    miss = missing_column_keys(col_mapping, ['jG', 'jL', 'Flow Pattern'])
    if miss:
        return None, miss
    df_plot = df.copy().reset_index(drop=True)
    jl_col = col_mapping['jL']
    jg_col = col_mapping['jG']
    flow_pattern_col = col_mapping['Flow Pattern']
    jl_series_col = 'jL_raw' if 'jL_raw' in df_plot.columns else jl_col
    jg_data = pd.to_numeric(_one_col_series(df_plot, jg_col), errors='coerce')
    jl_data = pd.to_numeric(_one_col_series(df_plot, jl_series_col), errors='coerce')
    fp_data = _one_col_series(df_plot, flow_pattern_col)
    valid = (
        pd.notna(jg_data)
        & pd.notna(jl_data)
        & (jg_data > 0)
        & (jl_data > 0)
    )
    if valid.sum() == 0:
        return None, None
    return {
        'jg_v': jg_data[valid].to_numpy(dtype=float),
        'jl_v': jl_data[valid].to_numpy(dtype=float),
        'fp_v': fp_data[valid],
        'df_leg': df_plot.loc[valid].copy(),
        'flow_pattern_col': flow_pattern_col,
    }, None


def _decorate_jl_jg_matrix_subplot(
    ax,
    theta,
    *,
    show_xlabel: bool,
    show_ylabel: bool,
):
    """Painel do mosaico: sem legenda local; rótulos só onde solicitado."""
    if show_xlabel:
        ax.set_xlabel(r'$J_{g}$ [m/s]', fontsize=22, fontfamily='serif')
    else:
        ax.set_xlabel('')
    if show_ylabel:
        ax.set_ylabel(r'$J_{l}$ [m/s]', fontsize=22, fontfamily='serif')
    else:
        ax.set_ylabel('')
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=13)
    _set_ticklabels_font_serif(ax)
    apply_subtle_gray_grid(ax)
    _add_theta_annotation_box(ax, theta)


def _finalize_jl_jg_matrix_axes(ax, *, legend_elements, theta):
    """Rótulos, legenda e anotação de inclinação comuns aos dois modos (log / linear)."""
    ax.set_xlabel(r'$J_{g}$ [m/s]', fontsize=26, fontfamily='serif')
    ax.set_ylabel(r'$J_{l}$ [m/s]', fontsize=26, fontfamily='serif')
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=16)
    _set_ticklabels_font_serif(ax)
    apply_subtle_gray_grid(ax)
    if legend_elements:
        _ncol = legend_ncol_flow_patterns_only(len(legend_elements))
        ax.legend(
            handles=legend_elements,
            ncol=_ncol,
            **LEGEND_TOP_KWARGS,
        )
    _add_theta_annotation_box(ax, theta)


def generate_jl_vs_jg_flow_pattern_matrix_plot(df, sheet_name, fluid_1, fluid_2, theta):
    """
    Matriz experimental j_L vs j_G: gera dois gráficos com os mesmos pontos —
    (1) escalas logarítmicas e (2) escalas lineares — com marcadores por padrão de escoamento.

    Destina-se ao modo em que não se usa a opção 'all' no carregamento (uma ou mais
    abas escolhidas manualmente); não é chamada quando `is_all_selected` é True.
    """
    try:
        col_mapping = build_column_mapping(df)
        miss = missing_column_keys(col_mapping, ['jG', 'jL', 'Flow Pattern'])
        if miss:
            print(f'Colunas ausentes para matriz j_L vs j_G: {miss}')
            print(f'Colunas disponíveis: {list(col_mapping.keys())}')
            return

        setup_plot_style()
        df_plot = df.copy().reset_index(drop=True)
        jl_col = col_mapping['jL']
        jg_col = col_mapping['jG']
        flow_pattern_col = col_mapping['Flow Pattern']

        jg_data = pd.to_numeric(_one_col_series(df_plot, jg_col), errors='coerce')
        jl_data = pd.to_numeric(_one_col_series(df_plot, jl_col), errors='coerce')
        fp_data = _one_col_series(df_plot, flow_pattern_col)

        valid = (
            pd.notna(jg_data)
            & pd.notna(jl_data)
            & (jg_data > 0)
            & (jl_data > 0)
        )
        n_skip = int((~valid).sum())
        if valid.sum() == 0:
            print(f'Matriz j_L vs j_G ({sheet_name}): nenhum ponto com j_G > 0 e j_L > 0.')
            return
        if n_skip:
            print(
                f'Matriz j_L vs j_G ({sheet_name}): ignoradas {n_skip} linhas '
                '(valores não positivos ou inválidos).'
            )

        flow_pattern_symbols = get_flow_pattern_symbols()

        jg_v = jg_data[valid].to_numpy(dtype=float)
        jl_v = jl_data[valid].to_numpy(dtype=float)
        fp_v = fp_data[valid]

        df_leg = df_plot.loc[valid].copy()
        legend_elements = flow_pattern_legend_handles(
            df_leg, flow_pattern_col, flow_pattern_symbols
        )

        def scatter_points(ax):
            for jg_val, jl_val, fp in zip(jg_v, jl_v, fp_v):
                pd_ = style_for_flow_pattern_cell(fp, flow_pattern_symbols)
                symbol = pd_['symbol']
                color = pd_['color']
                ax.scatter(
                    jg_val,
                    jl_val,
                    c=color,
                    marker=symbol,
                    s=120,
                    edgecolors=color,
                    linewidth=1,
                    zorder=2,
                )

        # --- Log-log: limites compactos a partir dos dados (igual critério ao mosaico / Re_sg)
        x_lo, x_hi = _log_axis_compact_limits(jg_v)
        y_lo, y_hi = _log_axis_compact_limits(jl_v)
        y_lo, y_hi = _jl_jg_flow_pattern_matrix_log_ylim_cap(jl_v, y_lo, y_hi)

        fig_log, ax_log = plt.subplots(figsize=PLOT_FIGSIZE)
        scatter_points(ax_log)
        ax_log.set_xscale('log')
        ax_log.set_yscale('log')
        ax_log.set_xlim(x_lo, x_hi)
        ax_log.set_ylim(y_lo, y_hi)
        ax_log.xaxis.set_major_formatter(FuncFormatter(_axis_tick_decimal_log))
        ax_log.yaxis.set_major_formatter(FuncFormatter(_axis_tick_decimal_log))
        _finalize_jl_jg_matrix_axes(ax_log, legend_elements=legend_elements, theta=theta)
        plt.tight_layout()
        save_figure_to_sheet_dir(
            f'{sheet_name}_jl_vs_jg_flow_pattern_matrix_log', sheet_name
        )
        plt.close(fig_log)

        # --- Linear: mesmos pontos; limites “arredondados” para a grandeza mais próxima
        xl0, xl1 = _jl_jg_linear_axis_limits(jg_v)
        yl0, yl1 = _jl_jg_linear_axis_limits(jl_v)
        if xl0 is None or yl0 is None:
            print(f'Matriz j_L vs j_G ({sheet_name}): não foi possível definir limites lineares.')
            return

        fig_lin, ax_lin = plt.subplots(figsize=PLOT_FIGSIZE)
        scatter_points(ax_lin)
        ax_lin.set_xlim(xl0, xl1)
        ax_lin.set_ylim(yl0, yl1)
        ax_lin.xaxis.set_major_formatter(FuncFormatter(_axis_tick_decimal_linear))
        ax_lin.yaxis.set_major_formatter(FuncFormatter(_axis_tick_decimal_linear))
        ax_lin.minorticks_on()
        _finalize_jl_jg_matrix_axes(ax_lin, legend_elements=legend_elements, theta=theta)
        plt.tight_layout()
        save_figure_to_sheet_dir(
            f'{sheet_name}_jl_vs_jg_flow_pattern_matrix_linear', sheet_name
        )
        plt.close(fig_lin)

        print(
            f'Matriz j_L vs j_G ({sheet_name}): {len(jg_v)} pontos — '
            f'guardados log e linear (limites log [{x_lo:g}, {x_hi:g}] × [{y_lo:g}, {y_hi:g}], '
            f'linear [{xl0:g}, {xl1:g}] × [{yl0:g}, {yl1:g}]).'
        )

    except Exception as e:
        print(f'Erro ao gerar matriz j_L vs j_G: {e}')
        import traceback
        traceback.print_exc()


def _save_jl_jg_mosaic_figure(fig, base_filename):
    """PDF/PNG na pasta ``orientation_plots`` junto ao ficheiro Excel (multi-inclinação)."""
    out_dir = os.path.join(os.path.dirname(file_path), 'orientation_plots')
    os.makedirs(out_dir, exist_ok=True)
    pdf_file = os.path.join(out_dir, f'{base_filename}.pdf')
    png_file = os.path.join(out_dir, f'{base_filename}.png')
    fig.savefig(pdf_file, **PLOT_SAVEFIG_KWARGS)
    fig.savefig(png_file, **PLOT_SAVEFIG_KWARGS)
    print(f'Mosaico guardado: {pdf_file}')
    print(f'Mosaico guardado: {png_file}')


def generate_jl_vs_jg_flow_pattern_matrix_mosaic_plot(all_dataframes, selected_sheets):
    """
    Mosaico (2 colunas) das matrizes experimentais j_L vs j_G com flow patterns,
    uma subfigura por inclinação/aba. Cada painel usa caixa de eixos quadrada
    (mesmo lado em polegadas); última linha ímpar: painel centralizado,
    quadrangular e alinhado à largura de uma coluna. Escalas log com limites
    compactos comuns a todos os painéis (união implícita: min/max sobre todas as abas).
    Uma única legenda de flow pattern no topo.

    Destinado ao modo em que se escolhe 'all' e várias abas/inclinações para analisar.

    Args:
        all_dataframes (dict): sheet_name -> DataFrame
        selected_sheets (list): abas a incluir (normalmente as chaves escolhidas pelo utilizador)
    """
    try:
        names = [s for s in selected_sheets if s in all_dataframes]
        if len(names) < 2:
            print(
                'Mosaico j_L vs j_G: precisa de pelo menos 2 abas/inclinações; ignorado.'
            )
            return

        setup_plot_style()
        flow_pattern_symbols = get_flow_pattern_symbols()

        panels = []
        for sheet_name in names:
            df = all_dataframes[sheet_name]
            pack, miss = _jl_jg_matrix_load_valid_series(df)
            if pack is None:
                if miss:
                    print(
                        f'Mosaico j_L vs j_G: aba {sheet_name} sem colunas {miss}; ignorada.'
                    )
                else:
                    print(
                        f'Mosaico j_L vs j_G: aba {sheet_name} sem pontos j_G, j_L > 0; ignorada.'
                    )
                continue
            fluid_1, fluid_2, direction, theta, ID, is_validation = extract_info_from_filename(
                sheet_name
            )
            if direction == 'Downward':
                theta = -theta
            panels.append(
                {
                    'sheet_name': sheet_name,
                    'theta': theta,
                    'jg_v': pack['jg_v'],
                    'jl_v': pack['jl_v'],
                    'fp_v': pack['fp_v'],
                    'df_leg': pack['df_leg'],
                    'flow_pattern_col': pack['flow_pattern_col'],
                }
            )

        if len(panels) < 2:
            print(
                'Mosaico j_L vs j_G: menos de 2 abas com dados válidos; mosaico não gerado.'
            )
            return

        jg_cat = np.concatenate([p['jg_v'] for p in panels])
        jl_cat = np.concatenate([p['jl_v'] for p in panels])

        # Limites log iguais em todos os painéis (dados de todas as abas)
        x_lo, x_hi = _log_axis_compact_limits(jg_cat)
        y_lo, y_hi = _log_axis_compact_limits(jl_cat)
        y_lo, y_hi = _jl_jg_flow_pattern_matrix_log_ylim_cap(jl_cat, y_lo, y_hi)
        xl0, xl1 = _jl_jg_linear_axis_limits(jg_cat)
        yl0, yl1 = _jl_jg_linear_axis_limits(jl_cat)
        if xl0 is None or yl0 is None:
            print('Mosaico j_L vs j_G: não foi possível calcular limites lineares globais.')
            return

        fp_col = panels[0]['flow_pattern_col']
        df_leg_all = pd.concat([p['df_leg'] for p in panels], ignore_index=True)
        legend_handles = flow_pattern_legend_handles(
            df_leg_all, fp_col, flow_pattern_symbols
        )

        ncols = 2
        n = len(panels)
        nrows = (n + ncols - 1) // ncols
        panel = MOSAIC_JL_JG_PANEL_IN
        fig_w = ncols * panel + 1.0
        fig_h = nrows * panel + 1.45

        def _draw_mosaic(scale_log: bool):
            fig = plt.figure(figsize=(fig_w, fig_h))
            gs = GridSpec(nrows, ncols, figure=fig)

            axes_list = []
            for i in range(n):
                r, c = divmod(i, ncols)
                if i == n - 1 and (n % ncols == 1):
                    inner = GridSpecFromSubplotSpec(
                        1,
                        3,
                        subplot_spec=gs[r, :],
                        width_ratios=[1, 2, 1],
                        wspace=0,
                    )
                    ax = fig.add_subplot(inner[0, 1])
                else:
                    ax = fig.add_subplot(gs[r, c])
                axes_list.append(ax)

            for i, ax in enumerate(axes_list):
                p = panels[i]
                for jg_val, jl_val, fp in zip(p['jg_v'], p['jl_v'], p['fp_v']):
                    pd_ = style_for_flow_pattern_cell(fp, flow_pattern_symbols)
                    ax.scatter(
                        jg_val,
                        jl_val,
                        c=pd_['color'],
                        marker=pd_['symbol'],
                        s=85,
                        edgecolors=pd_['color'],
                        linewidth=0.9,
                        zorder=2,
                    )
                if scale_log:
                    ax.set_xscale('log')
                    ax.set_yscale('log')
                    ax.set_xlim(x_lo, x_hi)
                    ax.set_ylim(y_lo, y_hi)
                    ax.xaxis.set_major_formatter(FuncFormatter(_axis_tick_decimal_log))
                    ax.yaxis.set_major_formatter(FuncFormatter(_axis_tick_decimal_log))
                else:
                    ax.set_xlim(xl0, xl1)
                    ax.set_ylim(yl0, yl1)
                    ax.xaxis.set_major_formatter(FuncFormatter(_axis_tick_decimal_linear))
                    ax.yaxis.set_major_formatter(FuncFormatter(_axis_tick_decimal_linear))
                    ax.minorticks_on()

                # Caixa do eixo quadrada (mesmo tamanho visual em todos os painéis)
                ax.set_box_aspect(1)

                r = i // ncols
                c = i % ncols
                is_bottom = r == nrows - 1
                is_left = (c == 0) or (i == n - 1 and (n % ncols == 1))
                _decorate_jl_jg_matrix_subplot(
                    ax,
                    p['theta'],
                    show_xlabel=is_bottom,
                    show_ylabel=is_left,
                )

            if legend_handles:
                ncol_leg = legend_ncol_flow_patterns_only(len(legend_handles))
                fig.legend(
                    handles=legend_handles,
                    ncol=ncol_leg,
                    loc='lower center',
                    bbox_to_anchor=(0.5, 1.0),
                    frameon=False,
                    fontsize=20,
                    prop={'family': 'serif'},
                )

            # top mais alto aproxima os painéis da legenda (menos faixa branca entre ambos)
            fig.subplots_adjust(left=0.09, right=0.97, top=0.93, bottom=0.08, hspace=0.38, wspace=0.30)
            return fig

        fig_log = _draw_mosaic(scale_log=True)
        _save_jl_jg_mosaic_figure(fig_log, 'mosaic_jl_vs_jg_flow_pattern_matrix_log')
        plt.close(fig_log)

        fig_lin = _draw_mosaic(scale_log=False)
        _save_jl_jg_mosaic_figure(fig_lin, 'mosaic_jl_vs_jg_flow_pattern_matrix_linear')
        plt.close(fig_lin)

        print(
            f'Mosaico j_L vs j_G: {n} painéis quadrados (log + linear); log '
            f'[{x_lo:g}, {x_hi:g}] × [{y_lo:g}, {y_hi:g}] — legenda com '
            f'{len(legend_handles)} padrões.'
        )

    except Exception as e:
        print(f'Erro ao gerar mosaico j_L vs j_G: {e}')
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
        col_mapping = build_column_mapping(df)
        miss = missing_column_keys(col_mapping, ['jG', 'jL', 'dp/dz_F', 'Flow Pattern'])
        if miss:
            print(f'Colunas ausentes para plot: {miss}')
            print(f'Colunas disponíveis: {list(col_mapping.keys())}')
            return

        setup_plot_style()
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)

        jl_col = col_mapping['jL']
        jg_col = col_mapping['jG']
        dp_dz_f_col = col_mapping['dp/dz_F']
        flow_pattern_col = col_mapping['Flow Pattern']
        
        # Agrupar séries por j_L medido na planilha (valores próximos → média, 2 d.p.)
        df_plot = df.copy().reset_index(drop=True)
        _cluster_measured_jl_legend_labels(df_plot, jl_col)
        jl_series = sorted(df_plot['_jl_legend_group_mean'].dropna().unique())
        jl_series = [jl for jl in jl_series if pd.notna(jl)]
        
        print(f"Séries de jL encontradas (médias por grupo de valores medidos próximos): {jl_series}")

        line_styles, _ = get_series_line_and_marker_styles()
        flow_pattern_symbols = get_flow_pattern_symbols()
        dp_dz_f_unc_col = _find_point_uncertainty_column(df_plot, col_mapping, 'U_dpdz_F')
        lowest_jl_uncertainty_indices = _indices_for_lowest_jl_series(
            df_plot,
            jl_col,
            _one_col_series(df_plot, jg_col),
            _one_col_series(df_plot, dp_dz_f_col),
        )

        for i, jl_lbl in enumerate(jl_series):
            mask = df_plot['_jl_legend_group_mean'] == jl_lbl
            jl_disp = float(jl_lbl)

            jg_data = _one_col_series_masked(df_plot, mask, jg_col)
            frictional_data = _one_col_series_masked(df_plot, mask, dp_dz_f_col)
            flow_pattern_data = _one_col_series_masked(df_plot, mask, flow_pattern_col)
            
            # Remover valores nulos
            valid_mask = pd.notna(jg_data) & pd.notna(frictional_data)
            jg_clean = jg_data[valid_mask]
            frictional_clean = frictional_data[valid_mask]/1000     #To plot in kPa
            flow_pattern_clean = flow_pattern_data[valid_mask]
            if len(jg_clean) > 0:
                line_style = line_styles[i % len(line_styles)]
                sorted_data = sorted(zip(jg_clean, frictional_clean, flow_pattern_clean))
                jg_sorted = [x[0] for x in sorted_data]
                frictional_sorted = [x[1] for x in sorted_data]
                flow_pattern_sorted = [x[2] for x in sorted_data]

                ax.plot(
                    jg_sorted,
                    frictional_sorted,
                    line_style,
                    color='black',
                    linewidth=1.5,
                    zorder=1,
                )

                for j, (jg_val, frictional_val, flow_pattern) in enumerate(
                    zip(jg_sorted, frictional_sorted, flow_pattern_sorted)
                ):
                    pattern_data = style_for_flow_pattern_cell(
                        flow_pattern, flow_pattern_symbols
                    )
                    symbol = pattern_data['symbol']
                    color = pattern_data['color']
                    ax.scatter(
                        jg_val,
                        frictional_val,
                        c=color,
                        marker=symbol,
                        s=100,
                        edgecolors=color,
                        linewidth=1,
                        zorder=2,
                    )

                print(f"Plotando série jL = {jl_disp:.2f} m/s com {len(jg_clean)} pontos")

        n_unc = _plot_selected_y_uncertainty(
            ax,
            df_plot,
            lowest_jl_uncertainty_indices,
            jg_col,
            dp_dz_f_col,
            dp_dz_f_unc_col,
            y_scale=1 / 1000,
        )
        if n_unc:
            print(f"Incerteza de gradiente friccional incluida nos {n_unc} pontos de menor jL.")

        legend_elements = flow_pattern_legend_handles(
            df_plot, flow_pattern_col, flow_pattern_symbols
        )

        x_label = r'$J_{g} [m/s]$'
        # Configurar eixos com fonte acadêmica
        ax.set_xlabel(x_label, fontsize=26, fontfamily='serif')
        ax.set_ylabel(YLABEL_DP_DZ_F, fontsize=26, fontfamily='serif')
        # Remover título conforme solicitado
        
        ax.set_axisbelow(True)
        
        # Configurar ticks menores para grade mais detalhada
        ax.minorticks_on()

        jg_vals = pd.to_numeric(_one_col_series(df_plot, jg_col), errors='coerce').to_numpy()
        configure_linear_jg_axis_tick_locators(ax, jg_vals)

        y_lo, y_hi = dpdz_y_limits_for_plot('f')
        if y_lo is not None or y_hi is not None:
            apply_fixed_dpdz_y_axis(ax, 'f')
        else:
            ax.set_ylim(bottom=0)
            configure_dpdz_y_axis_minor_ticks(ax)
        ax.tick_params(axis='both', which='major', labelsize=20)
        _set_ticklabels_font_serif(ax)
        apply_subtle_gray_grid(ax)
        _apply_linear_axes_one_decimal_format(ax)
        finalize_jg_plot_xlim(ax)

        jl_legend_elements = []
        for i, jl_lbl in enumerate(jl_series):
            jl_disp = float(jl_lbl)
            line_style = line_styles[i % len(line_styles)]
            jl_legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    color='black',
                    linestyle=line_style,
                    linewidth=1.5,
                    label=rf'$J_{{l}}$ = {jl_disp:.2f} m/s',
                )
            )

        apply_two_row_top_legend(ax, jl_legend_elements, legend_elements)
        add_theta_and_mean_uncertainty_text(
            ax,
            df_plot,
            theta,
            col_mapping,
            uncertainty_key='U_dpdz_F',
            unit_latex=r'\mathrm{kPa/m}',
            placement='top',
            fontsize=17,
            kind='kpa_per_m',
        )

        plt.tight_layout()
        save_figure_to_sheet_dir(f'{sheet_name}_frictional_vs_jg', sheet_name)

    except Exception as e:
        print(f"Erro ao gerar plot: {e}")
        import traceback
        traceback.print_exc()


def generate_dpdzt_vs_jg_plot(df, sheet_name, fluid_1, fluid_2, theta):
    """
    Gera um plot científico de jG vs (∂P/∂z)_t, com a mesma estrutura de frictional_vs_jg:
    uma série por jL (média por grupo arredondada a 2 casas decimais), símbolos por Flow Pattern.
    O eixo Y não é forçado a y ≥ 0 (gradiente total pode ser negativo).
    """
    try:
        col_mapping = build_column_mapping(df)
        miss = missing_column_keys(col_mapping, ['jG', 'jL', 'dp/dz_T', 'Flow Pattern'])
        if miss:
            print(f'Colunas ausentes para plot total vs jG: {miss}')
            print(f'Colunas disponíveis: {list(col_mapping.keys())}')
            return

        setup_plot_style()
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)

        jl_col = col_mapping['jL']
        jg_col = col_mapping['jG']
        dp_dz_t_col = col_mapping['dp/dz_T']
        flow_pattern_col = col_mapping['Flow Pattern']

        df_plot = df.copy().reset_index(drop=True)
        _cluster_measured_jl_legend_labels(df_plot, jl_col)

        jl_series = sorted(df_plot['_jl_legend_group_mean'].dropna().unique())
        jl_series = [jl for jl in jl_series if pd.notna(jl)]

        print(f"[total_vs_jg] Séries de jL (médias por grupo de valores medidos próximos): {jl_series}")

        line_styles, _ = get_series_line_and_marker_styles()
        flow_pattern_symbols = get_flow_pattern_symbols()
        dp_dz_t_unc_col = _find_point_uncertainty_column(df_plot, col_mapping, 'U_dpdz_T')
        lowest_jl_uncertainty_indices = _indices_for_lowest_jl_series(
            df_plot,
            jl_col,
            _one_col_series(df_plot, jg_col),
            _one_col_series(df_plot, dp_dz_t_col),
        )

        for i, jl_lbl in enumerate(jl_series):
            mask = df_plot['_jl_legend_group_mean'] == jl_lbl
            jl_disp = float(jl_lbl)

            jg_data = _one_col_series_masked(df_plot, mask, jg_col)
            total_data = _one_col_series_masked(df_plot, mask, dp_dz_t_col)
            flow_pattern_data = _one_col_series_masked(df_plot, mask, flow_pattern_col)

            valid_mask = pd.notna(jg_data) & pd.notna(total_data)
            jg_clean = jg_data[valid_mask]
            total_clean = total_data[valid_mask] / 1000
            flow_pattern_clean = flow_pattern_data[valid_mask]
            if len(jg_clean) > 0:
                line_style = line_styles[i % len(line_styles)]
                sorted_data = sorted(zip(jg_clean, total_clean, flow_pattern_clean))
                jg_sorted = [x[0] for x in sorted_data]
                total_sorted = [x[1] for x in sorted_data]
                flow_pattern_sorted = [x[2] for x in sorted_data]

                ax.plot(
                    jg_sorted,
                    total_sorted,
                    line_style,
                    color='black',
                    linewidth=1.5,
                    zorder=1,
                )

                for jg_val, total_val, flow_pattern in zip(
                    jg_sorted, total_sorted, flow_pattern_sorted
                ):
                    pattern_data = style_for_flow_pattern_cell(
                        flow_pattern, flow_pattern_symbols
                    )
                    symbol = pattern_data['symbol']
                    color = pattern_data['color']
                    ax.scatter(
                        jg_val,
                        total_val,
                        c=color,
                        marker=symbol,
                        s=100,
                        edgecolors=color,
                        linewidth=1,
                        zorder=2,
                    )

                print(f"[total_vs_jg] Plotando série jL = {jl_disp:.2f} m/s com {len(jg_clean)} pontos")

        n_unc = _plot_selected_y_uncertainty(
            ax,
            df_plot,
            lowest_jl_uncertainty_indices,
            jg_col,
            dp_dz_t_col,
            dp_dz_t_unc_col,
            y_scale=1 / 1000,
        )
        if n_unc:
            print(f"Incerteza de gradiente total incluida nos {n_unc} pontos de menor jL.")

        legend_elements = flow_pattern_legend_handles(
            df_plot, flow_pattern_col, flow_pattern_symbols
        )

        x_label = r'$J_{g} [m/s]$'
        ax.set_xlabel(x_label, fontsize=26, fontfamily='serif')
        ax.set_ylabel(YLABEL_DP_DZ_T, fontsize=26, fontfamily='serif')

        ax.set_axisbelow(True)
        ax.minorticks_on()

        jg_vals = pd.to_numeric(_one_col_series(df_plot, jg_col), errors='coerce').to_numpy()
        configure_linear_jg_axis_tick_locators(ax, jg_vals)
        y_lo, y_hi = dpdz_y_limits_for_plot('t')
        if y_lo is not None or y_hi is not None:
            apply_fixed_dpdz_y_axis(ax, 't')

        ax.tick_params(axis='both', which='major', labelsize=20)
        _set_ticklabels_font_serif(ax)
        apply_subtle_gray_grid(ax)
        _apply_linear_axes_one_decimal_format(ax)
        finalize_jg_plot_xlim(ax)

        jl_legend_elements = []
        for i, jl_lbl in enumerate(jl_series):
            jl_disp = float(jl_lbl)
            line_style = line_styles[i % len(line_styles)]
            jl_legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    color='black',
                    linestyle=line_style,
                    linewidth=1.5,
                    label=rf'$J_{{l}}$ = {jl_disp:.2f} m/s',
                )
            )

        apply_two_row_top_legend(ax, jl_legend_elements, legend_elements)
        add_theta_and_mean_uncertainty_text(
            ax,
            df_plot,
            theta,
            col_mapping,
            uncertainty_key='U_dpdz_T',
            unit_latex=r'\mathrm{kPa/m}',
            placement='bottom_right',
            fontsize=17,
            kind='kpa_per_m',
        )

        plt.tight_layout()
        save_figure_to_sheet_dir(f'{sheet_name}_total_vs_jg', sheet_name)

    except Exception as e:
        print(f"Erro ao gerar plot total vs jG: {e}")
        import traceback
        traceback.print_exc()


def generate_alpha_vs_Reg_plot(df, sheet_name, fluid_1, fluid_2, theta):
    """
    Gera um plot de Re_sg vs α (void fraction), com a mesma abordagem dos plots *_vs_Re_g:
    uma série por Re_sl (agrupado), símbolos por Flow Pattern, eixo X em escala log.
    """
    try:
        available_cols = list(df.columns)
        col_mapping = build_column_mapping(df)
        if not ensure_alpha_in_column_mapping(col_mapping, available_cols):
            print('Colunas ausentes para plot Re_sg vs α: α (ou Alpha / Void fraction)')
            print(f'Colunas disponíveis: {list(col_mapping.keys())}')
            return
        miss = missing_column_keys(
            col_mapping,
            ['jG', 'jL', 'Flow Pattern', 'Temp.', 'Gauge Pressure'],
        )
        if miss:
            print(f'Colunas ausentes para plot Re_sg vs α: {miss}')
            print(f'Colunas disponíveis: {list(col_mapping.keys())}')
            return

        setup_plot_style()
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)

        alpha_col = col_mapping['α']
        flow_pattern_col = col_mapping['Flow Pattern']
        compute_Re_sg_column(df, col_mapping, fluid_1)

        df_plot = df.copy().reset_index(drop=True)
        _assign_numeric_to_first_named_column(
            df_plot, alpha_col, _one_col_series(df_plot, alpha_col)
        )

        if 'Re_sl_group' not in df.columns:
            standardize_liquid_conditions({sheet_name: df})

        df_plot['Re_sl_group'] = _one_col_series(df, 'Re_sl_group').values

        Re_sl_series = sorted(df_plot['Re_sl_group'].dropna().unique())
        Re_sl_series = [re_l for re_l in Re_sl_series if pd.notna(re_l)]

        Re_sg_plot = _one_col_series(df_plot, 'Re_sg')

        print(f"Séries de Re_sl (padronizado) para α vs Re_sg: {Re_sl_series}")

        line_styles, _ = get_series_line_and_marker_styles()
        flow_pattern_symbols = get_flow_pattern_symbols()
        alpha_unc_col = _find_point_uncertainty_column(df_plot, col_mapping, 'U_alpha')
        lowest_resl_uncertainty_indices = _indices_for_lowest_resl_series(
            df_plot,
            _one_col_series(df_plot, 'Re_sg'),
            _one_col_series(df_plot, alpha_col),
        )
        # Só séries com pontos entram na legenda (estilo de linha por Re_sl)
        legend_series_meta = []

        for i, re_l in enumerate(Re_sl_series):
            mask = _mask_re_sl_group(df_plot['Re_sl_group'], re_l)

            alpha_data = _one_col_series_masked(df_plot, mask, alpha_col)
            flow_pattern_data = _one_col_series_masked(df_plot, mask, flow_pattern_col)
            Re_sg_masked = Re_sg_plot[mask]
            valid_mask = pd.notna(Re_sg_masked) & pd.notna(alpha_data)
            Re_sg_clean = Re_sg_masked[valid_mask]
            alpha_clean = alpha_data[valid_mask]
            flow_pattern_clean = flow_pattern_data[valid_mask]

            if len(Re_sg_clean) > 0:
                line_style = line_styles[i % len(line_styles)]
                legend_series_meta.append((re_l, line_style))

                sorted_data = sorted(zip(Re_sg_clean, alpha_clean, flow_pattern_clean))
                Re_sg_sorted = [x[0] for x in sorted_data]
                alpha_sorted = [x[1] for x in sorted_data]
                flow_pattern_sorted = [x[2] for x in sorted_data]

                ax.plot(
                    Re_sg_sorted,
                    alpha_sorted,
                    line_style,
                    color='black',
                    linewidth=1.5,
                    zorder=1,
                )

                for Re_sg_val, alpha_val, flow_pattern in zip(
                    Re_sg_sorted, alpha_sorted, flow_pattern_sorted
                ):
                    pattern_data = style_for_flow_pattern_cell(
                        flow_pattern, flow_pattern_symbols
                    )
                    symbol = pattern_data['symbol']
                    p_color = pattern_data['color']
                    ax.scatter(
                        Re_sg_val,
                        alpha_val,
                        c=p_color,
                        marker=symbol,
                        s=100,
                        edgecolors=p_color,
                        linewidth=1,
                        zorder=2,
                    )

                print(f"Plotando α vs Re_sg, série Re_sl = {re_l:.1f} com {len(Re_sg_clean)} pontos")

        n_unc = _plot_selected_y_uncertainty(
            ax,
            df_plot,
            lowest_resl_uncertainty_indices,
            'Re_sg',
            alpha_col,
            alpha_unc_col,
        )
        if n_unc:
            print(f"Incerteza de alpha incluida nos {n_unc} pontos de menor Re_sl.")

        legend_elements = flow_pattern_legend_handles(
            df_plot, flow_pattern_col, flow_pattern_symbols
        )

        ax.set_xlabel(r'$Re_{sg}$ [-]', fontsize=26, fontfamily='serif')
        ax.set_ylabel(r'$\alpha$ [-]', fontsize=26, fontfamily='serif')
        style_axes_re_g_log_x(ax, y_major_step=0.1, y_lim_bottom=0, y_lim_top=1.0)
        configure_alpha_y_axis_ticks(ax)

        Re_sl_legend_elements = re_sl_legend_handles_from_meta(legend_series_meta)

        apply_two_row_top_legend(ax, Re_sl_legend_elements, legend_elements)
        add_theta_and_mean_uncertainty_text(
            ax,
            df_plot,
            theta,
            col_mapping,
            uncertainty_key='U_alpha',
            placement='bottom',
            fontsize=17,
            kind='alpha',
        )

        plt.tight_layout()
        save_figure_to_sheet_dir(f'{sheet_name}_alpha_vs_Re_g', sheet_name)

    except Exception as e:
        print(f"Erro ao gerar plot α vs Re_sg: {e}")
        import traceback
        traceback.print_exc()


def generate_dpdzf_vs_Reg_plot(df, sheet_name, fluid_1, fluid_2, theta):
    """
    Gera um plot científico de Re_sg vs dp/dz_F, onde cada Re_sl é uma série diferente.
    Inclui símbolos diferentes para cada Flow Pattern.
    """
    try:
        col_mapping = build_column_mapping(df)
        miss = missing_column_keys(
            col_mapping,
            ['jG', 'jL', 'dp/dz_F', 'Flow Pattern', 'Temp.', 'Gauge Pressure'],
        )
        if miss:
            print(f'Colunas ausentes para plot: {miss}')
            print(f'Colunas disponíveis: {list(col_mapping.keys())}')
            return

        setup_plot_style()
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)

        jl_col = col_mapping['jL']
        compute_Re_sg_column(df, col_mapping, fluid_1)

        dp_dz_f_col = col_mapping['dp/dz_F']
        flow_pattern_col = col_mapping['Flow Pattern']
        
        # Padronizar antes de copiar; legendas de j_L usam valores medidos (jL_raw) agrupados
        if 'Re_sl_group' not in df.columns:
            standardize_liquid_conditions({sheet_name: df})

        df_plot = df.copy().reset_index(drop=True)
        if 'Re_sl_group' in df.columns:
            df_plot['Re_sl_group'] = _one_col_series(df, 'Re_sl_group').values
        if 'jL_raw' in df.columns:
            df_plot['jL_raw'] = _one_col_series(df, 'jL_raw').values
        _cluster_measured_jl_legend_labels(df_plot, jl_col)
        jl_series = sorted(df_plot['_jl_legend_group_mean'].dropna().unique())
        jl_series = [jl for jl in jl_series if pd.notna(jl)]

        # Obter séries únicas de Re_sl (agrupado/padronizado)
        Re_sl_series = sorted(df_plot['Re_sl_group'].dropna().unique())
        Re_sl_series = [re_l for re_l in Re_sl_series if pd.notna(re_l)]
        
        Re_sg_plot = _one_col_series(df_plot, 'Re_sg')
        
        print(f"Séries de jL (médias por grupo de valores medidos próximos): {jl_series}")
        print(f"Séries de Re_sl (padronizado) encontradas: {Re_sl_series}")

        line_styles, _ = get_series_line_and_marker_styles()
        flow_pattern_symbols = get_flow_pattern_symbols()
        dp_dz_f_unc_col = _find_point_uncertainty_column(df_plot, col_mapping, 'U_dpdz_F')
        lowest_resl_uncertainty_indices = _indices_for_lowest_resl_series(
            df_plot,
            _one_col_series(df_plot, 'Re_sg'),
            _one_col_series(df_plot, dp_dz_f_col),
        )
        legend_series_meta = []

        for i, re_l in enumerate(Re_sl_series):
            mask = _mask_re_sl_group(df_plot['Re_sl_group'], re_l)

            frictional_data = _one_col_series_masked(df_plot, mask, dp_dz_f_col)
            flow_pattern_data = _one_col_series_masked(df_plot, mask, flow_pattern_col)
            Re_sg_masked = Re_sg_plot[mask]
            valid_mask = pd.notna(Re_sg_masked) & pd.notna(frictional_data)
            Re_sg_clean = Re_sg_masked[valid_mask]
            frictional_clean = frictional_data[valid_mask] / 1000
            flow_pattern_clean = flow_pattern_data[valid_mask]

            if len(Re_sg_clean) > 0:
                line_style = line_styles[i % len(line_styles)]
                legend_series_meta.append((re_l, line_style))

                sorted_data = sorted(zip(Re_sg_clean, frictional_clean, flow_pattern_clean))
                Re_sg_sorted = [x[0] for x in sorted_data]
                frictional_sorted = [x[1] for x in sorted_data]
                flow_pattern_sorted = [x[2] for x in sorted_data]

                ax.plot(
                    Re_sg_sorted,
                    frictional_sorted,
                    line_style,
                    color='black',
                    linewidth=1.5,
                    zorder=1,
                )

                for Re_sg_val, frictional_val, flow_pattern in zip(
                    Re_sg_sorted, frictional_sorted, flow_pattern_sorted
                ):
                    pattern_data = style_for_flow_pattern_cell(
                        flow_pattern, flow_pattern_symbols
                    )
                    symbol = pattern_data['symbol']
                    color = pattern_data['color']
                    ax.scatter(
                        Re_sg_val,
                        frictional_val,
                        c=color,
                        marker=symbol,
                        s=100,
                        edgecolors=color,
                        linewidth=1,
                        zorder=2,
                    )

                print(f"Plotando série Re_sl = {re_l:.1f} com {len(Re_sg_clean)} pontos")

        n_unc = _plot_selected_y_uncertainty(
            ax,
            df_plot,
            lowest_resl_uncertainty_indices,
            'Re_sg',
            dp_dz_f_col,
            dp_dz_f_unc_col,
            y_scale=1 / 1000,
        )
        if n_unc:
            print(f"Incerteza de gradiente friccional incluida nos {n_unc} pontos de menor Re_sl.")

        legend_elements = flow_pattern_legend_handles(
            df_plot, flow_pattern_col, flow_pattern_symbols
        )

        ax.set_xlabel(r'$Re_{sg}$ [-]', fontsize=26, fontfamily='serif')
        ax.set_ylabel(YLABEL_DP_DZ_F, fontsize=26, fontfamily='serif')
        y_lo, y_hi = dpdz_y_limits_for_plot('f')
        y_step = dpdz_y_major_step_for_plot('f')
        if y_lo is not None or y_hi is not None:
            style_axes_re_g_log_x(
                ax, y_major_step=y_step, y_lim_bottom=y_lo, y_lim_top=y_hi
            )
            configure_dpdz_y_axis_minor_ticks(ax)
        else:
            style_axes_re_g_log_x(ax, y_major_step=0.5, y_lim_bottom=0, y_lim_top=None)

        Re_sl_legend_elements = re_sl_legend_handles_from_meta(legend_series_meta)
        apply_two_row_top_legend(ax, Re_sl_legend_elements, legend_elements)
        add_theta_and_mean_uncertainty_text(
            ax,
            df_plot,
            theta,
            col_mapping,
            uncertainty_key='U_dpdz_F',
            unit_latex=r'\mathrm{kPa/m}',
            placement='top',
            fontsize=17,
            kind='kpa_per_m',
        )

        plt.tight_layout()
        save_figure_to_sheet_dir(f'{sheet_name}_frictional_vs_Re_g', sheet_name)

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
        col_mapping = build_column_mapping(df)
        miss = missing_column_keys(
            col_mapping,
            ['jG', 'jL', 'dp/dz_T', 'Flow Pattern', 'Temp.', 'Gauge Pressure'],
        )
        if miss:
            print(f'Colunas ausentes para plot: {miss}')
            print(f'Colunas disponíveis: {list(col_mapping.keys())}')
            return

        setup_plot_style()
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)

        jl_col = col_mapping['jL']
        compute_Re_sg_column(df, col_mapping, fluid_1)

        dp_dz_t_col = col_mapping['dp/dz_T']
        flow_pattern_col = col_mapping['Flow Pattern']
        
        # Padronizar antes de copiar; legendas de j_L usam valores medidos (jL_raw) agrupados
        if 'Re_sl_group' not in df.columns:
            standardize_liquid_conditions({sheet_name: df})

        df_plot = df.copy().reset_index(drop=True)
        if 'Re_sl_group' in df.columns:
            df_plot['Re_sl_group'] = _one_col_series(df, 'Re_sl_group').values
        if 'jL_raw' in df.columns:
            df_plot['jL_raw'] = _one_col_series(df, 'jL_raw').values
        _cluster_measured_jl_legend_labels(df_plot, jl_col)
        jl_series = sorted(df_plot['_jl_legend_group_mean'].dropna().unique())
        jl_series = [jl for jl in jl_series if pd.notna(jl)]

        # Obter séries únicas de Re_sl (agrupado/padronizado)
        Re_sl_series = sorted(df_plot['Re_sl_group'].dropna().unique())
        Re_sl_series = [re_l for re_l in Re_sl_series if pd.notna(re_l)]
        
        Re_sg_plot = _one_col_series(df_plot, 'Re_sg')
        
        print(f"Séries de jL (médias por grupo de valores medidos próximos): {jl_series}")
        print(f"Séries de Re_sl (padronizado) encontradas: {Re_sl_series}")

        line_styles, _ = get_series_line_and_marker_styles()
        flow_pattern_symbols = get_flow_pattern_symbols()
        dp_dz_t_unc_col = _find_point_uncertainty_column(df_plot, col_mapping, 'U_dpdz_T')
        lowest_resl_uncertainty_indices = _indices_for_lowest_resl_series(
            df_plot,
            _one_col_series(df_plot, 'Re_sg'),
            _one_col_series(df_plot, dp_dz_t_col),
        )
        legend_series_meta = []

        for i, re_l in enumerate(Re_sl_series):
            mask = _mask_re_sl_group(df_plot['Re_sl_group'], re_l)

            total_data = _one_col_series_masked(df_plot, mask, dp_dz_t_col)
            flow_pattern_data = _one_col_series_masked(df_plot, mask, flow_pattern_col)
            Re_sg_masked = Re_sg_plot[mask]
            valid_mask = pd.notna(Re_sg_masked) & pd.notna(total_data)
            Re_sg_clean = Re_sg_masked[valid_mask]
            total_clean = total_data[valid_mask] / 1000
            flow_pattern_clean = flow_pattern_data[valid_mask]

            if len(Re_sg_clean) > 0:
                line_style = line_styles[i % len(line_styles)]
                legend_series_meta.append((re_l, line_style))

                sorted_data = sorted(zip(Re_sg_clean, total_clean, flow_pattern_clean))
                Re_sg_sorted = [x[0] for x in sorted_data]
                total_sorted = [x[1] for x in sorted_data]
                flow_pattern_sorted = [x[2] for x in sorted_data]

                ax.plot(
                    Re_sg_sorted,
                    total_sorted,
                    line_style,
                    color='black',
                    linewidth=1.5,
                    zorder=1,
                )

                for Re_sg_val, total_val, flow_pattern in zip(
                    Re_sg_sorted, total_sorted, flow_pattern_sorted
                ):
                    pattern_data = style_for_flow_pattern_cell(
                        flow_pattern, flow_pattern_symbols
                    )
                    symbol = pattern_data['symbol']
                    color = pattern_data['color']
                    ax.scatter(
                        Re_sg_val,
                        total_val,
                        c=color,
                        marker=symbol,
                        s=100,
                        edgecolors=color,
                        linewidth=1,
                        zorder=2,
                    )

                print(f"Plotando série Re_sl = {re_l:.1f} com {len(Re_sg_clean)} pontos")

        n_unc = _plot_selected_y_uncertainty(
            ax,
            df_plot,
            lowest_resl_uncertainty_indices,
            'Re_sg',
            dp_dz_t_col,
            dp_dz_t_unc_col,
            y_scale=1 / 1000,
        )
        if n_unc:
            print(f"Incerteza de gradiente total incluida nos {n_unc} pontos de menor Re_sl.")

        legend_elements = flow_pattern_legend_handles(
            df_plot, flow_pattern_col, flow_pattern_symbols
        )

        ax.set_xlabel(r'$Re_{sg}$ [-]', fontsize=26, fontfamily='serif')
        ax.set_ylabel(YLABEL_DP_DZ_T, fontsize=26, fontfamily='serif')
        y_lo, y_hi = dpdz_y_limits_for_plot('t')
        y_step = dpdz_y_major_step_for_plot('t')
        if y_lo is not None or y_hi is not None:
            style_axes_re_g_log_x(
                ax, y_major_step=y_step, y_lim_bottom=y_lo, y_lim_top=y_hi
            )
            configure_dpdz_y_axis_minor_ticks(ax)
        else:
            # Total: não forçar y ≥ 0 (gradiente pode ser negativo); ticks Y automáticos.
            style_axes_re_g_log_x(ax, y_major_step=None, y_lim_bottom=None, y_lim_top=None)

        Re_sl_legend_elements = re_sl_legend_handles_from_meta(legend_series_meta)
        apply_two_row_top_legend(ax, Re_sl_legend_elements, legend_elements)
        add_theta_and_mean_uncertainty_text(
            ax,
            df_plot,
            theta,
            col_mapping,
            uncertainty_key='U_dpdz_T',
            unit_latex=r'\mathrm{kPa/m}',
            placement='bottom_right',
            fontsize=17,
            kind='kpa_per_m',
        )

        plt.tight_layout()
        save_figure_to_sheet_dir(f'{sheet_name}_total_vs_Re_g', sheet_name)

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
        
        # Verificar colunas necessárias (build_column_mapping inclui aliases -dpdz_* dos ficheiros Mean)
        required_cols = ['jL', 'dp/dz_F', 'Flow Pattern']
        col_mapping = build_column_mapping(df)

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

        # Re_sg: mesma regra que nos plots *_vs_Re_g (Rho_gas/Mu_gas quando existirem)
        try:
            compute_Re_sg_column(df_temp, col_mapping, fluid_1)
            Re_sg_series = pd.to_numeric(df_temp['Re_sg'], errors='coerce')
        except Exception as e:
            Re_sg_series = pd.Series([np.nan] * len(df_temp), index=df_temp.index)
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
            if _is_blank_excel_value(fp_val):
                fp_val = FLOW_PATTERN_UNCLASSIFIED
            elif pd.notna(fp_val) and isinstance(fp_val, str):
                key = fp_val.strip().upper()
                fp_val = NAS_FLOW_PATTERN_MAP.get(key, NAS_FLOW_PATTERN_MAP.get(fp_val.strip(), fp_val))
            alpha_val = alpha_series.iloc[idx]
            re_sg_val = Re_sg_series.iloc[idx] if not Re_sg_series.isna().all() else np.nan
            point_id = f"P{idx+1:02d}"
            
            if pd.isna(re_sl_val) or pd.isna(fric_val):
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
    *,
    integer_x_axis=False,
):
    """
    Gera gráficos genéricos de (quantidade y_column) vs orientação θ para cada Re_sl,
    conectando pontos com o mesmo point_id (P01, P02, ...) entre inclinações.
    Se houver mais de um sistema bifásico (AW, AO, SO, AD), cada sistema usa uma cor
    e a legenda indica o sistema; com um único sistema as linhas permanecem pretas.
    Proporção da figura 12:9; legenda no topo (ncol, estilo) e ticks como nos plots por aba.
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
        line_styles, _ = get_series_line_and_marker_styles()

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

        # Mesma proporção que os demais plots (12:9)
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
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
                        pattern_data = style_for_flow_pattern_cell(
                            row['flow_pattern'], flow_pattern_symbols
                        )
                        symbol = pattern_data['symbol']
                        pattern_color = pattern_data['color']
                        ax.scatter(
                            row['theta'],
                            row[y_column],
                            c=pattern_color,
                            marker=symbol,
                            s=150,
                            edgecolors=pattern_color,
                            linewidth=1.5,
                            zorder=2,
                            alpha=0.9,
                        )
            # Legenda: primeiro entradas $Re_{sg}$ (séries), depois sistema (cor)
            re_sg_legend_elements = []
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
                re_sg_legend_elements.append(
                    plt.Line2D(
                        [0], [0],
                        linestyle=line_styles[i % len(line_styles)],
                        color='gray',
                        linewidth=1.5,
                        label=series_label,
                    )
                )
            system_legend_elements = []
            for sys in systems:
                system_legend_elements.append(
                    plt.Line2D(
                        [0], [0],
                        linestyle='-',
                        color=system_to_color[sys],
                        linewidth=2.5,
                        label=str(sys),
                    )
                )
            series_legend_elements = re_sg_legend_elements + system_legend_elements
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
                    pattern_data = style_for_flow_pattern_cell(
                        row['flow_pattern'], flow_pattern_symbols
                    )
                    symbol = pattern_data['symbol']
                    pattern_color = pattern_data['color']
                    ax.scatter(
                        row['theta'],
                        row[y_column],
                        c=pattern_color,
                        marker=symbol,
                        s=150,
                        edgecolors=pattern_color,
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

        ax.set_xlabel('Pipe inclination (θ) [°]', fontsize=26, fontfamily='serif')
        ax.set_ylabel(y_label, fontsize=26, fontfamily='serif')
        ax.set_axisbelow(True)
        ax.minorticks_on()
        # Ticks como em alpha_vs_jg
        ax.tick_params(axis='both', which='major', labelsize=20)
        _set_ticklabels_font_serif(ax)

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

        # Limites e ticks Y para void fraction (como alpha_vs_jg)
        if y_column == 'alpha':
            ax.set_ylim(bottom=0.0, top=1.0)
            configure_alpha_y_axis_ticks(ax)
        elif y_column == 'frictional':
            apply_fixed_dpdz_y_axis(ax, 'f')
        elif y_column == 'total':
            apply_fixed_dpdz_y_axis(ax, 't')

        used_labels = {
            flow_pattern_display_label(v) for v in data_re['flow_pattern']
        }
        pattern_legend_elements = []
        for pattern in sorted(used_labels, key=str):
            pattern_data = style_for_flow_pattern_cell(pattern, flow_pattern_symbols)
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
                    markeredgecolor=color,
                    markeredgewidth=1,
                    label=flow_pattern_legend_label(pattern),
                )
            )

        apply_subtle_gray_grid(ax)

        _apply_linear_axes_one_decimal_format(ax, integer_x=integer_x_axis)

        apply_two_row_top_legend(
            ax,
            series_legend_elements,
            pattern_legend_elements,
            tail_fontsize=LEGEND_ORIENTATION_FLOW_PATTERN_FONTSIZE,
            tail_markerscale=LEGEND_ORIENTATION_FLOW_PATTERN_MARKERSCALE,
        )

        # Reynolds de líquido nominal (Re_sl), arredondado à centena (mesmo que *_vs_Re_g).
        # Frictional e total: centro inferior; alpha: canto inferior direito.
        try:
            label = rf"$Re_{{sl}} \approx {round_re_sl_for_display(re_l)}$"
            if y_column in ('frictional', 'total'):
                ax.text(
                    0.5,
                    0.03,
                    label,
                    transform=ax.transAxes,
                    fontsize=18,
                    fontfamily='serif',
                    ha='center',
                    va='bottom',
                )
            else:
                x_min, x_max = ax.get_xlim()
                y_min, y_max = ax.get_ylim()
                x_text = x_max - 0.02 * (x_max - x_min)
                y_text = y_min + 0.05 * (y_max - y_min)
                ax.text(
                    x_text,
                    y_text,
                    label,
                    fontsize=18,
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
            y_label=YLABEL_DP_DZ_F,
            base_name_prefix='frictional_vs_orientation',
            integer_x_axis=True,
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
            y_label=r'$\alpha$ [-]',
            base_name_prefix='alpha_vs_orientation',
            integer_x_axis=True,
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
            y_label=YLABEL_DP_DZ_T,
            base_name_prefix='total_vs_orientation',
            integer_x_axis=True,
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

            # Paridade α vs β homogêneo — apenas uma inclinação (uma única aba nos dados)
            if isinstance(df, dict):
                if len(df) == 1:
                    sheet_name_pb, sheet_df_pb = next(iter(df.items()))
                    fluid_1_pb, fluid_2_pb, direction_pb, theta_pb, ID_pb, is_validation_pb = (
                        extract_info_from_filename(sheet_name_pb)
                    )
                    if direction_pb == 'Downward':
                        theta_pb = -theta_pb
                    generate_alpha_vs_beta_homogeneous_parity_plot(
                        sheet_df_pb, sheet_name_pb, fluid_1_pb, fluid_2_pb, theta_pb
                    )
            else:
                generate_alpha_vs_beta_homogeneous_parity_plot(
                    df, sheet_name, fluid_1, fluid_2, theta
                )

            # Matriz experimental j_L vs j_G (log-log) com flow patterns — apenas sem opção 'all'
            if isinstance(df, dict):
                for sheet_name, sheet_df in df.items():
                    fluid_1, fluid_2, direction, theta, ID, is_validation = extract_info_from_filename(sheet_name)
                    if direction == 'Downward':
                        theta = -theta
                    generate_jl_vs_jg_flow_pattern_matrix_plot(
                        sheet_df, sheet_name, fluid_1, fluid_2, theta
                    )
            else:
                generate_jl_vs_jg_flow_pattern_matrix_plot(
                    df, sheet_name, fluid_1, fluid_2, theta
                )
            
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
                    fluid_1, fluid_2, direction, theta, ID, is_validation = extract_info_from_filename(sheet_name)
                    if direction == 'Downward':
                        theta = -theta
                    generate_dpdzt_vs_jg_plot(sheet_df, sheet_name, fluid_1, fluid_2, theta)
            else:
                generate_dpdzt_vs_jg_plot(df, sheet_name, fluid_1, fluid_2, theta)
            
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
                    fluid_1, fluid_2, direction, theta, ID, is_validation = extract_info_from_filename(sheet_name)
                    if direction == 'Downward':
                        theta = -theta
                    generate_alpha_vs_Reg_plot(sheet_df, sheet_name, fluid_1, fluid_2, theta)
            else:
                generate_alpha_vs_Reg_plot(df, sheet_name, fluid_1, fluid_2, theta)

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
                if len(selected) >= 2:
                    generate_jl_vs_jg_flow_pattern_matrix_mosaic_plot(df, selected)
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
                output_excel = os.path.join(
                    base_dir, f'processed_all_sheets_{base_name}.xlsx'
                )
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
                # Uma aba: nome inclui o identificador da aba processada
                suffix = _safe_processed_filename_segment(sheet_name)
                output_excel = os.path.join(
                    base_dir, f'processed_{base_name}_{suffix}.xlsx'
                )
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
