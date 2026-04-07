# Manual básico — `plot_tool.py`

Ferramenta para ler planilhas de resultados experimentais, padronizar condições de líquido e gerar figuras (PDF e PNG) de fração de vazio, gradientes de pressão e Reynolds do gás.

## Requisitos

- Python 3 com: `pandas`, `numpy`, `matplotlib`, `openpyxl`, **CoolProp** (cálculo de `Re_sg` e propriedades).
- Caminho do Excel acessível (use `/` no caminho, mesmo no Windows).

## Configuração antes de correr

No início do ficheiro `plot_tool.py`, secção **INPUTS**:

| Variável | Função |
|----------|--------|
| `file_path` | Caminho completo do ficheiro `.xlsx` / `.xlsm`. |
| `NAS_file` | `True` = formato NAS (cabeçalhos e colunas renomeados automaticamente). `False` = formato Excel “clássico” do script (linhas 3–4 cabeçalho/unidades). |

Com `NAS_file = True`, só entram abas cujo nome está em `ALLOWED_SHEETS_NAS`.

## Como executar

```bash
python plot_tool.py
```

## Interação no terminal

1. **Uma aba** — aparece lista numerada: digite o número da aba (ex.: `16`).
2. **Várias abas** — escolha `all`; depois indique os números das abas separados por espaços (ex.: `2 3 16`) ou outra vez `all` para todas as abas filtradas.

- **Uma aba ou seleção parcial:** gera os **plots por aba** (ver abaixo) e guarda também um Excel processado.
- **Seleção que resulta em várias abas (`all` + lista):** mesmo comportamento — um conjunto de gráficos **por cada aba** carregada.
- **Fluxo “all” que devolve dicionário de abas** (conforme menu): **não** gera os plots individuais por aba nessa execução; em vez disso gera os **gráficos de orientação** (friccional, α, total vs inclinação), desde que existam dados e colunas necessárias.

## Onde ficam os ficheiros

- **Plots por aba:** pasta `…/nome_da_aba/` junto ao Excel de entrada, com nomes do tipo `{aba}_alpha_vs_jg.pdf` (e `.png`), idem para `frictional_vs_jg`, `total_vs_jg`, `*_vs_Re_g`.
- **Plots de orientação:** subpasta `orientation_plots/` no mesmo diretório do Excel; resumo agrupado em `orientation_summary_grouped_Re_sl.xlsx`.
- **Excel processado:** `{nome_do_ficheiro}_processed.xlsx` ou `_processed_all_sheets.xlsx` junto ao ficheiro de entrada.

## Plots gerados (por aba, quando aplicável)

- `α` vs `j_g` e vs `Re_{sg}`
- Gradiente **friccional** \((\partial P/\partial z)_f\) vs `j_g` e vs `Re_{sg}`
- Gradiente **total** \((\partial P/\partial z)_t\) vs `j_g` e vs `Re_{sg}`

Séries por `j_L` ou `Re_{sl}` (conforme o gráfico); símbolos por **flow pattern**. Células vazias em α, gradientes ou flow pattern são tratadas como dados em falta (ou “Unclassified” no flow pattern). `Re_sg` não é calculado com P/T inválidos.

## Convenções do nome da aba

O script infere fluidos, direção e inclinação a partir do **nome da aba** (ex.: inclinação e “Downward” para correção do sinal de θ). Mantenha a convenção esperada pelo vosso fluxo de trabalho.

## Problemas frequentes

- **Erro ao ler aba:** verificar `NAS_file`, caminho e se a aba está em `ALLOWED_SHEETS_NAS` (modo NAS).
- **Colunas em falta:** mensagem no terminal indica o que falta; o plot correspondente é ignorado.
- **CoolProp:** pressão/temperatura em falta por linha → `Re_sg` em NaN nessa linha (sem crash).

Para alterar legenda, tamanho da figura (12×9), rótulos dos eixos ou mapeamento NAS → nomes completos de flow pattern, edite as constantes e funções no próprio `plot_tool.py` (secção inicial e funções `get_flow_pattern_symbols`, `read_single_sheet_nas`, etc.).
