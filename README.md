# Análise de Dados Experimentais - Processamento de Séries Temporais

Este programa realiza a análise de dados experimentais de escoamento bifásico, processando séries temporais e calculando parâmetros importantes como fração de vazio (Alpha) e gradientes de pressão.

## Requisitos

- Python 3.x
- Bibliotecas necessárias:
  - uncertainties
  - pandas
  - numpy
  - matplotlib
  - CoolProp

## Configuração Inicial

1. Abra o arquivo `exp_unc.py`
2. Configure os parâmetros iniciais na seção "INPUTS":
   ```python
   file_path = 'caminho/para/seu/arquivo'  # Caminho do arquivo a ser analisado
   ```

3. Configure os sensores:
   ```python
   sensor_Yokogawa = 'PDT-M-0101D-30Kpa_mA'  # Nome da coluna do sensor Yokogawa
   sensor_Endress = 'PDT-M-0101-40kPa_mA'     # Nome da coluna do sensor Endress
   ```

4. Ajuste os valores de calibração do densitômetro:
   ```python
   I_g = 252883    # Intensidade padrão para o gás
   I_f = 151287    # Intensidade padrão para o líquido
   ```

## Execução do Programa

1. Execute o programa:
   ```bash
   python exp_unc.py
   ```

2. O programa irá:
   - Ler o arquivo de dados
   - Mostrar as dimensões do DataFrame e nomes das colunas
   - Exibir informações extraídas do nome do arquivo (fluidos, direção, inclinação)
   - Plotar as séries temporais das variáveis disponíveis

3. Escolha o tipo de janela de análise:
   - Opção 1: Janela Manual
     - Digite o tempo inicial e final desejados
   - Opção 2: Janela Automática
     - Escolha a variável critério para análise
     - Defina o tamanho mínimo e máximo da janela

4. O programa irá:
   - Calcular as estatísticas para cada variável
   - Gerar gráficos das séries temporais
   - Calcular e plotar Alpha
   - Calcular e plotar os gradientes de pressão
   - Salvar os resultados em um arquivo de texto

## Arquivos de Saída

INSTALAR PRÉ-REQUISITOS:
   python install_requirements.py

O programa gera os seguintes arquivos no mesmo diretório do arquivo de entrada:

1. `[nome_arquivo]_tratado.txt`:
   - Contém os resultados da análise
   - Inclui estatísticas das variáveis
   - Apresenta os dados da janela selecionada

2. Gráficos:
   - `series_full-[nome_arquivo].png`: Séries temporais completas
   - `janelas-[nome_arquivo].png`: Janelas das variáveis
   - `alpha-[nome_arquivo].png`: Fração de vazio
   - `dP_F_dz-[nome_arquivo].png`: Gradientes de pressão

## Observações Importantes

1. Use sempre a barra normal '/' nos caminhos de arquivo
2. O arquivo de entrada deve estar no formato correto com cabeçalho adequado
3. Para janela manual, as estatísticas são calculadas apenas para referência
4. Para janela automática, o programa busca a janela com menor desvio padrão
5. IMPORTANTE: confira sempre que o seu arquivo esteja no formato correto Ex: AWD45P01

## Formato do Nome do Arquivo

O nome do arquivo deve seguir o padrão: `XXX##ID##` onde:
- X: letra indicando o fluido (A:Air, W:Water, O:Oil, S:SF6)
- #: número indicando a inclinação em graus
- ID: identificador do ponto experimental

Exemplo: `AWD45P15` indica:
- A: Air
- W: Water
- D: Downward
- 45: 45 graus
- P15: ID do ponto 