file_path = 'example/AWD45/AWD45ID04/AWD45ID04'
variavel_criterio = 'J Ar'          # Escolha a variável de critério para realizar o janelamento - Digite o nome exato da coluna para análise
window_min = 100                    # Digite o tamanho mínimo da janela em segundos
window_max = 150                    # Digite o tamanho máximo da janela em segundos

### Colunas de interesse
colunas_analise =       ['PIT-M-0101',
                        'PDT-M-0101-40kPa_mA',
                        'PDT-M-0101B-10kPa_mA',
                        'PDT-M-0101C-3kPa_mA',
                        'PDT-M-0101D-30Kpa_mA',
                        'PDT-M-0101D-30Kpa_mA_tara',
                        'TIT-M-0101',
                        'Densitometro',
                        'J Ar',
                        'J Água']

### Cálculo de propagação de incerteza ###
# Valores de calibração do densitômetro
I_g = 275191.0                      # Insira a intensidade padrão para o gás (Calibração do densitômetro)
I_l = 164175.5                      # Insira a intensidade padrão para o líquido (Calibração do densitômetro)

#
E_PIT_M_0101 = 0.2                  # Insira o erro de medição do sensor de pressão
E_PDT_M_0101_40kPa = 0.1            # Insira o erro de medição do diferencial de pressão de 40kPa
E_PDT_M_0101B_10kPa = 0.1           # Insira o erro de medição do diferencial de pressão de 10kPa
E_PDT_M_0101C_3kPa = 0.1            # Insira o erro de medição do diferencial de pressão de 3kPa

E_TIT_M_0101 = 0.1                  # Insira o erro de medição do sensor de temperatura

E_Densitometro = 0.1

E_J_Ar = 0.1                          # Insira o erro de medição do sensor de vazão
E_J_Agua = 0.1                        # Insira o erro de medição do sensor de vazão