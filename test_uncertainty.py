import uncertainties as unc

u_mf = 2 / 400
u_me = 2 / 200
u_delta_t = 0.2/10

m_f = unc.ufloat(400, 2)
m_e = unc.ufloat(200, 2)
delta_t = unc.ufloat(10, 0.2)

m_ponto = (m_f-m_e)/delta_t
print(m_ponto)
print(f"{m_ponto:.3f}")
print(m_ponto.nominal_value)
print(m_ponto.std_dev)





