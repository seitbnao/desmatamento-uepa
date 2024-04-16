import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

df = pd.read_csv("municipio_bioma.csv")

numeric_columns = ['ano', 'id_municipio', 'area_total', 'desmatado', 'vegetacao_natural', 'nao_vegetacao_natural', 'hidrografia']
df[numeric_columns] = df[numeric_columns].astype(float)

maraba_df = df[df['id_municipio'] == 1504208]
altamira_df = df[df['id_municipio'] == 1500602]
braganca_df = df[df['id_municipio'] == 1501709]
itaituba_df = df[df['id_municipio'] == 1503606]
redencao_df = df[df['id_municipio'] == 1506138]

desmatamento_altamira = altamira_df.groupby('ano')['desmatado'].sum()
desmatamento_maraba = maraba_df.groupby('ano')['desmatado'].sum()
desmatamento_braganca = braganca_df.groupby('ano')['desmatado'].sum()
desmatamento_itaituba = itaituba_df.groupby('ano')['desmatado'].sum()
desmatamento_redencao = redencao_df.groupby('ano')['desmatado'].sum()

total_desmatamento_maraba = desmatamento_maraba.sum()
total_desmatamento_altamira = desmatamento_altamira.sum()
total_desmatamento_braganca = desmatamento_braganca.sum()
total_desmatamento_itaituba = desmatamento_itaituba.sum()
total_desmatamento_redencao = desmatamento_redencao.sum()

print("Total de desmatamento em Marabá:", total_desmatamento_maraba, "hectares")
print("Total de desmatamento em Altamira:", total_desmatamento_altamira, "hectares")
print("Total de desmatamento em Bragança:", total_desmatamento_braganca, "hectares")
print("Total de desmatamento em Itaituba:", total_desmatamento_itaituba, "hectares")
print("Total de desmatamento em Redenção:", total_desmatamento_redencao, "hectares")

plt.figure(figsize=(10, 6))
plt.plot(desmatamento_maraba.index, desmatamento_maraba.values, marker='o', linestyle='-', label='Marabá (1504208)')
plt.plot(desmatamento_altamira.index, desmatamento_altamira.values, marker='o', linestyle='-', label='Altamira (1500602)')
plt.plot(desmatamento_braganca.index, desmatamento_braganca.values, marker='o', linestyle='-', label='Bragança (1501709)')
plt.plot(desmatamento_itaituba.index, desmatamento_itaituba.values, marker='o', linestyle='-', label='Itaituba (1501709)')
plt.plot(desmatamento_redencao.index, desmatamento_redencao.values, marker='o', linestyle='-', label='Redenção (1506138)')

def plot_regresion(desmatamento, label):
    X = np.array(desmatamento.index).reshape(-1, 1)
    y = desmatamento.values

    model = LinearRegression()
    model.fit(X, y)

    future_years = np.arange(desmatamento.index[-1] + 1, desmatamento.index[-1] + 31).reshape(-1, 1)
    predicted_desmatamento = model.predict(future_years)

    plt.plot(future_years, predicted_desmatamento, linestyle='--', label=f'Previsão de Desmatamento - {label}')

plot_regresion(desmatamento_maraba, 'Marabá (1504208)')
plot_regresion(desmatamento_altamira, 'Altamira (1500602)')
plot_regresion(desmatamento_braganca, 'Bragança (1501709)')
plot_regresion(desmatamento_itaituba, 'Itaituba (1501709)')
plot_regresion(desmatamento_redencao, 'Redenção (1506138)')

plt.title('Desmatamento entre Marabá, Altamira, Bragança. Itaituba e Redenção')
plt.xlabel('Ano')
plt.ylabel('Desmatamento (em hectares)')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
