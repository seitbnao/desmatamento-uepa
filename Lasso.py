import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# Carregar os dados
df = pd.read_csv("municipio_bioma.csv")

# Converter as colunas numéricas para tipo float
numeric_columns = ['ano', 'id_municipio', 'area_total', 'desmatado', 'vegetacao_natural', 'nao_vegetacao_natural', 'hidrografia']
df[numeric_columns] = df[numeric_columns].astype(float)

# Definir os municípios
municipios = {
    1504208: 'Marabá',
    1500602: 'Altamira',
    1501709: 'Bragança',
    1503606: 'Itaituba',
    1506138: 'Redenção'
}

plt.figure(figsize=(10, 6))

# Configurar o modelo Lasso
model = Lasso(alpha=0.1)

# Iterar sobre os municípios
for municipio_id, label in municipios.items():
    municipio_df = df[df['id_municipio'] == municipio_id]
    desmatamento = municipio_df.groupby('ano')['desmatado'].sum()
    total_desmatamento = desmatamento.sum()
    print(f"Total de desmatamento em {label}: {total_desmatamento} hectares")
    
    X = np.array(desmatamento.index).reshape(-1, 1)
    y = desmatamento.values

    # Dividir os dados em conjuntos de treino e teste (25% para teste)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)

    # Normalizar os dados
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Treinar o modelo Lasso
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = r2_score(y_test, y_pred)
    print(f"Acurácia do modelo para {label}: {accuracy}")

    # Treinar o modelo com todos os dados
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)

    future_years = np.arange(desmatamento.index[-1] + 1, desmatamento.index[-1] + 31).reshape(-1, 1)
    future_years_scaled = scaler.transform(future_years)
    predicted_desmatamento = model.predict(future_years_scaled)

    plt.plot(desmatamento.index, desmatamento.values, marker='o', linestyle='-', label=f'{label} ({municipio_id})')
    plt.plot(future_years, predicted_desmatamento, linestyle='--', label=f'Previsão de Desmatamento - {label}')

plt.title('Desmatamento entre Marabá, Altamira, Bragança, Itaituba e Redenção')
plt.xlabel('Ano')
plt.ylabel('Desmatamento (em hectares)')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
