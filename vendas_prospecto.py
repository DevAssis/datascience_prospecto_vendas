# Carregar as bibliotecas
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# Importação e tratamento de dados
df = pd.read_csv('advertising.csv')
print(df)
print(df.info())

# Análise e correlação de dados
mask = np.triu(np.ones_like(df.corr(), dtype=bool))
cmap = sns.color_palette("Spectral", as_cmap=True)
sns.heatmap(df.corr(), mask=mask, cmap=cmap, annot=True, vmax=.3, center=0,
            linewidths=.5, cbar_kws={"shrink": .5})
plt.show()

# Preparação dos Dados para treinar o Modelo ML

y = df['Vendas']
x = df.drop('Vendas', axis=1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3)

# Criando os modelos A.I
modelo_regressaolinear = LinearRegression()
modelo_randomforest = RandomForestRegressor()

# Treinando as A.I
modelo_regressaolinear.fit(x_train, y_train)
modelo_randomforest.fit(x_train, y_train)

# Teste da AI e visualizão para comparar a IA
previsao_regressaolinear = modelo_regressaolinear.predict(x_test)
previsao_randomforest = modelo_randomforest.predict(x_test)

r2_rl = metrics.r2_score(y_test, previsao_regressaolinear)
r2_rf = metrics.r2_score(y_test, previsao_randomforest)
print(f'R² da Regressão Linear: {r2_rl:.2%}')
print(f'R² da Árvore de Decisão: {r2_rf:.2%}')

# Visualização gráfica
df_grafico = pd.DataFrame()
df_grafico['y_teste'] = y_test
df_grafico['Regressão Linear:'] = previsao_regressaolinear
df_grafico['Árvore de Decisão:'] = previsao_randomforest

plt.figure(figsize=(10, 3))
sns.lineplot(data=df_grafico)
plt.show()

# importa a nova tabela com informações de propaganda
df_nova = pd.read_csv('novos.csv')
nova_df = modelo_randomforest.predict(df_nova)

print(df_nova)
print('Previsão de Vendas conforme investimentos acima:')
print(nova_df)
