import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Ler os dados
df_confrontos = pd.read_excel('C:/Users/Allan/OneDrive/Área de Trabalho/fut/tabela_ok.xlsx', sheet_name='confrontos')

# Dividir a coluna 'resultado' em 'gols_mandante' e 'gols_visitante'
df_confrontos['gols_mandante'] = df_confrontos['resultado'].apply(lambda x: int(x.split('-')[0]))
df_confrontos['gols_visitante'] = df_confrontos['resultado'].apply(lambda x: int(x.split('-')[1]))

# Remover a coluna 'resultado'
df_confrontos.drop('resultado', axis=1, inplace=True)

# Ler o ranking
df_ranking = pd.read_excel('C:/Users/Allan/OneDrive/Área de Trabalho/fut/tabela_ok.xlsx', sheet_name='Tabela simples')
df_ranking.columns = [str(col) + '_ranking' for col in df_ranking.columns]
df_ranking.rename(columns={'Equipe_ranking': 'Equipe'}, inplace=True)

# Fazendo o join com base no nome das equipes mandante e visitante
df_confrontos = pd.merge(df_confrontos, df_ranking, left_on='mandante', right_on='Equipe', how='left')
df_confrontos = pd.merge(df_confrontos, df_ranking, left_on='visitante', right_on='Equipe', how='left', suffixes=('_mandante', '_visitante'))

# Remover as colunas de Equipe que foram adicionadas pelo merge
df_confrontos.drop(['Equipe_mandante', 'Equipe_visitante'], axis=1, inplace=True)

# Codificar colunas categóricas
df_encoded = pd.get_dummies(df_confrontos, columns=['data', 'hora', 'mandante', 'visitante', 'fase'])

# Separar as features e o target
X = df_encoded.drop(['gols_mandante', 'gols_visitante'], axis=1)
y_mandante = df_confrontos['gols_mandante']
y_visitante = df_confrontos['gols_visitante']

# Dividir os dados em treino e teste
X_train, X_test, y_mandante_train, y_mandante_test = train_test_split(X, y_mandante, test_size=0.2, random_state=42)
_, _, y_visitante_train, y_visitante_test = train_test_split(X, y_visitante, test_size=0.2, random_state=42)

# Treinar o modelo para gols do mandante
model_mandante = RandomForestRegressor(random_state=42)
model_mandante.fit(X_train, y_mandante_train)

# Treinar o modelo para gols do visitante
model_visitante = RandomForestRegressor(random_state=42)
model_visitante.fit(X_train, y_visitante_train)

print("Modelo para gols do mandante treinado com sucesso!")
print("Modelo para gols do visitante treinado com sucesso!")
# Capturar dados do novo jogo via console
data = input("Digite a data do jogo (ex: 2023-04-15): ")
hora = input("Digite a hora do jogo (ex: 16:00): ")
mandante = input("Digite o nome do time mandante: ")
visitante = input("Digite o nome do time visitante: ")
fase = input("Digite a fase do jogo (ex: R1): ")

# Criar um DataFrame com os novos dados
new_data = pd.DataFrame({'data': [data], 'hora': [hora], 'mandante': [mandante], 'visitante': [visitante], 'fase': [fase]})

# Aplicar o mesmo pré-processamento que foi feito nos dados de treinamento
new_data_encoded = pd.get_dummies(new_data, columns=['data', 'hora', 'mandante', 'visitante', 'fase'])

# Certificar-se de que todas as colunas que estão no modelo de treinamento também existem no novo DataFrame
missing_cols = set(X_train.columns) - set(new_data_encoded.columns)

# Criar um DataFrame para as colunas faltantes e preencher com 0
missing_cols_df = pd.DataFrame(0, index=[0], columns=list(missing_cols))


# Concatenar o DataFrame original com o DataFrame das colunas faltantes
new_data_encoded = pd.concat([new_data_encoded, missing_cols_df], axis=1)

# Ordenar as colunas para que correspondam ao modelo de treinamento
new_data_encoded = new_data_encoded[X_train.columns]

# Fazer previsões
gols_mandante_pred = model_mandante.predict(new_data_encoded)
gols_visitante_pred = model_visitante.predict(new_data_encoded)

print(f"Previsão de gols do mandante: {gols_mandante_pred[0]}")
print(f"Previsão de gols do visitante: {gols_visitante_pred[0]}")