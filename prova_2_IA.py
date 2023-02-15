# -*- coding: utf-8 -*-
"""Prova 2  - IA.ipynb

---
"""

import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import classification_report,  mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import v_measure_score


abalone = pd.read_csv('/content/drive/MyDrive/Basededados/AgeOfAbalone/abalone.csv')
type(abalone)

"""Todos os random_state estão setados como 10

Questão 1

Pré-processamento dos dados

Verificar se existe dados duplicados ou vazios
"""

qtdDuplicados = abalone.duplicated().sum()
print(" quantidade de dados duplicados: ", qtdDuplicados)

if( qtdDuplicados > 0):
    abalone = abalone.drop_duplicates(keep = 'first')
    print(" Após remoção: ",abalone.duplicated().sum())

print(" quantidade de dados vazios:\n",abalone.isnull().sum())

"""Substituindo os valores da classe sexo para valores inteiros: \\
Masculino - 0 \\
Feminino - 0.5  \\
Infant - 1    \\
"""

mapeamento = {'M': 0, 'F': 0.5, 'I': 1}
abalone = abalone.replace({'sex': mapeamento})

"""Verificando a existencia de valores negativos


"""

( abalone < 0).any().any()

"""Redimencionando atributos: \\
size = ( diameter + height + length ) / 3 \\
weight = ( shucked_wt + viscera_wt + shell_wt ) / 3 \\
"""

size = ( abalone['diameter'].values + abalone['height'].values + abalone['length'].values ) / 3
weight = ( abalone['shucked_wt'].values + abalone['viscera_wt'].values + abalone['shell_wt'].values ) / 3

abalone_size = pd.DataFrame(np.float_(np.array(size)), columns = ['size'])
abalone.reset_index(drop = True, inplace = True)
abalone_size.reset_index(drop = True, inplace = True)
abalone_media = pd.concat([abalone,abalone_size], axis = 1)

abalone_weight = pd.DataFrame(np.float_(np.array(weight)), columns = ['weight'])
abalone.reset_index(drop = True, inplace = True)
abalone_weight.reset_index(drop = True, inplace = True)
abalone_media = pd.concat([abalone_media,abalone_weight], axis = 1)

abalone_media = abalone_media.drop(['diameter','height', 'length', 'shucked_wt', 'viscera_wt','shell_wt'], axis = 1)

"""Normalizando os atributos das classes de entrada"""

data_normalizado = ( abalone - abalone.min() ) / ( abalone.max() - abalone.min() )
data_media_normalizado = ( abalone_media - abalone_media.min() ) / ( abalone_media.max() - abalone_media.min() )

"""Removendo o atributo para o qual será treinado para identificar,  
 Separando as amostras para treino e teste
"""

train = data_normalizado.drop('age', axis= 1)
test = data_normalizado['age'].values

xTrain, xTest, yTrain, yTest = train_test_split(train, test, test_size=0.3, random_state = 10)

train_media = data_media_normalizado.drop('age', axis= 1)
test_media = data_media_normalizado['age'].values

xTrain_media, xTest_media, yTrain_media, yTest_media = train_test_split(train_media, test_media, test_size=0.3, random_state = 10)

"""Árvore de decisão: \\
data 1 = data com todos os atributos  
data 2 = data com a média dos atributos de tamanho, e peso \\
Ambas estão normalizadas

Hiperparâmetro: Profundidade = 2
"""

arvoreD2 = DecisionTreeRegressor(max_depth=2, random_state=10)
arvoreD2.fit(xTrain,yTrain)

arvore_mediaD2 = DecisionTreeRegressor(max_depth=2, random_state=10)
arvore_mediaD2.fit(xTrain_media,yTrain_media)

errorD2 = mean_squared_error(yTest, arvoreD2.predict(xTest), squared=False)
print("Erro médio do treinamento para o data 1: ", errorD2)

error_mediaD2 = mean_squared_error(yTest_media, arvore_mediaD2.predict(xTest_media), squared=False)
print("Erro médio do treinamento para o data 2: ", error_mediaD2)

"""Hiperparâmetro: Profundidade = 6"""

arvoreD6 = DecisionTreeRegressor(max_depth = 6, random_state = 10)
arvoreD6.fit(xTrain,yTrain)

arvore_mediaD6 = DecisionTreeRegressor(max_depth = 6, random_state = 10)
arvore_mediaD6.fit(xTrain_media,yTrain_media)

errorD6 = mean_squared_error(yTest, arvoreD6.predict(xTest), squared=False)
print("Erro médio do treinamento para o data 1: ", errorD6)

error_mediaD6 = mean_squared_error(yTest_media, arvore_mediaD6.predict(xTest_media), squared=False)
print("Erro médio do treinamento para o data 2: ", error_mediaD6)

"""MLP"""

mpl = MLPRegressor(random_state=10, max_iter=200).fit(xTrain, yTrain)

mpl_media = MLPRegressor(random_state=10, max_iter=200).fit(xTrain_media, yTrain_media)

error = mean_squared_error(yTest, mpl.predict(xTest), squared=False)
print("Erro médio do treinamento para o data 1: ", error)

error_media = mean_squared_error(yTest_media, mpl_media.predict(xTest_media), squared=False)
print("Erro médio do treinamento para o data 2: ", error_media)

"""A métrica de erro escolhida foi o erro quadrático médio. \\
O método retorna um valor flutuante onde quanto mais perto de 0, melhor.

Para o método Árvore de decisão, usar todos os atributos ou usar 2 médias que representam os atributos de tamanho e peso do abalone não teve nenhuma diferença significativa.
"""

print("Para a profundidade igual a 2, a diferença entre os erros é igual a ", (errorD2-error_mediaD2))

print("Para a profundidade igual a 6, a diferença entre os erros é igual a ", (errorD6-error_mediaD6))

"""Para o método MLP foi encontrada uma diferença entre os erros maior, mas ainda sim bem baixa."""

print("A diferença entre os erros para o metodo MLP é igual a ", (-error+error_media))

"""Questão 2

Tratando a coluna " age " para que ela esteja no intervalo para o número de clusters, k = 3
"""

train_k3 = data_normalizado.drop('age', axis= 1)
test_k3 = data_normalizado['age'].copy()

test_k3.loc[data_normalizado['age'] < 1/3] = 0
test_k3.loc[(data_normalizado['age'] >= 1/3) & (data_normalizado['age'] < 2/3) ] = 1
test_k3.loc[(data_normalizado['age'] >= 2/3) & (data_normalizado['age'] <= 1) ] = 2

xTrain_k3, xTest_k3, yTrain_k3, yTest_k3 = train_test_split(train_k3, test_k3, test_size=0.3, random_state = 10)

"""K-means

K = 3
"""

kmeans_k3 = KMeans(n_clusters = 3, random_state = 10).fit(xTrain_k3, yTrain_k3)

predict_k3 = kmeans_k3.predict(xTest_k3)

measure_Km3 = v_measure_score(yTest_k3,predict_k3)
print("V measure score = ", measure_Km3)

"""Hierárquico

Clusters = 3
"""

single_linkage_k3 = AgglomerativeClustering(n_clusters = 3, linkage="single").fit(xTrain_k3,yTrain_k3)
average_linkage_k3 = AgglomerativeClustering(n_clusters = 3, linkage="average").fit(xTrain_k3,yTrain_k3)

predict_singlek3 = single_linkage_k3.fit_predict(xTest_k3,yTest_k3)
predict_averagek3 = average_linkage_k3.fit_predict(xTest_k3,yTest_k3)

measure_singlek3 = v_measure_score(yTest_k3,predict_singlek3)
print("V measure score para o single linkage = ", measure_singlek3)
measure_averagek3 = v_measure_score(yTest_k3,predict_averagek3)
print("V measure score para o average linkage = ", measure_averagek3)

"""Tratando a coluna " age " para que ela esteja no intervalo para o número de clusters, k = 5"""

train_k5 = data_normalizado.drop('age', axis= 1)
test_k5 = data_normalizado['age'].copy()

test_k5.loc[data_normalizado['age'] < 1/5] = 0
test_k5.loc[(data_normalizado['age'] >= 1/5) & (data_normalizado['age'] < 2/5) ] = 1
test_k5.loc[(data_normalizado['age'] >= 2/5) & (data_normalizado['age'] < 3/5) ] = 2
test_k5.loc[(data_normalizado['age'] >= 3/5) & (data_normalizado['age'] < 4/5) ] = 3
test_k5.loc[(data_normalizado['age'] >= 4/5) & (data_normalizado['age'] <= 1) ] = 4

xTrain_k5, xTest_k5, yTrain_k5, yTest_k5 = train_test_split(train_k5, test_k5, test_size=0.3, random_state = 10)

"""K-means

K = 5
"""

kmeans_k5 = KMeans(n_clusters = 5, random_state = 10).fit(xTrain_k5, yTrain_k5)

predict_k5 = kmeans_k5.predict(xTest_k5)

measure_Km5 = v_measure_score(yTest_k5,predict_k5)
print("V measure score = ", measure_Km5)

"""Hierárquico

Clusters = 5
"""

single_linkage_k5 = AgglomerativeClustering(n_clusters = 5, linkage="single").fit(xTrain_k5,yTrain_k5)
average_linkage_k5 = AgglomerativeClustering(n_clusters = 5, linkage="average").fit(xTrain_k5,yTrain_k5)

predict_singlek5 = single_linkage_k5.fit_predict(xTest_k5,yTest_k5)
predict_averagek5 = average_linkage_k5.fit_predict(xTest_k5,yTest_k5)

measure_singlek5 = v_measure_score(yTest_k5,predict_singlek5)
print("V measure score para o single linkage = ", measure_singlek5)
measure_averagek5 = v_measure_score(yTest_k5,predict_averagek5)
print("V measure score para o average linkage = ", measure_averagek5)

"""A métrica de avaliação escolhida foi o v_measure_score, onde a metrica retorna uma pontuação em valor flutuante que representa o quão perfeitamente completa está a rotulagem feita. \\

Abaixo uma lista com as melhores pontuações encontradas:
"""

print("Average linkage, com clusters igual a 3. Pontuação igual a ", np.around(measure_averagek3,4))
print("Average linkage, com clusters igual a 5. Pontuação igual a ", np.around(measure_averagek5,4))
print("K-means, com k igual a 3. Pontuação igual a ", np.around(measure_Km3,4))
print("K-means, com k igual a 5. Pontuação igual a ", np.around(measure_Km5,4))
print("Single linkage, com clusters igual a 5. Pontuação igual a ", np.around(measure_singlek5,4))
print("Single linkage, com clusters igual a 3. Pontuação igual a ", np.around(measure_singlek3,4))
