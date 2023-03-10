# -*- coding: utf-8 -*-
"""
Estimador de taxa de aprovação de alunos do ensino fundamental e médio

link para a base de dados:  https://basedosdados.org/dataset/br-inep-ideb?bdm_table=uf_ano_escolar

"""

import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import neighbors 
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import classification_report,  mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import v_measure_score

df = pd.read_csv('/content/drive/MyDrive/Basededados/TaxaDeAprovação/uf_ano_escolar.csv')
type(df)

"""Pré-processamento dos dados

Todos os random_state estão setados como 10
"""

print("Tamanho do banco de dados antes do pré-processamento:", len(df.index))

"""Verificar se existe dados duplicados ou vazios"""

qtdDuplicados = df.duplicated().sum()
print(" quantidade de dados duplicados: ", qtdDuplicados)

if( qtdDuplicados > 0):
    abalone = df.drop_duplicates(keep = 'first')
    print(" Após remoção: ",df.duplicated().sum())

print(" quantidade de dados vazios:\n",df.isnull().sum())
print("\n")
df = df.dropna()

print("Após remoção de dados vazios:\n", df.isnull().sum())

"""Removendo o calculo de media total dos estados da base de dados"""

df.drop(df[df['rede'] == "total"].index, inplace = True)

"""Substituindo os valores de sigla_uf para valores inteiros:

[AC - 0]
&emsp;[AL - 1]
&emsp;[AP - 2]
&emsp;[AM - 3]
&emsp;[BA - 4] \\
[CE - 5]
&emsp;[DF - 6]
&emsp;[ES - 7]
&emsp;[GO - 8]
&emsp;[MA - 9]

[MT - 10]
&emsp;[MS - 11]
&emsp;[MG - 12]
&emsp;[PA - 13]
&emsp;[PB - 14]

[PR - 15]
&emsp;[PE - 16]
&emsp;[PI - 17]
&emsp;[RJ - 18]
&emsp;[RN - 19]
&emsp;[RS - 20]

[RO - 21]
&emsp;[RR - 22]
&emsp;[SC - 23]
&emsp;[SP - 24]
&emsp;[SE - 25]
&emsp;[TO - 26]
"""

mapeamento = {'AC': 0,'AL': 1,'AP': 2,'AM': 3,'BA': 4,'CE': 5,'DF' : 6,'ES': 7,'GO': 8,'MA' : 9,'MT': 10,'MS': 11,'MG': 12,'PA':13,
'PB': 14,'PR': 15,'PE': 16,'PI': 17,'RJ': 18,'RN': 19,'RS': 20,'RO': 21, 'RR': 22,'SC': 23,'SP': 24,'SE': 25,'TO': 26}

df = df.replace({'sigla_uf': mapeamento})

"""Substituindo os valores da coluna rede para valores inteiros: \\

estadual - 0 \\
privada - 1  \\
publica - 2


"""

mapeamento = {'estadual': 0, 'privada': 1, 'publica': 2}

df = df.replace({'rede': mapeamento})

"""Substituindo os valores da coluna ensino para valores inteiros: \\

fundamental - 0 \\
medio - 1  \\
"""

mapeamento = {'fundamental': 0, 'medio': 1}

df = df.replace({'ensino': mapeamento})

"""Verificar se existe valores negativos"""

( df < 0).any().any()

"""Verificar se existe alguma taxa de aprovação maior que 100%"""

( df > 100).taxa_aprovacao.any()

print("Tamanho do banco de dados após o pré-processamento:", len(df.index))

"""Normalizando"""

df_normalizado = ( df - df.min() ) / ( df.max() - df.min() )

"""Selecionando as variaveis de treino e test para metodo de regressão"""

train = df_normalizado.drop('taxa_aprovacao', axis= 1)
test = df_normalizado['taxa_aprovacao'].copy()

xTrain_reg, xTest_reg, yTrain_reg, yTest_reg = train_test_split(train, test, test_size=0.3, random_state = 10)

"""Selecionando as variaveis de treino e test para metodo de classificação"""

train = df_normalizado.drop('taxa_aprovacao', axis= 1)
test = df['taxa_aprovacao'].copy()

test.loc[df['taxa_aprovacao'] < 10] = 0
test.loc[(df['taxa_aprovacao'] >= 10) & (df['taxa_aprovacao'] < 20) ] = 1
test.loc[(df['taxa_aprovacao'] >= 20) & (df['taxa_aprovacao'] < 30) ] = 2
test.loc[(df['taxa_aprovacao'] >= 30) & (df['taxa_aprovacao'] < 40) ] = 3
test.loc[(df['taxa_aprovacao'] >= 40) & (df['taxa_aprovacao'] < 50) ] = 4
test.loc[(df['taxa_aprovacao'] >= 50) & (df['taxa_aprovacao'] < 60) ] = 5
test.loc[(df['taxa_aprovacao'] >= 60) & (df['taxa_aprovacao'] < 70) ] = 6
test.loc[(df['taxa_aprovacao'] >= 70) & (df['taxa_aprovacao'] < 80) ] = 7
test.loc[(df['taxa_aprovacao'] >= 80) & (df['taxa_aprovacao'] < 90) ] = 8
test.loc[(df['taxa_aprovacao'] >= 90) & (df['taxa_aprovacao'] < 99) ] = 9
test.loc[(df['taxa_aprovacao'] >= 99) ] = 10

xTrain_cla, xTest_cla, yTrain_cla, yTest_cla = train_test_split(train, test, test_size=0.3, random_state = 10)

"""Métodos de aprendizado:

K vizinhos
"""

knn_3_mink = neighbors.KNeighborsClassifier(n_neighbors = 3, metric="minkowski")
knn_7_mink = neighbors.KNeighborsClassifier(n_neighbors = 7, metric="minkowski")

knn_3_manh = neighbors.KNeighborsClassifier(n_neighbors = 3, metric="manhattan")
knn_7_manh = neighbors.KNeighborsClassifier(n_neighbors = 7, metric="manhattan")

knn_3_city = neighbors.KNeighborsClassifier(n_neighbors = 3, metric="cityblock")
knn_7_city = neighbors.KNeighborsClassifier(n_neighbors = 7, metric="cityblock")

knn_3_mink.fit(xTrain_cla,yTrain_cla)
knn_3_mink.fit(xTrain_cla,yTrain_cla)
predict_knn_3_mink = knn_3_mink.predict(xTest_cla)
predict_knn_7_mink = knn_3_mink.predict(xTest_cla)

knn_3_manh.fit(xTrain_cla,yTrain_cla)
knn_3_manh.fit(xTrain_cla,yTrain_cla)
predict_knn_3_manh = knn_3_manh.predict(xTest_cla)
predict_knn_7_manh = knn_3_manh.predict(xTest_cla)

knn_3_city.fit(xTrain_cla,yTrain_cla)
knn_7_city.fit(xTrain_cla,yTrain_cla)
predict_knn_3_city = knn_3_city.predict(xTest_cla)
predict_knn_7_city = knn_7_city.predict(xTest_cla)

print("Método minkowski")
print("k = 3 Acurácia: %0.3f" % (metrics.accuracy_score(yTest_cla, predict_knn_3_mink)))
print("k = 7 Acurácia: %0.3f" % (metrics.accuracy_score(yTest_cla, predict_knn_7_mink)))
print("")
print("Método manhattan")
print("k = 3 Acurácia: %0.3f" % (metrics.accuracy_score(yTest_cla, predict_knn_3_manh)))
print("k = 7 Acurácia: %0.3f" % (metrics.accuracy_score(yTest_cla, predict_knn_7_manh)))
print("")
print("Método cityblock")
print("k = 3 Acurácia: %0.3f" % (metrics.accuracy_score(yTest_cla, predict_knn_3_city)))
print("k = 7 Acurácia: %0.3f" % (metrics.accuracy_score(yTest_cla, predict_knn_7_city)))

"""Hierárquico: \\
Com clusters igual a 11, em cada cluster possui um intervalo da taxa de aprovação. \\
[0:10) [10:20) [20:30) [30:40) [40:50) 
[50:60)   
[60:70) [70:80) [80:90) [90:99)
[99:100]
"""

ward_linkage = AgglomerativeClustering(n_clusters = 11, linkage="ward").fit(xTrain_cla,yTrain_cla)
single_linkage = AgglomerativeClustering(n_clusters = 11, linkage="single").fit(xTrain_cla,yTrain_cla)
average_linkage = AgglomerativeClustering(n_clusters = 11, linkage="average").fit(xTrain_cla,yTrain_cla)
complete_linkage = AgglomerativeClustering(n_clusters = 11, linkage="complete").fit(xTrain_cla,yTrain_cla)

predict_ward = ward_linkage.fit_predict(xTest_cla,yTest_cla)
predict_single = single_linkage.fit_predict(xTest_cla,yTest_cla)
predict_average = average_linkage.fit_predict(xTest_cla,yTest_cla)
predict_complete = complete_linkage.fit_predict(xTest_cla,yTest_cla)

measure_ward = v_measure_score(yTest_cla,predict_ward)
measure_single = v_measure_score(yTest_cla,predict_single)
measure_average = v_measure_score(yTest_cla,predict_average)
measure_complete = v_measure_score(yTest_cla,predict_complete)

print("Ward linkage")
print("Acurácia: %0.3f" % (metrics.accuracy_score(yTest_cla, predict_ward)))
print("V measure score: %0.3f" % (measure_ward))
print("")
print("Single linkage")
print("Acurácia: %0.3f" % (metrics.accuracy_score(yTest_cla, predict_single)))
print("V measure score: %0.3f" % (measure_single))
print("")
print("Average linkage")
print("Acurácia: %0.3f" % (metrics.accuracy_score(yTest_cla, predict_average)))
print("V measure score: %0.3f" % (measure_average))
print("")
print("Complete linkage")
print("Acurácia: %0.3f" % (metrics.accuracy_score(yTest_cla, predict_complete)))
print("V measure score: %0.3f" % (measure_complete))

"""Árvore de decisão"""

arvoreD2 = DecisionTreeRegressor(max_depth=2, random_state=10)
arvoreD2.fit(xTrain_reg,yTrain_reg)

arvoreD5 = DecisionTreeRegressor(max_depth=5, random_state=10)
arvoreD5.fit(xTrain_reg,yTrain_reg)

errorD2_treino = mean_squared_error(yTrain_reg, arvoreD2.predict(xTrain_reg), squared=False)
errorD2_test = mean_squared_error(yTest_reg, arvoreD2.predict(xTest_reg), squared=False)

print("Hiperparâmetro: Profundidade = 2")
print("Erro médio do treinamento: %0.3f" % (errorD2_treino))
print("Erro médio do teste: %0.3f" % (errorD2_test))
print("")

errorD5_treino = mean_squared_error(yTrain_reg, arvoreD5.predict(xTrain_reg), squared=False)
errorD5_test = mean_squared_error(yTest_reg, arvoreD5.predict(xTest_reg), squared=False)
print("Hiperparâmetro: Profundidade = 5")
print("Erro médio do treinamento: %0.3f" % (errorD2_treino))
print("Erro médio do teste: %0.3f" % (errorD5_test))

"""MLP"""

mpl = MLPRegressor(random_state=10, max_iter=200).fit(xTrain_reg, yTrain_reg)

error_treino = mean_squared_error(yTrain_reg, mpl.predict(xTrain_reg), squared=False)
error_test = mean_squared_error(yTest_reg, mpl.predict(xTest_reg), squared=False)

print("Erro médio do treinamento: %0.3f" % (error_treino))
print("Erro médio do teste: %0.3f" % (error_test))
