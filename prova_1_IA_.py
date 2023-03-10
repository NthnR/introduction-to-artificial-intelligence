# -*- coding: utf-8 -*-
"""Prova 1 - IA .ipynb

Base de dados: [Medical Insurance Payout](https://https://www.kaggle.com/datasets/harshsingh2209/medical-insurance-payout)
"""

import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


medicalInsurancePayout = pd.read_csv('/content/drive/MyDrive/Basededados/MedicalInsurancePayout/expenses.csv')
type(medicalInsurancePayout)

medicalInsurancePayout.sample(5)

"""Descrição da tabela"""

medicalInsurancePayout.describe()

"""Verifica se existe atributos vazios e duplicados"""

print(medicalInsurancePayout.isnull().sum())

print("Dados duplicados: ",medicalInsurancePayout.duplicated().sum())

medicalInsurancePayout = medicalInsurancePayout.drop_duplicates(keep = 'first')
print("Apos remoção: ", medicalInsurancePayout.duplicated().sum())

"""Graficos explorando os dados da tabela:

Sexo: 0 - Male, 1 - Female.

Região:1 - southeast, 2 - southwest, 3 - northwest, 4 - northeast.

"""

sns.pairplot(medicalInsurancePayout.sample(50), hue="sex", corner=True, kind='reg')

"""Correlação

"""

medicalInsurancePayout_corr = medicalInsurancePayout.corr()
medicalInsurancePayout_corr.style.background_gradient(cmap='coolwarm')

"""Substituindo os valores das classes fumante, região e sexo para valores inteiros"""

mapeamento = {'southeast': 1, 'southwest': 2, 'northwest': 3, 'northeast': 4}
medicalInsurancePayout = medicalInsurancePayout.replace({'region': mapeamento})
mapeamento = {'no': 0, 'yes': 1}
medicalInsurancePayout = medicalInsurancePayout.replace({'smoker': mapeamento})
mapeamento = {'male': 0, 'female': 1}
medicalInsurancePayout = medicalInsurancePayout.replace({'sex': mapeamento})

"""Verificando se existe valores menores que zero

"""

(medicalInsurancePayout < 0).any().any()

"""Normalizando os atributos das classes de entrada"""

data_normalizado = (medicalInsurancePayout - medicalInsurancePayout.min()) / (medicalInsurancePayout.max() - medicalInsurancePayout.min())

"""Separando as variaveis de entrada e test """

from sklearn.model_selection import train_test_split, cross_val_score

train = data_normalizado.drop('charges', axis= 1)
test = data_normalizado['charges'].values


xTrain, xTest, yTrain, yTest = train_test_split(train, test, test_size=0.3)

print(xTrain)

"""Metodo K vizinhos

Metrica Minkowski
"""

from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics 

knn_3 = KNeighborsRegressor(n_neighbors = 3, metric="minkowski")
knn_6 = KNeighborsRegressor(n_neighbors = 6, metric="minkowski")


knn_3.fit(xTrain,yTrain)
knn_6.fit(xTrain,yTrain)

cutoff = 0.6
predic_knn3 = knn_3.predict(xTest)
predic_knn6 = knn_6.predict(xTest)

yPredic_knn3 = np.zeros_like(predic_knn3)
yPredic_knn3[predic_knn3 > 1] = 1

yPredic_knn6 = np.zeros_like(predic_knn6)
yPredic_knn6[predic_knn6 > 1] = 1

y_test_classes = np.zeros_like(predic_knn3)
y_test_classes[yTest > cutoff] = 1

print("Para k = 3, temos um coeficiente de determinação de %0.2f" % (knn_3.score(xTrain,yTrain)))
print("Matriz de confusão:")
matrizDeConfusao = metrics.confusion_matrix(y_test_classes,yPredic_knn3)
print(matrizDeConfusao)
print("Acuracia: %0.2f " %(metrics.accuracy_score(y_test_classes,yPredic_knn3)))
print("Especificidade: %0.2f" % (matrizDeConfusao[1][0]/(matrizDeConfusao[1][0] + matrizDeConfusao[0][1])))
print("Sensibilidade: %0.2f" % (matrizDeConfusao[0][0]/(matrizDeConfusao[0][0] + matrizDeConfusao[1][1])))

y_test_classes = np.zeros_like(predic_knn6)
y_test_classes[yTest > cutoff] = 1

print("\nPara k = 6, temos um coeficiente de determinação de %0.2f" % (knn_6.score(xTrain,yTrain)))
print("Matriz de confusão:")
matrizDeConfusao = metrics.confusion_matrix(y_test_classes,yPredic_knn6)
print(matrizDeConfusao)
print("Acuracia: %0.2f " %(metrics.accuracy_score(y_test_classes,yPredic_knn6)))
print("Especificidade: %0.2f" % (matrizDeConfusao[1][0]/(matrizDeConfusao[1][0] + matrizDeConfusao[0][1])))
print("Sensibilidade: %0.2f" % (matrizDeConfusao[0][0]/(matrizDeConfusao[0][0] + matrizDeConfusao[1][1])))

"""Metrica Cityblock"""

knn_3 = KNeighborsRegressor(n_neighbors = 3, metric="cityblock", p=2)
knn_6 = KNeighborsRegressor(n_neighbors = 6, metric="cityblock", p=2)


knn_3.fit(xTrain,yTrain)
knn_6.fit(xTrain,yTrain)

cutoff = 0.6
predic_knn3 = knn_3.predict(xTest)
predic_knn6 = knn_6.predict(xTest)

yPredic_knn3 = np.zeros_like(predic_knn3)
yPredic_knn3[predic_knn3 > 1] = 1

yPredic_knn6 = np.zeros_like(predic_knn6)
yPredic_knn6[predic_knn6 > 1] = 1

y_test_classes = np.zeros_like(predic_knn3)
y_test_classes[yTest > cutoff] = 1

print("Para k = 3, temos um coeficiente de determinação de %0.2f" % (knn_3.score(xTrain,yTrain)))
print("Matriz de confusão:")
matrizDeConfusao = metrics.confusion_matrix(y_test_classes,yPredic_knn3)
print(matrizDeConfusao)
print("Acuracia: %0.2f " %(metrics.accuracy_score(y_test_classes,yPredic_knn3)))
print("Especificidade: %0.2f" % (matrizDeConfusao[1][0]/(matrizDeConfusao[1][0] + matrizDeConfusao[0][1])))
print("Sensibilidade: %0.2f" % (matrizDeConfusao[0][0]/(matrizDeConfusao[0][0] + matrizDeConfusao[1][1])))

y_test_classes = np.zeros_like(predic_knn6)
y_test_classes[yTest > cutoff] = 1

print("\nPara k = 6, temos um coeficiente de determinação de %0.2f" % (knn_6.score(xTrain,yTrain)))
print("Matriz de confusão:")
matrizDeConfusao = metrics.confusion_matrix(y_test_classes,yPredic_knn6)
print(matrizDeConfusao)
print("Acuracia: %0.2f " %(metrics.accuracy_score(y_test_classes,yPredic_knn6)))
print("Especificidade: %0.2f" % (matrizDeConfusao[1][0]/(matrizDeConfusao[1][0] + matrizDeConfusao[0][1])))
print("Sensibilidade: %0.2f" % (matrizDeConfusao[0][0]/(matrizDeConfusao[0][0] + matrizDeConfusao[1][1])))
