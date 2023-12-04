import cv2
import csv
import glob
import numpy as np
from sklearn import tree
from joblib import dump, load

from hu_moments_generation import generate_hu_moments_file

file_name = "shapes-hu-moments.csv"
trainData = []
trainLabels = []

#Separa los valores de hu y las etiquetas en 2 arrays
def divide_moments_labels():
    global trainData
    global trainLabels
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            class_label = row.pop()  # saca el ultimo elemento de la lista
            floats = []
            for n in row:
                floats.append(float(n))  # tiene los momentos de Hu transformados a float.
            trainData.append(np.array(floats, dtype=np.float32))  # momentos de Hu
            trainLabels.append(np.array([class_label], dtype=np.int32))  # Resultados
            # Valores y resultados se necesitan por separados
    trainData = np.array(trainData, dtype=np.float32)
    trainLabels = np.array(trainLabels, dtype=np.int32)


# transforma los arrays a arrays de forma numpy


#Define las etiquetas del archivo respecto a la figura que representan
def read_labels(l):
    if l == 1:
        return "star"
    if l == 2:
        return "square"
    if l == 3:
        return "triangle"
    if l == 4:
        return "invalid"



def get_trained_model():
    divide_moments_labels()
    tree = cv2.ml.DTrees_create()
    tree.setCVFolds(1)
    tree.setMaxDepth(10)
    tree.train(trainData, cv2.ml.ROW_SAMPLE, trainLabels)
    return tree

labels = []
moments = []

def divide_MandL_alt():
    with open(file_name) as file:
        reader = csv.reader(file)
        for row in reader:
            labels.append(row.pop())
            moments.append(row)
    return moments, labels



def train_model():
    x, y = divide_MandL_alt()
    clasificador = tree.DecisionTreeClassifier().fit(x, y)
#    tree.plot_tree(clasificador)
    dump(clasificador, 'filename.train_proy2')



def get_trained_model_alt():
    return load("filename.train_proy2")


get_trained_model_alt()