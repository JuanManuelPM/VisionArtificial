import cv2
import csv
import glob
import numpy as np
from sklearn import tree
from joblib import dump, load

from hu_moments_generation import hu_moments_of_file

"""el archivo de fabri ya tiene los momentos de hu aplicados el logaritmo, voy a reescribirlo para adaptar los datos
1. 5-point-star
2. rectangle
3. triangle

path: "C:/Users/martu/OneDrive/Escritorio/Tercero/VisArt/visionArtificial-master/machine/generated-files/shapes-hu-moments.csv"
"""


def write_hu_moments_alt(label, writer):
    files = glob.glob('./shapes/' + label + '/*')  # label recibe el nombre de la carpeta
    hu_moments = []
    for file in files:
        hu_moments.append(hu_moments_of_file(file)) #uso el metodo de fabri por la practicidad y ya aplica el log el
    for mom in hu_moments:
        flattened = mom.ravel()  # paso de un array de arrays a un array simple.
        row = np.append(flattened, label)  # le metes el flattened array y le agregas el label
        writer.writerow(row)  # Escribe una linea en el archivo.


def generate_hu_moments_file_alt():
    with open('./shapes-hu-moments-number.csv', 'w',
              newline='') as file:  # Se genera un archivo nuevo (W=Write)
        writer = csv.writer(file)
        write_hu_moments_alt("1", writer)
        write_hu_moments_alt("2", writer)
        write_hu_moments_alt("3", writer)


file_name = "./shapes-hu-moments-number.csv"


# LISTO EL GENERADOR
# ahora vamos con el entrenador y despues el clasificador

trainData = []
trainLabels = []
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



def read_labels(l):
    if l == 1:
        return "5-point-star"
    if l == 2:
        return "rectangle"
    if l == 3:
        return "triangle"



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

#train_model()


def get_trained_model_alt():
    return load("filename.train_proy2")


get_trained_model_alt()