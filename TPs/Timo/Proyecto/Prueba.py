# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import pyradiomics

# Cargar el archivo DICOM del fantoma
phantom_ds = pydicom.dcmread("phantom.dcm")

# Obtener la imagen del fantoma en formato numpy
phantom_image = phantom_ds.pixel_array

# Crear una lista de proyecciones vacías
projections = []

# Generar proyecciones de imagen del fantoma desde diferentes ángulos
num_angles = 180
angles = np.linspace(0, 180, num_angles, endpoint=False)
for angle in angles:
    # Rotar la imagen del fantoma según el ángulo actual
    rotated_image = pyradiomics.misc.image.rotate_image(phantom_image, angle)
    
    # Simular la adquisición de la proyección
    projection = np.sum(rotated_image, axis=0)
    
    # Agregar la proyección a la lista de proyecciones
    projections.append(projection)

# Convertir la lista de proyecciones en un sinograma
sinogram = np.vstack(projections)

# Mostrar el sinograma
plt.imshow(sinogram, cmap="gray")
plt.title("Sinograma")
plt.xlabel("Ángulo de proyección")
plt.ylabel("Posición axial")
plt.colorbar()
plt.show()
