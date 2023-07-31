import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Meassured points
measuredPoints = np.array([
    [611.076, 523.954, 278.609],
    [663.378, 716.959, 284.796],
    [712.111, 934.086, 255.697],
    [433.926, 1216.223, 255.779],
    [460.329, 1641.783, 255.694],
    [550.137, 1612.65, 613.95],
    [629.971, 1701.17, 664.328]
])

#Plane points
planePoints = np.array([
    [0, 0, 0],
    [199.885, 0, 0],
    [420.68, 10.804, -35.883],
    [619.358, 352.94, -34.728],
    [1033.714, 438.916, -47.298],
    [1040.679, 343.745, 308.735],
    [1149.313, 290.046, 354.541]
])



matrix_12x12 = np.array([
        #r1                                                 #r2                                                         #r3                                             #t1
    [np.sum(2*(planePoints[:, 0]) ** 2), np.sum(2*(planePoints[:, 0])*(planePoints[:, 2])), np.sum(2*(planePoints[:, 0])*(planePoints[:, 1])), 0, 0, 0, 0, 0, 0, np.sum(2*(planePoints[:, 0])), 0, 0],
    [np.sum(2*(planePoints[:, 0])*(planePoints[:, 2])), np.sum(2*(planePoints[:, 2]) ** 2), np.sum(2*(planePoints[:, 2])*(planePoints[:, 1])), 0, 0, 0, 0, 0, 0, np.sum(2*(planePoints[:, 2])), 0, 0],
    [np.sum(2*(planePoints[:, 0])*(planePoints[:, 1])), np.sum(2*(planePoints[:, 2])*(planePoints[:, 1])), np.sum(2*(planePoints[:, 1]) ** 2), 0, 0, 0, 0, 0, 0, np.sum(2*(planePoints[:, 1])), 0, 0],
                    #r4                                                 #r5                                                         #r6                                     #t2
    [0, 0, 0, np.sum(2*(planePoints[:, 0]) ** 2), np.sum(2*(planePoints[:, 0])*(planePoints[:, 2])), np.sum(2*(planePoints[:, 0])*(planePoints[:, 1])), 0, 0, 0, 0, np.sum(2*(planePoints[:, 0])), 0],
    [0, 0, 0, np.sum(2*(planePoints[:, 0])*(planePoints[:, 2])), np.sum(2*(planePoints[:, 2]) ** 2), np.sum(2*(planePoints[:, 2])*(planePoints[:, 1])), 0, 0, 0, 0, np.sum(2*(planePoints[:, 2])), 0],
    [0, 0, 0, np.sum(2*(planePoints[:, 0])*(planePoints[:, 1])), np.sum(2*(planePoints[:, 2])*(planePoints[:, 1])), np.sum(2*(planePoints[:, 1]) ** 2), 0, 0, 0, 0, np.sum(2*(planePoints[:, 1])), 0],
                                #r7                                                 #r8                                                         #r9                             #t3
    [0, 0, 0, 0, 0, 0, np.sum(2*(planePoints[:, 0]) ** 2), np.sum(2*(planePoints[:, 0])*(planePoints[:, 2])), np.sum(2*(planePoints[:, 0])*(planePoints[:, 1])), 0, 0, np.sum(2*(planePoints[:, 0]))],
    [0, 0, 0, 0, 0, 0, np.sum(2*(planePoints[:, 0])*(planePoints[:, 2])), np.sum(2*(planePoints[:, 2]) ** 2), np.sum(2*(planePoints[:, 2])*(planePoints[:, 1])), 0, 0, np.sum(2*(planePoints[:, 2]))],
    [0, 0, 0, 0, 0, 0, np.sum(2*(planePoints[:, 0])*(planePoints[:, 1])), np.sum(2*(planePoints[:, 2])*(planePoints[:, 1])), np.sum(2*(planePoints[:, 1]) ** 2), 0, 0, np.sum(2*(planePoints[:, 1]))],
    
    [np.sum(2*(planePoints[:, 0])), np.sum(2*(planePoints[:, 2])), np.sum(2*(planePoints[:, 1])), 0, 0, 0, 0, 0, 0, 2, 0, 0],
    [0, 0, 0, np.sum(2*(planePoints[:, 0])), np.sum(2*(planePoints[:, 2])), np.sum(2*(planePoints[:, 1])), 0, 0, 0, 0, 2, 0],
    [0, 0, 0, 0, 0, 0, np.sum(2*(planePoints[:, 0])), np.sum(2*(planePoints[:, 2])), np.sum(2*(planePoints[:, 1])), 0, 0, 2]
])


# Calculate the inverse of the matrix
inv_matrix = np.linalg.inv(matrix_12x12)

# Define your independent terms
independent_terms = np.array([
                            np.sum(2 * (planePoints[:, 0]) * measuredPoints[:, 0]),
                            np.sum(2 * (planePoints[:, 2]) * measuredPoints[:, 0]),
                            np.sum(2 * (planePoints[:, 1]) * measuredPoints[:, 0]),
                            np.sum(2 * (planePoints[:, 0]) * measuredPoints[:, 1]),
                            np.sum(2 * (planePoints[:, 2]) * measuredPoints[:, 1]),
                            np.sum(2 * (planePoints[:, 1]) * measuredPoints[:, 1]),
                            np.sum(2 * (planePoints[:, 0]) * measuredPoints[:, 2]),
                            np.sum(2 * (planePoints[:, 2]) * measuredPoints[:, 2]),
                            np.sum(2 * (planePoints[:, 1]) * measuredPoints[:, 2]),
                            np.sum(2 * measuredPoints[:, 0]),
                            np.sum(2 * measuredPoints[:, 1]),
                            np.sum(2 * measuredPoints[:, 2]), 
                            ])
solution = inv_matrix @ independent_terms

# null_space_matrix will be a NumPy array containing the vectors representing the null space.


homog_matrix = [    [solution[0], solution[1], solution[2], solution[9] ],
                    [solution[3], solution[4], solution[5], solution[10]],
                    [solution[6], solution[7], solution[8], solution[11]],
                    [0          , 0          , 0          , 1           ]
                ]

# # Convertir la lista a un arreglo NumPy para facilitar el cálculo
# homog_matrix = np.array(homog_matrix)

# # Extraer las primeras tres columnas (vectores) de la matriz
# v1 = homog_matrix[:3, 0]
# v2 = homog_matrix[:3, 1]
# v3 = homog_matrix[:3, 2]

# # Normalizar los vectores dividiéndolos por su magnitud (longitud)
# v1_normalized = v1 / np.linalg.norm(v1)
# v2_normalized = v2 / np.linalg.norm(v2)
# v3_normalized = v3 / np.linalg.norm(v3)

# # Asignar los vectores normalizados de vuelta a la matriz
# homog_matrix[:3, 0] = v1_normalized
# homog_matrix[:3, 1] = v2_normalized
# homog_matrix[:3, 2] = v3_normalized

print("Matrix Homogenea:")
print(homog_matrix)

transformedPoints = []

for vector_3d in measuredPoints:
    vector_4d = np.append(vector_3d, 1)
    result_vector = np.dot(homog_matrix, vector_4d)
    new_vector_3d = result_vector[:3]
    transformedPoints.append(new_vector_3d)

transformedPoints = np.array(transformedPoints)
    
# Grafico
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot puntos medidos
ax.scatter(measuredPoints[:, 0], measuredPoints[:, 1], measuredPoints[:, 2], c='blue', label='Puntos medidos')

# Plot puntos ideales
ax.scatter(planePoints[:, 0], planePoints[:, 1], planePoints[:, 2], c='green', label='Puntos ideales')

# Plot puntos transformados
ax.scatter(transformedPoints[:, 0], transformedPoints[:, 1], transformedPoints[:, 2], c='red', label='Puntos transformados')

# Set plot labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

# Save plot
plt.savefig('aligned_points_plot.pdf')  
plt.show()
