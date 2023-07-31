import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import procrustes

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

# Extract the rotation matrix and translation vector from your solution
rotation_matrix = solution[:9].reshape(3, 3)
translation_vector = solution[9:]

# Perform the Procrustes analysis to enforce the proper rotation constraints
U, _, Vt = np.linalg.svd(rotation_matrix)
proper_rotation_matrix = np.dot(U, Vt)

# Reconstruct the homogeneous transformation matrix with the proper rotation and translation
homog_matrix = np.zeros((4, 4))
homog_matrix[:3, :3] = proper_rotation_matrix
homog_matrix[:3, 3] = translation_vector
homog_matrix[3, 3] = 1

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
