import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree


def procrustes(source_points, target_points):
    # Calcular las matrices de covarianza y la matriz de procrustes
    source_cov = np.cov(source_points, rowvar=False)
    target_cov = np.cov(target_points, rowvar=False)
    u, _, vt = np.linalg.svd(np.dot(target_cov, source_cov.T))
    rotation_matrix = np.dot(vt.T, u.T)

    # Calcular el vector de traslación
    translation_vector = np.mean(target_points, axis=0) - np.mean(np.dot(source_points, rotation_matrix.T), axis=0)

    return rotation_matrix, translation_vector


def objective_function(params, source_points, target_points):
    # 'params' es un array que contiene los parámetros de la matriz de rotación y el vector de traslación
    # params[0:9] corresponde a la matriz de rotación (una matriz 3x3)
    # params[9:12] corresponde al vector de traslación (un vector 3x1)

    # Obtener la matriz de rotación y el vector de traslación a partir de 'params'
    rotation_matrix = params[:9].reshape((3, 3))
    translation_vector = params[9:]

    # Aplicar la matriz de rotación y la traslación a los puntos fuente
    transformed_points = np.dot(source_points, rotation_matrix.T) + translation_vector

    # Calcular la suma de las distancias al cuadrado entre los puntos transformados y los puntos objetivo
    squared_distances = np.sum((target_points - transformed_points) ** 2)

    return squared_distances

def gradient(params, source_points, target_points):
    # Cálculo del gradiente de la función objetivo con respecto a los parámetros
    epsilon = 1e-6
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += epsilon
        grad[i] = (objective_function(params_plus, source_points, target_points) -
                   objective_function(params, source_points, target_points)) / epsilon
    return grad

def initial_translation(source_points, target_points):
    # Calcular el centroide de cada conjunto de puntos
    centroid_source = np.mean(source_points, axis=0)
    centroid_target = np.mean(target_points, axis=0)

    # Calcular la traslación que iguala los centroides
    return centroid_target - centroid_source

def constrain_translation(translation, target_points):
    # Calcular el centroide de los puntos objetivo
    centroid_target = np.mean(target_points, axis=0)
    # Calcular la máxima distancia en cada dimensión entre los puntos objetivo y su centroide
    max_translation = np.max(np.abs(target_points - centroid_target), axis=0)
    # Restringir la traslación dentro del entorno definido por las máximas distancias
    constrained_translation = np.clip(translation, -max_translation, max_translation)
    return constrained_translation

def gradient_descent_with_initial_translation(source_points, target_points, learning_rate=0.001, max_iterations=1000, tolerance=1e-6):
    # Obtener la matriz de procrustes como estimación inicial de la matriz de rotación
    rotation, translation = procrustes(source_points, target_points)

    # Obtener la traslación inicial que iguala los centroides de los dos conjuntos de datos
    initial_translation = np.mean(target_points, axis=0) - np.mean(np.dot(source_points, rotation.T), axis=0)

    # Combinar los parámetros de la matriz de rotación y la traslación
    params = np.concatenate((rotation.ravel(), initial_translation))

    prev_loss = float('inf')

    for iteration in range(max_iterations):
        grad = gradient(params, source_points, target_points)

        # Actualizar los parámetros usando el gradiente descendente
        params -= learning_rate * grad

        # Calcular el valor de la función objetivo
        current_loss = objective_function(params, source_points, target_points)

        # Verificar la convergencia
        if iteration > 0 and abs(current_loss - prev_loss) < tolerance:
            break

        prev_loss = current_loss

    # Obtener la matriz de rotación y el vector de traslación a partir de los parámetros finales
    rotation_matrix = params[:9].reshape((3, 3))
    translation_vector = params[9:]

    # Aplicar la restricción a la traslación
    constrained_translation = constrain_translation(translation_vector, target_points)

    return rotation_matrix, constrained_translation

def visualize_point_clouds(source_points, target_points, transformed_points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(source_points[:, 0], source_points[:, 1], source_points[:, 2], color='blue', label='Measured Points')
    ax.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2], color='red', label='Points on Planes')
    ax.scatter(transformed_points[:, 0], transformed_points[:, 1], transformed_points[:, 2], color='green', label='Transformed Points')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()


def normalize_points(points):
    # Calcular la media y la desviación estándar en cada dimensión
    mean = np.mean(points, axis=0)
    std = np.std(points, axis=0)

    # Normalizar los puntos restando la media y dividiendo por la desviación estándar
    normalized_points = (points - mean) / std

    return normalized_points, mean, std

def denormalize_translation(translation, mean, std):
    # Denormalizar la traslación multiplicando por la desviación estándar y sumando la media
    denormalized_translation = translation * std + mean

    return denormalized_translation



def find_correspondences(source_points, target_points):
    # Buscar correspondencias de puntos entre los dos conjuntos usando el árbol kd
    tree = cKDTree(target_points)
    distances, indices = tree.query(source_points)

    # Seleccionar solo las correspondencias cuya distancia sea menor que un umbral
    threshold = 1.0
    valid_correspondences = distances < threshold

    source_correspondences = source_points[valid_correspondences]
    target_correspondences = target_points[indices[valid_correspondences]]

    return source_correspondences, target_correspondences



# Ejemplo de datos para puntos en los planos
points_on_planes = np.array([
    [0, 0, 0],
    [199.885, 0, 0],
    [420.68, 10.804, -35.883],
    [619.358, 352.94, -34.728],
    [1033.714, 438.916, -47.298],
    [1040.679, 343.745, 308.735],
    [1149.313, 290.046, 354.541]
])

# Ejemplo de datos para puntos medidos (correspondientes a puntos en los planos)
measured_points = np.array([
    [611.076, 523.954, 278.609],
    [663.378, 716.959, 284.796],
    [712.111, 934.086, 255.697],
    [433.926, 1216.223, 255.779],
    [460.329, 1641.783, 255.694],
    [550.137, 1612.65, 613.95],
    [629.971, 1701.17, 664.328]
])

# Normalizar los puntos antes de aplicar el algoritmo de optimización
normalized_measured_points, mean, std = normalize_points(measured_points)
normalized_points_on_planes, _, _ = normalize_points(points_on_planes)

# Encontrar correspondencias usando el algoritmo ICP
source_correspondences, target_correspondences = find_correspondences(normalized_measured_points, normalized_points_on_planes)

# Aplicar el algoritmo de descenso del gradiente con la traslación inicial y la restricción para encontrar la matriz de rotación y el vector de traslación óptimos
rotation, translation = gradient_descent_with_initial_translation(source_correspondences, target_correspondences)

# Denormalizar la traslación para obtener el resultado final
denormalized_translation = denormalize_translation(translation, mean, std)

print("Matriz de Rotación:\n", rotation)
print("Vector de Traslación:\n", denormalized_translation)

# Transformar los puntos medidos usando la matriz de rotación y el vector de traslación obtenidos
transformed_points = np.dot(measured_points, rotation.T) + denormalized_translation

# Visualizar los puntos originales, los puntos objetivo y los puntos transformados
visualize_point_clouds(measured_points, points_on_planes, transformed_points)
