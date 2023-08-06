import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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


# Grafico
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot puntos medidos
ax.scatter(measuredPoints[:, 0], measuredPoints[:, 1], measuredPoints[:, 2], c='blue', label='Puntos medidos')


def translate_to_centroid(points, centroid):
    """
    Translate the points to the specified centroid.

    Parameters:
        points (numpy array): The set of 3D points.
        centroid (numpy array): The target centroid to which the points will be translated.

    Returns:
        translated_points (numpy array): The translated points.
    """
    translated_points = points + (centroid - np.mean(points, axis=0))
    return translated_points



# Calculate the centroid of measuredPoints and planePoints
measuredPoints_centroid = np.mean(measuredPoints, axis=0)
planePoints_centroid = np.mean(planePoints, axis=0)

# Translate measuredPoints to the centroid of planePoints
measuredPoints = translate_to_centroid(measuredPoints, planePoints_centroid)


def normalize_data(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    normalized_data = (data - mean) / std
    return normalized_data, mean, std


normalized_measuredPoints, mean_measured, std_measured = normalize_data(measuredPoints)
normalized_planePoints, mean_plane, std_plane = normalize_data(planePoints)



def rotation_matrix(angle, axis):
    """
    Generate a 3D orthogonal rotation matrix.

    Parameters:
        angle (float): The rotation angle in radians.
        axis (numpy array): The axis of rotation as a 3D vector.

    Returns:
        R (numpy array): The 3D orthogonal rotation matrix.
    """
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    R = np.array([[c + x**2*(1 - c), x*y*(1 - c) - z*s, x*z*(1 - c) + y*s],
                  [y*x*(1 - c) + z*s, c + y**2*(1 - c), y*z*(1 - c) - x*s],
                  [z*x*(1 - c) - y*s, z*y*(1 - c) + x*s, c + z**2*(1 - c)]])

    # Ensure orthogonality using SVD
    U, _, Vt = np.linalg.svd(R)
    R_orthogonal = np.dot(U, Vt)

    return R_orthogonal


def distance_squared(p1, p2):
    """
    Calculate the squared distance between two 3D points.

    Parameters:
        p1, p2 (numpy arrays): The 3D points.

    Returns:
        d_squared (float): The squared distance between p1 and p2.
    """
    return np.sum((p1 - p2)**2)

def error_function(measured_points, plane_points, rotation_matrix):
    """
    Calculate the error between measured points and corresponding points on the plane after rotation.

    Parameters:
        measured_points (numpy array): The set of 3D points to be rotated.
        plane_points (numpy array): The set of 3D points representing the plane.
        rotation_matrix (numpy array): The 3D rotation matrix.

    Returns:
        error (float): The sum of squared distances between the rotated measured points and plane points.
    """
    rotated_points = np.dot(measured_points, rotation_matrix.T)
    error = sum(distance_squared(rotated_points[i], plane_points[i]) for i in range(len(measured_points)))
    return error



def gradient_rotation_matrix(rotation_matrix, measured_points, plane_points):
    """
    Calculate the gradient of the error function with respect to the rotation matrix.

    Parameters:
        rotation_matrix (numpy array): The current rotation matrix.
        measured_points (numpy array): The set of 3D points to be rotated.
        plane_points (numpy array): The set of 3D points representing the plane.

    Returns:
        gradient (numpy array): The gradient of the error function with respect to the rotation matrix.
    """
    rotated_points = np.dot(measured_points, rotation_matrix.T)
    residuals = rotated_points - plane_points
    gradient = np.zeros((3, 3))

    for i in range(len(measured_points)):
        gradient += 2 * np.outer(residuals[i], measured_points[i]) - np.dot(residuals[i], rotation_matrix.T)

    return gradient




def orthogonal_rotation_matrix_gradient_descent(measured_points, plane_points, learning_rate=0.0001, num_iterations=5000):
    """
    Perform gradient descent to find the optimal orthogonal rotation matrix.

    Parameters:
        measured_points (numpy array): The set of 3D points to be rotated.
        plane_points (numpy array): The set of 3D points representing the plane.
        learning_rate (float): The learning rate for gradient descent (default: 0.01).
        num_iterations (int): The number of iterations for gradient descent (default: 1000).

    Returns:
        optimal_rotation_matrix (numpy array): The orthogonal rotation matrix that minimizes the error.
        errors (list): List of errors at each iteration.
    """
    # Initialize the rotation matrix with the identity matrix
    rotation_matrix_estimate = np.eye(3)

    errors = []

    for i in range(num_iterations):
        # Calculate the gradient of the error function with respect to the rotation matrix
        gradient = gradient_rotation_matrix(rotation_matrix_estimate, measured_points, plane_points)

        # Update the rotation matrix using the gradient while ensuring orthogonality
        rotation_matrix_estimate -= learning_rate * gradient
        U, _, Vt = np.linalg.svd(rotation_matrix_estimate)
        rotation_matrix_estimate = np.dot(U, Vt)

        # Calculate the error with the updated rotation matrix
        error = error_function(measured_points, plane_points, rotation_matrix_estimate)
        errors.append(error)

    return rotation_matrix_estimate, errors





def find_nearest_neighbors(P, Q):
    """
    Find nearest neighbors between two sets of points using Euclidean distance.

    Parameters:
        P, Q (numpy arrays): Two sets of 3D points.

    Returns:
        nearest_neighbors (list of tuples): List of tuples representing the nearest neighbors.
                                           Each tuple contains the index of a point in P and its
                                           corresponding index in Q.
    """
    nearest_neighbors = []
    for i, p in enumerate(P):
        min_dist = float('inf')
        nearest_q_index = None
        for j, q in enumerate(Q):
            dist = np.linalg.norm(p - q)
            if dist < min_dist:
                min_dist = dist
                nearest_q_index = j
        nearest_neighbors.append((i, nearest_q_index))
    return nearest_neighbors

def kabsch_rmsd(P, Q, nearest_neighbors):
    """
    Kabsch algorithm to find the optimal rotation matrix that aligns two sets of points.
    
    Parameters:
        P, Q (numpy arrays): Two sets of 3D points with the same number of points.
        nearest_neighbors (list of tuples): List of tuples representing the nearest neighbors.
                                           Each tuple contains the index of a point in P and its
                                           corresponding index in Q.
        
    Returns:
        R (numpy array): The optimal rotation matrix.
        rmsd (float): The root-mean-square deviation between the two point sets after alignment.
    """
    # Convert nearest_neighbors to a NumPy array
    nearest_neighbors = np.array(nearest_neighbors)

    # Step 1: Centering the points
    P_centroid = np.mean(P, axis=0)
    Q_centroid = np.mean(Q, axis=0)
    
    P_centered = P - P_centroid
    Q_centered = Q - Q_centroid
    
    # Step 2: Calculate the covariance matrix H
    H = np.dot(P_centered[nearest_neighbors[:, 0]].T, Q_centered[nearest_neighbors[:, 1]])
    
    # Step 3: Singular Value Decomposition (SVD)
    U, _, Vt = np.linalg.svd(H)
    
    # Step 4: Calculate the optimal rotation matrix R
    d = np.linalg.det(np.dot(Vt.T, U.T))
    S = np.eye(3)
    S[2, 2] = d
    R = np.dot(np.dot(Vt.T, S), U.T)
    
    # Step 5: Calculate the root-mean-square deviation (RMSD)
    P_aligned = np.dot(P_centered, R.T) + Q_centroid
    diff = P_aligned - Q
    rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
    
    return R, rmsd

def calculate_rmsd(P_aligned, Q):
    """
    Calculate the Root-Mean-Square Deviation (RMSD) between the aligned points and the target points.

    Parameters:
        P_aligned, Q (numpy arrays): Two sets of 3D points with the same number of points.

    Returns:
        rmsd (float): The root-mean-square deviation between the aligned points and the target points.
    """
    diff = P_aligned - Q
    rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
    return rmsd


def calculate_mae(P_aligned, Q):
    """
    Calculate the Mean Absolute Error (MAE) between the aligned points and the target points.

    Parameters:
        P_aligned, Q (numpy arrays): Two sets of 3D points with the same number of points.

    Returns:
        mae (float): The mean absolute error between the aligned points and the target points.
    """
    diff = P_aligned - Q
    mae = np.mean(np.abs(diff))
    return mae


def calculate_percentage_error(error_metric, data_range):
    """
    Calculate the percentage error relative to the data range.

    Parameters:
        error_metric (float): The error metric value (e.g., RMSD or MAE).
        data_range (float): The range of the data (e.g., max - min).

    Returns:
        percentage_error (float): The percentage error relative to the data range.
    """
    percentage_error = (error_metric / data_range) * 100.0
    return percentage_error



def calculate_distances_between_points(P_aligned, Q):
    """
    Calculate the distance between every pair of points in P_aligned and Q arrays.

    Parameters:
        P_aligned, Q (numpy arrays): Two sets of 3D points with the same number of points.

    Returns:
        distances (numpy array): A 1D array containing the distances between every pair of points.
    """
    num_p_aligned = P_aligned.shape[0]

    distances = np.zeros(num_p_aligned)

    for i in range(num_p_aligned):
        distances[i] = np.linalg.norm(P_aligned[i] - Q[i])

    return distances






# Find nearest neighbors between points in P and Q
nearest_neighbors = find_nearest_neighbors(measuredPoints, planePoints)

R, rmsd = kabsch_rmsd(measuredPoints, planePoints, nearest_neighbors)


# Calculate the aligned points using the optimal rotation matrix
measuredPoints = np.dot(measuredPoints - np.mean(measuredPoints, axis=0), R.T) + np.mean(planePoints, axis=0)





optimal_rotation_matrix, errors = orthogonal_rotation_matrix_gradient_descent(normalized_measuredPoints, normalized_planePoints)

# Denormalize the rotation matrix
optimal_rotation_matrix_denormalized = np.dot(np.dot(np.linalg.inv(np.diag(std_measured)), optimal_rotation_matrix), np.diag(std_plane))

# Apply the denormalized rotation matrix to the measured points
transformedPoints = np.dot(measuredPoints, optimal_rotation_matrix_denormalized.T)

# Denormalize the transformed points for plotting
denormalized_transformedPoints = np.dot(transformedPoints * std_measured + mean_measured, np.linalg.inv(optimal_rotation_matrix_denormalized).T)


# Print the results
print("Optimal Rotation Matrix:")
print(optimal_rotation_matrix_denormalized)
print("\nFinal Error:", errors[-1])



    


# Plot puntos ideales
ax.scatter(planePoints[:, 0], planePoints[:, 1], planePoints[:, 2], c='green', label='Puntos ideales')

# Plot puntos transformados
ax.scatter(transformedPoints[:, 0], transformedPoints[:, 1], transformedPoints[:, 2], c='red', label='Puntos transformados')

# Connect points in planePoints with lines (plot line segments)
for i in range(len(planePoints) - 1):
    ax.plot([planePoints[i, 0], planePoints[i + 1, 0]],
            [planePoints[i, 1], planePoints[i + 1, 1]],
            [planePoints[i, 2], planePoints[i + 1, 2]], color='blue')

# Connect points in transformedPoints with lines (plot line segments)
for i in range(len(transformedPoints) - 1):
    ax.plot([transformedPoints[i, 0], transformedPoints[i + 1, 0]],
            [transformedPoints[i, 1], transformedPoints[i + 1, 1]],
            [transformedPoints[i, 2], transformedPoints[i + 1, 2]], color='purple')


# Set plot labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

# Save plot
plt.savefig('aligned_points_plot.pdf')  
plt.show()



# Calculate the distance matrix between the aligned points and the target points
dist_matrix = calculate_distances_between_points(transformedPoints, planePoints)

# Denormalize the distance matrix
denormalized_dist_matrix = dist_matrix * (std_measured ** 2 / std_plane ** 2)

# Print the distance matrix
print("Distance Matrix:")
print(dist_matrix)


# Save the distance matrix to an Excel file using pandas
df_distance_matrix = pd.DataFrame(dist_matrix)
df_distance_matrix.to_excel('distance_matrix.xlsx', index=False)