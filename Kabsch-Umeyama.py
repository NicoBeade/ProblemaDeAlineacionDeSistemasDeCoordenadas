import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#Meassured points
P = np.array([
    [611.076,   523.954,    278.609],
    [663.378,   716.959,    284.796],
    [712.111,   934.086,    255.697],
    [433.926,   1216.223,   255.779],
    [460.329,   1641.783,   255.694],
    [550.137,   1612.65,    613.95],
    [629.971,   1701.17,    664.328]
])


#Plane points
Q = np.array([
    [0,         0,          0],
    [199.885,   0,          0],
    [420.68,    10.804,     -35.883],
    [619.358,   352.94,     -34.728],
    [1033.714,  438.916,    -47.298],
    [1040.679,  343.745,    308.735],
    [1149.313,  290.046,    354.541]
])
"""
#Test Rotational Matrix 45° in all axis and translation vector

theta_x = np.radians(45)
theta_y = np.radians(45)
theta_z = np.radians(45)

Rx =    np.array([[1, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x), np.cos(theta_x)]])
Ry =    np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
        [0, 1, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y)]])
Rz =    np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z), np.cos(theta_z), 0],
        [0, 0, 1]])

rot = Rz @ Ry @ Rx
trasl = np.array([50, 100, 150])

P = np.dot(Q, rot) 

for vector in P:
    random_vector = np.random.uniform(-2, 2, size=(1, 3))
    vector = rot @ vector + trasl
    #vector = vector + random_vector
"""

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


# Example usage with 3D visualization:
if __name__ == "__main__":
    # Two sets of 3D points
    # P = np.array([[1.03, 2.1, 3.002], [4.02, 5.006, 6.004], [7.654, 8.0021, 9.02]])
    # Q = np.array([[2, 1, 5], [5, 4, 8], [8, 7, 11]])

    # Find nearest neighbors between points in P and Q
    nearest_neighbors = find_nearest_neighbors(P, Q)

    R, rmsd = kabsch_rmsd(P, Q, nearest_neighbors)

    print("Optimal Rotation Matrix:")
    print(R)
    print("Root-Mean-Square Deviation (RMSD):", rmsd)

    # Print original points P and points Q
    print("Original Points (P):")
    print(P)
    print("Points Q:")
    print(Q)

    # Plotting the original and aligned points in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot original points P (in blue)
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], c='blue', label='Measured Points')

    # Plot points Q (in green)
    ax.scatter(Q[:, 0], Q[:, 1], Q[:, 2], c='green', label='Plane Points')

    # Calculate the aligned points using the optimal rotation matrix
    P_aligned = np.dot(P - np.mean(P, axis=0), R.T) + np.mean(Q, axis=0)

    # Plot aligned points (in red)
    ax.scatter(P_aligned[:, 0], P_aligned[:, 1], P_aligned[:, 2], c='red', label='Transformed Points')

    # Set plot labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    # Save plot
    plt.savefig('aligned_points_plot.pdf')  
    plt.show()

    # Calculate the RMSD and MAE between the aligned points and the target points Q
    rmsd_final = calculate_rmsd(P_aligned, Q)
    mae_final = calculate_mae(P_aligned, Q)
    print("Final Root-Mean-Square Deviation (RMSD) after alignment:", rmsd_final)
    print("Mean Absolute Error (MAE) after alignment:", mae_final)

    # Calculate the range of the data (max - min) for scaling the errors
    data_range = np.max(P) - np.min(P)

    # Calculate the percentage error relative to the data range for RMSD and MAE
    rmsd_percentage_error = calculate_percentage_error(rmsd_final, data_range)
    mae_percentage_error = calculate_percentage_error(mae_final, data_range)
    print("Percentage RMSD Error:", rmsd_percentage_error, "%")
    print("Percentage MAE Error:", mae_percentage_error, "%")