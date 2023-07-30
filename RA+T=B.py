import numpy as np
from scipy.optimize import least_squares

A = np.array([2, 3, 4])
B = np.array([5, 6, 7]) 

# Define rotation angles (in radians)
alpha = 0.5
beta = 0.3
gamma = 0.2

# Create rotation matrices for each axis
R_x = np.array([[1, 0, 0],
                [0, np.cos(alpha), -np.sin(alpha)],
                [0, np.sin(alpha), np.cos(alpha)]])

R_y = np.array([[np.cos(beta), 0, np.sin(beta)],
                [0, 1, 0],
                [-np.sin(beta), 0, np.cos(beta)]])

R_z = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                [np.sin(gamma), np.cos(gamma), 0],
                [0, 0, 1]])

# Combine the rotation matrices to get the overall rotation matrix R
R = np.dot(R_z, np.dot(R_y, R_x))

# Define translation amounts
tx = 2.0
ty = -1.5
tz = 3.0

# Create the translation vector T
T = np.array([tx, ty, tz])


def objective_function():

    # Calculate the difference between R · A + T and B
    diff = np.dot(A, R.T) + T - B

    # Flatten the difference array into a 1D array
    return diff.ravel()

def find_transformation(A, B):
    # A: vector of origin coordinates
    # B: vector of transformed coordinates

    # Initial guess for the optimization parameters (rotation matrix R and translation vector T)
    initial_params = np.zeros(9 + len(A[0]))  # Assuming 3x3 rotation matrix and T with the same dimensions as A

    # Perform the optimization
    result = least_squares(objective_function, initial_params, args=(A, B))

    # Extract the optimized rotation matrix and translation vector from the result
    R_optimized = result.x[:9].reshape(3, 3)
    T_optimized = result.x[9:]

    return R_optimized, T_optimized

# Example usage:
# A = np.array(...)  # Your vector of origin coordinates
# B = np.array(...)  # Your vector of transformed coordinates
# R, T = find_transformation(A, B)
# The optimized R and T will give you the best alignment between A and B.