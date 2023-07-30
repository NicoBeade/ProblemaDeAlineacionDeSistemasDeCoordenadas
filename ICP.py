import numpy as np
from scipy.spatial import KDTree
from scipy.linalg import orthogonal_procrustes
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def icp(source_points, target_points, max_iterations=100, tolerance=1e-6):
    for iteration in range(max_iterations):
        # Find the nearest neighbors from the source to the target points
        kdtree = KDTree(target_points)
        distances, indices = kdtree.query(source_points)

        # Estimate the transformation using Procrustes analysis
        source_centered = source_points - np.mean(source_points, axis=0)
        target_centered = target_points[indices] - np.mean(target_points[indices], axis=0)
        rotation, translation = orthogonal_procrustes(source_centered, target_centered)

        # Apply the transformation to the source points
        source_points_transformed = np.dot(source_points, rotation) + translation

        # Calculate the mean squared error (MSE) between transformed source and target points
        mse = np.mean(np.sum((target_points[indices] - source_points_transformed) ** 2, axis=1))

        # Check for convergence
        if iteration > 0 and abs(mse - prev_mse) < tolerance:
            break
        prev_mse = mse

        # Update the source points for the next iteration
        source_points = source_points_transformed

    return rotation, translation, mse



# Example data for points on the planes
points_on_planes = np.array([
    [0, 0, 0],
    [199.885, 0, 0],
    [420.68, 10.804, -35.883],
    [619.358, 352.94, -34.728],
    [1033.714, 438.916, -47.298],
    [1040.679, 343.745, 308.735],
    [1149.313, 290.046, 354.541]
])

# Example data for measured points (corresponding to points on the planes)
measured_points = np.array([
    [611.076, 523.954, 278.609],
    [663.378, 716.959, 284.796],
    [712.111, 934.086, 255.697],
    [433.926, 1216.223, 255.779],
    [460.329, 1641.783, 255.694],
    [550.137, 1612.65, 613.95],
    [629.971, 1701.17, 664.328]
])


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

# Call the ICP function
rotation, translation, mse = icp(measured_points, points_on_planes)

# Transform the measured points using the obtained transformation
transformed_points = np.dot(measured_points, rotation) + translation

print("Rotation matrix:\n", rotation)
print("Translation vector:\n", translation)
print("Mean Squared Error (MSE):", mse)

# Visualize the point clouds
visualize_point_clouds(measured_points, points_on_planes, transformed_points)
