import sys
import os
import struct

import matplotlib.pyplot as plt
import numpy as np

def process_input_file_text(file_path):
    """
    Processes a text file with the input format, skipping k centroids and reading N points.
    
    :param file_path: Path to the text file.
    :return: Tuple (N, d, k, points) where:
        - N: Number of points.
        - d: Number of dimensions for each point.
        - k: Number of centroids.
        - points: List of points as lists of floats.
    """
    try:
        with open(file_path, 'r') as file:
            first_line = file.readline().strip()
            try:
                N, d, k = map(int, first_line.split())
            except ValueError:
                raise ValueError("The first line must contain three integers: N, d, and k.")
            
            if N <= 0 or d <= 0 or k <= 0:
                raise ValueError("N, d, and k must be positive integers.")
            
            points = []
            for i in range(N):
                line = file.readline().strip()
                if not line:
                    raise ValueError(f"Missing data for point {i + 1}.")
                try:
                    coordinates = list(map(float, line.split()))
                except ValueError:
                    raise ValueError(f"Point {i + 1} contains invalid numeric values.")
                
                if len(coordinates) != d:
                    raise ValueError(f"Point {i + 1} must have exactly {d} dimensions.")
                
                points.append(coordinates)
            
            return N, d, k, points
    
    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    except ValueError as e:
        raise ValueError(f"Error processing the text file: {e}")

def process_input_file_binary(file_path):
    """
    Processes a binary file with the input format, skipping k centroids and reading N points.

    :param file_path: Path to the binary file.
    :return: Tuple (N, d, k, points) where:
        - N: Number of points.
        - d: Number of dimensions for each point.
        - k: Number of centroids (to skip).
        - points: List of points as lists of floats.
    """
    try:
        with open(file_path, 'rb') as file:
            header = file.read(12)
            if len(header) != 12:
                raise ValueError("File header is incomplete.")
            
            try:
                N, d, k = struct.unpack('iii', header)
            except struct.error:
                raise ValueError("File header contains invalid data.")
            
            if N <= 0 or d <= 0 or k <= 0:
                raise ValueError("N, d, and k must be positive integers.")
            
            skip_bytes = k * d * 4
            file.seek(skip_bytes, 1)
            
            points = []
            for i in range(N):
                point_data = file.read(4 * d)
                if len(point_data) != 4 * d:
                    raise ValueError(f"Incomplete data for point {i + 1}.")
                
                try:
                    point = struct.unpack(f'{d}f', point_data)
                except struct.error:
                    raise ValueError(f"Point {i + 1} contains invalid numeric data.")
                
                points.append(list(point))
            
            return N, d, k, points
    
    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    except ValueError as e:
        raise ValueError(f"Error processing the binary file: {e}")

def process_results_file_text(file_path, N, k, d):
    """
    Processes a text file containing k centroids followed by N membership indices.

    Format:
    - First k lines: Each line contains d float numbers (coordinates of centroids).
    - Next N lines: Each line contains an integer (membership index).

    :param file_path: Path to the text file.
    :param N: Number of points.
    :param k: Number of centroids.
    :param d: Number of dimensions for each centroid.
    :return: A tuple (centroids, membership), where:
        - centroids: List of k centroids, each a list of d floats.
        - membership: List of N membership indices (integers).
    """
    try:
        with open(file_path, 'r') as file:
            centroids = []
            for i in range(k):
                line = file.readline().strip()
                if not line:
                    raise ValueError(f"Missing data for centroid {i + 1}.")
                coordinates = list(map(float, line.split()))
                
                if len(coordinates) != d:
                    raise ValueError(f"Centroid {i + 1} must have exactly {d} dimensions.")
                
                centroids.append(coordinates)

            membership = []
            for i in range(N):
                line = file.readline().strip()
                if not line:
                    raise ValueError(f"Missing membership index for point {i + 1}.")
                try:
                    index = int(line)
                except ValueError:
                    raise ValueError(f"Membership index for point {i + 1} must be an integer.")
                
                membership.append(index)
            
            return centroids, membership
    
    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    except ValueError as e:
        raise ValueError(f"Error processing the text file: {e}")   

def visualize(centroids, points, membership, max_points_per_centroid=10_000):
    """
    Visualizes points in 3D space with colors based on cluster membership, optimized for large datasets.
    
    :param centroids: List of centroids, each a list of coordinates.
    :param points: List of points, each a list of coordinates.
    :param membership: List of cluster indices corresponding to each point.
    :param max_points_per_centroid: Maximum number of points to sample per centroid.
    """
    fig = plt.figure(num="K Means Clustering")
    ax = fig.add_subplot(111, projection='3d')

    colors = plt.cm.tab20.colors
    
    sampled_points = []
    for cluster_id in range(len(centroids)):
        cluster_points = np.array([points[i] for i in range(len(points)) if membership[i] == cluster_id])
        
        if len(cluster_points) > max_points_per_centroid:
            indices = np.random.choice(len(cluster_points), max_points_per_centroid, replace=False)
            cluster_points = cluster_points[indices]
        
        sampled_points.append((cluster_points, cluster_id))
    
    for cluster_points, cluster_id in sampled_points:
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2],
                   label=f'Cluster {cluster_id}', color=colors[cluster_id % len(colors)], s=20)
    
    centroids = np.array(centroids)
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
               label='Centroids', color='red', s=60, marker='s')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(loc='upper left')
    ax.set_title('3D Visualization of Centroids')

    plt.show()

def main():
    if len(sys.argv) != 4:
        print("Usage: python kmeans_visualize.py <txt|bin> <input_file> <results_file>")
        sys.exit(1)
    
    data_format = sys.argv[1]
    input_file = sys.argv[2]
    results_file = sys.argv[3]
    
    if data_format not in ("txt", "bin"):
        print("The first parameter must be 'txt' or 'bin'.")
        sys.exit(1)
    
    if not os.path.isfile(input_file):
        print(f"The input file '{input_file}' does not exist.")
        sys.exit(1)

    if not os.path.isfile(results_file):
        print(f"The results file '{input_file}' does not exist.")
        sys.exit(1)

    if(data_format == "txt"):
        N, d, k, points = process_input_file_text(input_file)
    else:
        N, d, k, points = process_input_file_binary(input_file)

    centroids, membership = process_results_file_text(results_file, N, k, d)
    visualize(centroids, points, membership)

    
if __name__ == "__main__":
    main()