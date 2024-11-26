import numpy as np
import argparse
import struct

def generate_data_files(N, d, k, file_format, input_file, results_file):
    """
    Generates a dataset with N points in d dimensions and k clusters, and saves it in either text or binary format.

    :param N: Number of points.
    :param d: Number of dimensions for each point.
    :param k: Number of clusters.
    :param input_file: Path to the input file where the data will be saved.
    :param results_file: Path to the results file where the data will be saved.
    :param file_format: Format of the files to be generated ('txt' or 'bin').
    """
    centroids = []
    for _ in range(k):
        centroid = np.random.uniform(-10, 10, size=d)
        centroids.append(centroid)
    
    points = []
    memberships = []
    
    for _ in range(N):
        centroid_id = np.random.randint(0, k)
        centroid = centroids[centroid_id]
        point = centroid + np.random.normal(scale=5, size=d)
        points.append(point)
        memberships.append(centroid_id)

    if file_format == 'txt':
        with open(input_file, 'w') as f:
            f.write(f"{N} {d} {k}\n")
            for centroid in centroids:
                f.write(' '.join(map(str, centroid)) + '\n')
            for point in points:
                f.write(' '.join(map(str, point)) + '\n')
    elif file_format == 'bin':
        with open(input_file, 'wb') as f:
            f.write(struct.pack('3i', N, d, k))
            for point in points:
                f.write(np.array(point, dtype=np.float32).tobytes())
            for centroid in centroids:
                f.write(np.array(centroid, dtype=np.float32).tobytes())

    print(f"Input generated and saved to {input_file}")

    with open(results_file, 'w') as f:
        for centroid in centroids:
            f.write(' '.join(map(str, centroid)) + '\n')
        
        for membership in memberships:
            f.write(f"{membership}\n")

    print(f"Results generated and saved to {results_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate a dataset with points and save results.")
    
    parser.add_argument("N", type=int, help="Number of points")
    parser.add_argument("d", type=int, help="Number of dimensions for each point")
    parser.add_argument("k", type=int, help="Number of clusters")
    parser.add_argument("file_format", choices=["txt", "bin"], help="File format to save ('txt' or 'bin')")
    parser.add_argument("input_file", type=str, help="Path to the input file")
    parser.add_argument("results_file", type=str, help="Path to the output results file")
    
    args = parser.parse_args()

    generate_data_files(args.N, args.d, args.k, args.file_format, args.input_file, args.results_file)

if __name__ == "__main__":
    main()
