import numpy as np
import argparse
import struct

def generate_data_files(N, d, k, file_format, input_file):
    """
    Generates a dataset with N points in d dimensions and k clusters, and saves it in either text or binary format.

    :param N: Number of points.
    :param d: Number of dimensions for each point.
    :param k: Number of clusters.
    :param input_file: Path to the input file where the data will be saved.
    :param file_format: Format of the files to be generated ('txt' or 'bin').
    """
    points = []

    m = max(N, k)

    for _ in range(m):
        point = np.random.uniform(-10, 10, size=d)
        points.append(point)

    if file_format == 'txt':
        with open(input_file, 'w') as f:
            f.write(f"{N} {d} {k}\n")
            for point in points:
                f.write(' '.join(map(str, point)) + '\n')
    elif file_format == 'bin':
        with open(input_file, 'wb') as f:
            f.write(struct.pack('3i', N, d, k))
            for point in points:
                f.write(np.array(point, dtype=np.float32).tobytes())

    print(f"Input generated and saved to {input_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate a dataset with points and save results.")
    
    parser.add_argument("N", type=int, help="Number of points")
    parser.add_argument("d", type=int, help="Number of dimensions for each point")
    parser.add_argument("k", type=int, help="Number of clusters")
    parser.add_argument("file_format", choices=["txt", "bin"], help="File format to save ('txt' or 'bin')")
    parser.add_argument("input_file", type=str, help="Path to the input file")
    
    args = parser.parse_args()

    generate_data_files(args.N, args.d, args.k, args.file_format, args.input_file)

if __name__ == "__main__":
    main()
