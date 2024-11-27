
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "HostTimer.cuh"
#include "Point.cuh"

using namespace Timers;
using namespace DataStructures;

#include <string>
#include <iostream>
#include <cstdio>
#include <stdexcept>
#include <memory>
#include <chrono>

#include <thrust/host_vector.h>

#define GENERATE_CASE(M)														  \
    case M: {																	  \
        auto instance = std::make_unique<ProgramManager<M>>(N, d, k, parameters); \
        instance->LoadDataFromInputFile(inputFile);								  \
        return instance;														  \
    }

#define THREADS_IN_ONE_BLOCK 1024

struct ProgramParameters {
	std::string DataFormat{};
	std::string ComputationMethod{};
	std::string InputFile{};
	std::string OutputFile{};
	bool Success{};
};

ProgramParameters ParseProgramParameters(int argc, char* argv[]) {
	ProgramParameters parameters{};

	if (argc != 5) {
		std::cerr << "Usage: KMeans data_format computation_method input_file output_file";
		return parameters;
	}

	parameters.DataFormat = argv[1];
	parameters.ComputationMethod = argv[2];
	parameters.InputFile = argv[3];
	parameters.OutputFile = argv[4];

	if (parameters.DataFormat != "txt" && parameters.DataFormat != "bin") {
		std::cerr << "Invalid data format. Use 'txt' or 'bin'.\n";
		return parameters;
	}

	if (parameters.ComputationMethod != "cpu" && parameters.ComputationMethod != "gpu1" && parameters.ComputationMethod != "gpu2") {
		std::cerr << "invalid computation method. Use 'cpu', 'gpu1' or 'gpu2'.\n";
	}

	parameters.Success = true;
	return parameters;
}

template<size_t dim>
class ClusteringCPU {
	HostTimer ComputeNewCentroidsTimer{};
	HostTimer UpdateCentroidsTimer{};

public:

	thrust::host_vector<size_t> PerformClustering(
		thrust::host_vector<Point<dim>>& points,
		thrust::host_vector<Point<dim>>& centroids)
	{
		thrust::host_vector<Point<dim>> updatedCentroids(centroids.size());
		thrust::host_vector<size_t> updatedCounts(centroids.size());
		thrust::host_vector<size_t> membership(points.size());

		size_t changes = points.size();
		size_t iteration = 0, maxIterations = 100;

		std::cout << "Starting clustering..." << std::endl;
		std::cout << "Number of points: " << points.size() << ", Number of centroids: " << centroids.size() << std::endl;

		while (iteration++ < maxIterations && changes != 0) {
			std::cout << "\n=== Iteration: " << iteration << " ===" << std::endl;

			changes = 0;

			std::cout << "-> Computing new centroids..." << std::endl;

			ComputeNewCentroidsTimer.Start();
			ComputeNewCentroids(points, centroids, updatedCentroids, updatedCounts, membership, changes);
			ComputeNewCentroidsTimer.Stop();

			std::cout << "   Computation time: " << ComputeNewCentroidsTimer.ElapsedMiliseconds() << " ms" << std::endl;

			std::cout << "-> Updating centroids..." << std::endl;

			UpdateCentroidsTimer.Start();
			UpdateCentroids(centroids, updatedCentroids, updatedCounts);
			UpdateCentroidsTimer.Stop();

			std::cout << "   Update time: " << UpdateCentroidsTimer.ElapsedMiliseconds() << " ms" << std::endl;

			std::cout << "-> Changes in membership: " << changes << std::endl;
		}

		if (changes == 0) {
			std::cout << "\nClustering completed: No changes in membership." << std::endl;
		}
		else {
			std::cout << "\nClustering completed: Maximum number of iterations reached." << std::endl;
		}

		return membership;
	}


private:
	size_t FindNearestCentroid(
		Point<dim> point,
		thrust::host_vector<Point<dim>>& centroids)
	{
		size_t nearest{};
		auto nearest_distance = std::numeric_limits<float>::infinity();
		auto k = centroids.size();

		for (size_t i = 0; i < k; ++i) {
			auto distance = Point<dim>::SquareDistance(point, centroids[i]);
			if (distance < nearest_distance) {
				nearest_distance = distance;
				nearest = i;
			}
		}

		return nearest;
	}

	void ComputeNewCentroids(
		thrust::host_vector<Point<dim>>& points,
		thrust::host_vector<Point<dim>>& centroids,
		thrust::host_vector<Point<dim>>& updatedCentroids,
		thrust::host_vector<size_t>& updatedCounts,
		thrust::host_vector<size_t>& membership,
		size_t& changes)
	{
		auto N = points.size();

		for (size_t i = 0; i < N; ++i) {
			auto nearest = FindNearestCentroid(points[i], centroids);

			if (membership[i] != nearest) {
				membership[i] = nearest;
				changes++;
			}

			updatedCentroids[nearest] += points[i];
			updatedCounts[nearest]++;
		}
	}

	void UpdateCentroids(
		thrust::host_vector<Point<dim>>& centroids,
		thrust::host_vector<Point<dim>>& updatedCentroids,
		thrust::host_vector<size_t>& updatedCounts)
	{
		auto k = centroids.size();
		for (size_t i = 0; i < k; ++i) {
			for (size_t j = 0; j < dim; ++j) {
				centroids[i].Coordinates[j] = updatedCentroids[i].Coordinates[j] /
					std::max(updatedCounts[i], (size_t)1);
				updatedCentroids[i].Coordinates[j] = 0.0f;
			}

			updatedCounts[i] = 0;
		}
	}
};


template<size_t dim>
__global__
void AoS2SoAKernel(float* deviceAoS, float* deviceSoA, size_t length) {
	// [x1, y1, z1, x2, y2, z2] -> [x1, x2, y1, y2, z1, z2]
	
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= length) return;

	for (size_t i = 0; i < dim; ++i) {
		deviceSoA[tid + i * length] = deviceAoS[tid * dim + i];
	}
}

template<size_t dim>
__global__
void SoA2AoSKernel(float* deviceSoA, float* deviceAoS, size_t length) {
	// [x1, x2, y1, y2, z1, z2] -> [x1, y1, z1, x2, y2, z2] 

	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= length) return;

	for (size_t i = 0; i < dim; ++i) {
		deviceAoS[tid * dim + i] = deviceSoA[tid + i * length];
	}
}

template<size_t dim>
class DevicePointsCollection {
private:
	size_t Size{};
	float* DevicePoints{};

public:
	DevicePointsCollection(size_t size)
	{
		Size = size;
		cudaMalloc(&DevicePoints, sizeof(Point<dim>) * Size);
		cudaMemset(DevicePoints, 0, sizeof(Point<dim>) * Size);
	}

	DevicePointsCollection(thrust::host_vector<Point<dim>>& points)
	{
		FromHost(points);
	}

	~DevicePointsCollection() {
		cudaFree(DeviePoints);
	}

	void operator=(thrust::host_vector<Point<dim>>& points) {
		FromHost(points);
	}

	size_t GetSize() {
		return Size;
	}

	__device__
	static double& GetCoordinate(double* devicePoints, size_t size, size_t pointIndex, size_t dimensionIdx) {
		return devicePoints[pointIndex + dimensionIdx * size];
	}

	float* RawAccess() {
		return DevicePoints;
	}

	thrust::host_vector<Point<dim>> ToHost() {
		float* deviceAoS{};

		cudaMalloc(&deviceAoS, sizeof(Point<dim>) * Size);

		size_t blockCount = (Size + THREADS_IN_ONE_BLOCK - 1) / THREADS_IN_ONE_BLOCK;

		// pomiar czasu na konwersję pomiędzy SoA -> AoS
		SoA2AoSKernel<dim> << < blockCount, THREADS_IN_ONE_BLOCK >> > (DevicePoints, deviceAoS, Size);
		// stiop pomiar czasu na konwersję pomiędzy SoA -> AoS

		thrust::host_vector<Point<dim>> points(Size);
		
		// pomiar czasu na transfer danych z gpu na cpu
		cudaMemcpy(points.data(), deviceAoS, sizeof(Point<dim>) * Size, cudaMemcpyDeviceToHost);
		// stop pomiaru czasu na transfer danych z gpu na cpu

		cudaFree(deviceAoS);
	}

private:
	void FromHost(thrust::host_vector<Point<dim>>& points) {
		Size = points.size();

		float* deviceAoS{};

		cudaMalloc(&DevicePoints, sizeof(Point<dim>) * Size);
		cudaMalloc(&deviceAoS, sizeof(Point<dim>) * Size);

		// pomiar czasu na transfer danych z cpu na gpu
		cudaMemcpy(deviceAos, points.data(), sizeof(Point<dim>) * Size, cudaMemcpyHostToDevice);
		// stop pomiaru czasu na transfer danych z cpu na gpu

		size_t blockCount = (Size + THREADS_IN_ONE_BLOCK - 1) / THREADS_IN_ONE_BLOCK;

		// pomiar czasu na konwersję pomiędzy AoS -> SoA
		AoS2SoAKernel<dim> << <blockCount, THREADS_IN_ONE_BLOCK >> > (deviceAoS, DevicePoints, Size);
		// stop pomiaru czasu na konwersję pomiędzy AoS -> SoA

		cudaFree(deviceAoS);
	}
};

template<size_t dim>
struct DeviceRawDataGPU1 {
	size_t PointsCount{};
	float* DevicePoints{};
	size_t CentroidsCount{};
	float* DeviceCentroids{};
	size_t* Membership{};
	float* deviceChanges{};
	size_t* PointsPermutation{};
	float* UpdatedCentroids{};
	size_t* UpdatedCentroidsCounts{};
};

template<size_t dim>
struct DeviceDataGPU1 {

};


template<size_t dim>
class ClusteringGPU1 {
public:

private:
	void UpdateCentroids() {

	}

};

class IProgramManager {
public:
	virtual thrust::host_vector<size_t> StartComputation() = 0;
	virtual void LoadDataFromInputFile(FILE* inputFile) = 0;
	virtual void SaveDataToOutputFile(thrust::host_vector<size_t>& membership) = 0;
	virtual ~IProgramManager() = default;
};

template<size_t dim>
class ProgramManager : public IProgramManager {
public:
	int N{};
	int d{};
	int k{};

	ProgramParameters Parameters;

	thrust::host_vector<Point<dim>> Points{};
	thrust::host_vector<Point<dim>> Centroids{};

	ProgramManager(int N, int d, int k, ProgramParameters parameters)
	{
		this->N = N;
		this->d = d;
		this->k = k;
		this->Parameters = parameters;
	}

	thrust::host_vector<size_t> StartComputation() override {
		if (Parameters.ComputationMethod == "cpu") {
			ClusteringCPU<dim> clustering{};
			auto membership = clustering.PerformClustering(Points, Centroids);
			return membership;
		}

		return thrust::host_vector<size_t>{};
	}

	void LoadDataFromInputFile(FILE* inputFile) override {
		if (Parameters.DataFormat == "txt") {
			LoadDataFromTextFile(inputFile);
		}
		else if (Parameters.DataFormat == "bin") {
			LoadDataFromBinaryFile(inputFile);
		}
		else {
			throw std::runtime_error("Invalid format " + Parameters.DataFormat);
		}
	}

	void SaveDataToOutputFile(thrust::host_vector<size_t>& membership) override {
		SaveDataToTextFile(membership);
	}

	~ProgramManager() = default;

private:
	void LoadDataFromTextFile(FILE* inputFile) {
		Points.resize(N);
		Centroids.resize(k);

		for (int i = 0; i < k; ++i) {
			for (size_t j = 0; j < dim; ++j) {
				if (fscanf_s(inputFile, "%f", &(Centroids[i].Coordinates[j])) != 1) {
					fclose(inputFile);
					throw std::runtime_error("Error while reading the coordinates of the centroids.");
				}
				if (i < N) {
					Points[i].Coordinates[j] = Centroids[i].Coordinates[j];
				}
			}
		}

		for (int i = k; i < N; ++i) {
			for (size_t j = 0; j < dim; ++j) {
				if (fscanf_s(inputFile, "%f", &(Points[i].Coordinates[j])) != 1) {
					fclose(inputFile);
					throw std::runtime_error("Error while reading the coordinates of the points.");
				}
			}
		}
	}
	void LoadDataFromBinaryFile(FILE* inputFile) {
		Points.resize(N);
		Centroids.resize(k);

		for (int i = 0; i < k; ++i) {
			if (fread(Centroids[i].Coordinates, sizeof(float), dim, inputFile) != dim) {
				fclose(inputFile);
				throw std::runtime_error("Error while reading the coordinates of the centroids.");
			}
			if (i < N) {
				for (size_t j = 0; j < dim; ++j) {
					Points[i].Coordinates[j] = Centroids[i].Coordinates[j];
				}
			}
		}

		for (int i = k; i < N; ++i) {
			if (fread(Points[i].Coordinates, sizeof(float), dim, inputFile) != dim) {
				fclose(inputFile);
				throw std::runtime_error("Error while reading the coordinates of the points.");
			}
		}
	}
	void SaveDataToTextFile(thrust::host_vector<size_t>& membership) {
		FILE* outputFile{};

		if (fopen_s(&outputFile, Parameters.OutputFile.c_str(), "w") != 0) {
			throw std::runtime_error("Could not open " + Parameters.OutputFile);
		}

		for (const auto& centroid : Centroids) {
			for (size_t i = 0; i < dim - 1; ++i) {
				fprintf(outputFile, "%.6f ", centroid.Coordinates[i]);
			}

			fprintf(outputFile, "%.6f\n", centroid.Coordinates[dim - 1]);
		}

		for (const auto& m : membership) {
			fprintf(outputFile, "%zu\n", m);
		}

		fclose(outputFile);
	}
	void SaveDataToBinaryFile(thrust::host_vector<size_t>& membership) {
		FILE* outputFile{};

		if (fopen_s(&outputFile, Parameters.OutputFile.c_str(), "wb") != 0) {
			throw std::runtime_error("Could not open " + Parameters.OutputFile);
		}

		for (const auto& centroid : Centroids) {
			if (fwrite(centroid.Coordinates, sizeof(float), dim, outputFile) != dim) {
				fclose(outputFile);
				throw std::runtime_error("Error writing centroids to binary file.");
			}
		}

		if (fwrite(membership.data(), sizeof(size_t), membership.size(), outputFile) != membership.size()) {
			fclose(outputFile);
			throw std::runtime_error("Error writing membership data to binary file.");
		}

		fclose(outputFile);
	}
};

std::unique_ptr<IProgramManager> CreateManagerInstance(ProgramParameters parameters) {
	FILE* inputFile{};
	int N, d, k;

	if (parameters.DataFormat == "txt") {
		if (fopen_s(&inputFile, parameters.InputFile.c_str(), "r") != 0) {
			throw std::runtime_error("Could not open " + parameters.InputFile);
		}

		if (fscanf_s(inputFile, "%d %d %d", &N, &d, &k) != 3) {
			fclose(inputFile);
			throw std::runtime_error("Error while reading the header of " + parameters.InputFile);
		}
	}
	else if (parameters.DataFormat == "bin") {
		if (fopen_s(&inputFile, parameters.InputFile.c_str(), "rb") != 0) {
			throw std::runtime_error("Could not open " + parameters.InputFile);
		}

		if (fread(&N, sizeof(int), 1, inputFile) != 1 ||
			fread(&d, sizeof(int), 1, inputFile) != 1 ||
			fread(&k, sizeof(int), 1, inputFile) != 1) {
			fclose(inputFile);
			throw std::runtime_error("Error while reading the header of " + parameters.InputFile);
		}
	}
	else {
		throw std::runtime_error("Invalid format " + parameters.DataFormat);
	}


	switch (d) {
		GENERATE_CASE(1);
		GENERATE_CASE(2);
		GENERATE_CASE(3);
		GENERATE_CASE(4);
		GENERATE_CASE(5);
		GENERATE_CASE(6);
		GENERATE_CASE(7);
		GENERATE_CASE(8);
		GENERATE_CASE(9);
		GENERATE_CASE(10);
		GENERATE_CASE(11);
		GENERATE_CASE(12);
		GENERATE_CASE(13);
		GENERATE_CASE(14);
		GENERATE_CASE(15);
		GENERATE_CASE(16);
		GENERATE_CASE(17);
		GENERATE_CASE(18);
		GENERATE_CASE(19);
		GENERATE_CASE(20);
	default:
		fclose(inputFile);
		throw std::runtime_error("Unhandled dimension");
	}
}

int main(int argc, char* argv[])
{
	auto parameters = ParseProgramParameters(argc, argv);

	if (!parameters.Success) {
		return 1;
	}

	std::unique_ptr<IProgramManager> manager{};

	try
	{
		manager = CreateManagerInstance(parameters);
		auto membership = manager->StartComputation();
		manager->SaveDataToOutputFile(membership);
		std::cout << "it works!\n";
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what();
		return 1;
	}
}