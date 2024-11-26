﻿
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <string>
#include <iostream>
#include <cstdio>
#include <stdexcept>
#include <memory>
#include <chrono>

#include <thrust/host_vector.h>

#define GENERATE_CASE(M)												   \
    case M: {															   \
        auto instance = new ProgramManager<M>(N - k, d, k);				   \
        instance->LoadDataFromInputFile(inputFile, parameters.DataFormat);  \
        return instance;												   \
    }


#define HOST_DEVICE __host__ __device__

template<size_t dim>
struct Point {
public:
	float Coordinates[dim];

	HOST_DEVICE
	Point() {
		for (int i = 0; i < dim; i++) {
			Coordinates[i] = 0.0;
		}
	}

	HOST_DEVICE
	static float SquareDistance(const Point<dim>& p, const Point<dim>& q) {
		float distance = 0.0;
		for (int i = 0; i < dim; i++) {
			float difference = p.Coordinates[i] - q.Coordinates[i];
			distance += difference * difference;
		}

		return distance;
	}

	HOST_DEVICE
	Point<dim>& operator+=(const Point<dim>& other) {
		for (int i = 0; i < dim; i++) {
			this->Coordinates[i] += other.Coordinates[i];
		}

		return *this;
	}

private:
	HOST_DEVICE
	Point(const float coordinates[dim]) {
		for (int i = 0; i < dim; i++) {
			Coordinates[i] = coordinates[i];
		}
	}
};

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

class Timer {
public:
	virtual void Start() = 0;
	virtual void Stop() = 0;
	virtual float ElapsedMiliseconds() = 0;
	virtual void Reset() = 0;
	virtual ~Timer() = default;
};

class HostTimer : public Timer {
	std::chrono::steady_clock::time_point StartTimePoint{};
	std::chrono::microseconds TotalElapsedMicroseconds{};

public:
	void Start() override {
		StartTimePoint = std::chrono::steady_clock::now();
	}

	void Stop() override {
		auto stopTimePoint = std::chrono::steady_clock::now();
		TotalElapsedMicroseconds += std::chrono::duration_cast<std::chrono::microseconds>(
			stopTimePoint - StartTimePoint
		);
	}

	float ElapsedMiliseconds() override {
		return TotalElapsedMicroseconds.count() / 1000.0f;
	}

	void Reset() override {
		StartTimePoint = std::chrono::steady_clock::time_point();
		TotalElapsedMicroseconds = std::chrono::microseconds();
	}

	~HostTimer() override = default;
};

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

class IProgramManager {
public:
	virtual void StartComputation(std::string computationMethod) = 0;
	virtual void LoadDataFromInputFile(FILE* inputFile, std::string dataFormat) = 0;
	virtual ~IProgramManager() = default;
};

template<size_t dim>
class ProgramManager : public IProgramManager {
public:
	int N{};
	int d{};
	int k{};

	thrust::host_vector<Point<dim>> Points{};
	thrust::host_vector<Point<dim>> Centroids{};

	ProgramManager(int N, int d, int k)
	{
		this->N = N;
		this->d = d;
		this->k = k;
	}

	void StartComputation(std::string computationMethod) override {
		if (computationMethod == "cpu") {
			ClusteringCPU<dim> clustering{};
			clustering.PerformClustering(Points, Centroids);
		}
	}

	void LoadDataFromInputFile(FILE* inputFile, std::string dataFormat) override {
		if (dataFormat == "txt") {
			LoadDataFromTextFile(inputFile);
		}
		else if (dataFormat == "bin") {
			LoadDataFromBinaryFile(inputFile);
		}
		else {
			throw std::runtime_error("Invalid format " + dataFormat);
		}
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
			}
		}

		for (int i = 0; i < N; ++i) {
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
		}

		for (int i = 0; i < N; ++i) {
			if (fread(Points[i].Coordinates, sizeof(float), dim, inputFile) != dim) {
				fclose(inputFile);
				throw std::runtime_error("Error while reading the coordinates of the points.");
			}
		}
	}
};

IProgramManager* CreateManagerInstance(ProgramParameters parameters) {
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

	IProgramManager* manager{};

	try
	{
		manager = CreateManagerInstance(parameters);
		manager->StartComputation(parameters.ComputationMethod);
		std::cout << "it works!\n";
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what();
		delete manager;
		return 1;
	}
}