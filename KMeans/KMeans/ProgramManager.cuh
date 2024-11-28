#pragma once

#include "ClusteringCPU.cuh"
#include "ClusteringGPU1.cuh"
#include "ClusteringGPU2.cuh"
#include "ProgramParameters.cuh"
#include "HostTimerManager.cuh"

#include <cstdio>
#include <stdexcept>
#include <iomanip>

class IProgramManager {
public:
	virtual int GetN() = 0;
	virtual void DisplaySummary() = 0;
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

		std::cout << std::setw(25) << std::left << "Number of points: " << N << std::endl;
		std::cout << std::setw(25) << std::left << "Dimension: " << d << std::endl;
		std::cout << std::setw(25) << std::left << "Number of centroids: " << k << std::endl;
		std::cout << std::setw(25) << std::left << "Conmputation method: " << parameters.ComputationMethod << std::endl << std::endl;
	}

	int GetN() override {
		return N;
	}

	thrust::host_vector<size_t> StartComputation() override {
		auto& timerManager = Timers::HostTimerManager::GetInstance();
		timerManager.PerformClusteringTimer.Start();

		if (Parameters.ComputationMethod == "cpu") {
			CPU::ClusteringCPU<dim> clustering{};
			auto membership = clustering.PerformClustering(Centroids, Points);
			return membership;
		}
		else if (Parameters.ComputationMethod == "gpu1") {
			GPU1::ClusteringGPU1<dim> clustering{};
			auto membership = clustering.PerformClustering(Centroids, Points);
			return membership;
		}
		else if (Parameters.ComputationMethod == "gpu2") {
			GPU2::ClusteringGPU2<dim> clustering{};
			auto membership = clustering.PerformClustering(Centroids, Points);
			return membership;
		}
		else {
			throw std::runtime_error("Invalid computation method " + Parameters.ComputationMethod);
		}

		timerManager.PerformClusteringTimer.Stop();
		std::cout << "Perform clustering time: " << timerManager.PerformClusteringTimer.ElapsedMiliseconds() << " ms" << std::endl;
	}

	void DisplaySummary() override {
		auto& timerManager = HostTimerManager::GetInstance();

		std::cout << std::endl;
		std::cout << std::setw(40) << std::left << "Loading data from input file time: " << timerManager.LoadDataFromInputFileTimer.ElapsedMiliseconds() << " ms." << std::endl;
		 
		if (Parameters.ComputationMethod == "cpu") {

			std::cout << std::setw(40) << std::left << "Computing new centroids time: " << timerManager.ComputeNewCentroidsTimer.ElapsedMiliseconds() << " ms." << std::endl;
			std::cout << std::setw(40) << std::left << "Updating centroids time: " << timerManager.UpdateCentroidsTimer.ElapsedMiliseconds() << " ms." << std::endl;
		}

		std::cout << std::setw(40) << std::left << "Performing clustering time: " << timerManager.PerformClusteringTimer.ElapsedMiliseconds() << " ms." << std::endl;
		std::cout << std::setw(40) << std::left << "Saving data to output file time: " << timerManager.SaveDataToOutputFileTimer.ElapsedMiliseconds() << " ms." << std::endl;
	}

	void LoadDataFromInputFile(FILE* inputFile) override {
		auto& timerManager = Timers::HostTimerManager::GetInstance();
		timerManager.LoadDataFromInputFileTimer.Start();

		if (Parameters.DataFormat == "txt") {
			LoadDataFromTextFile(inputFile);
		}
		else if (Parameters.DataFormat == "bin") {
			LoadDataFromBinaryFile(inputFile);
		}
		else {
			throw std::runtime_error("Invalid format " + Parameters.DataFormat);
		}

		timerManager.LoadDataFromInputFileTimer.Stop();
		std::cout << "Loading data from input file time: " << timerManager.LoadDataFromInputFileTimer.ElapsedMiliseconds() << " ms" << std::endl;
	}

	void SaveDataToOutputFile(thrust::host_vector<size_t>& membership) override {
		auto& timerManager = Timers::HostTimerManager::GetInstance();
		timerManager.SaveDataToOutputFileTimer.Start();

		SaveDataToTextFile(membership);

		timerManager.SaveDataToOutputFileTimer.Stop();
		std::cout << "Saving data to output file time: " << timerManager.SaveDataToOutputFileTimer.ElapsedMiliseconds() << " ms" << std::endl;
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
				fprintf(outputFile, "%.4f ", centroid.Coordinates[i]);
			}

			fprintf(outputFile, "%.4f\n", centroid.Coordinates[dim - 1]);
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