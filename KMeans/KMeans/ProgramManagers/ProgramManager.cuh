#pragma once

#include "../CPU/ClusteringCPU.cuh"
#include "../GPU1/ClusteringGPU1.cuh"
#include "../GPU2/ClusteringGPU2.cuh"
#include "ProgramParameters.cuh"
#include "../Timers/TimerManager.cuh"

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

		std::cout << std::setw(25) << std::left << "Number of points: " 
			<< N << std::endl;
		std::cout << std::setw(25) << std::left << "Dimension: " 
			<< d << std::endl;
		std::cout << std::setw(25) << std::left << "Number of centroids: " 
			<< k << std::endl;
		std::cout << std::setw(25) << std::left << "Computation method: " 
			<< parameters.ComputationMethod << std::endl;
		std::cout << std::setw(25) << std::left << "Data format: " 
			<< parameters.DataFormat << std::endl << std::endl << std::endl;
	}

	int GetN() override {
		return N;
	}

	thrust::host_vector<size_t> StartComputation() override {
		auto& timerManager = Timers::TimerManager::GetInstance();
		timerManager.PerformClusteringTimer.Start();

		thrust::host_vector<size_t> membership{};

		if (Parameters.ComputationMethod == "cpu") {
			CPU::ClusteringCPU<dim> clustering{};
			membership = clustering.PerformClustering(Centroids, Points);
		}
		else if (Parameters.ComputationMethod == "gpu1") {
			GPU1::ClusteringGPU1<dim> clustering{};
			membership = clustering.PerformClustering(Centroids, Points);
			CUDACHECK(cudaDeviceSynchronize());
		}
		else if (Parameters.ComputationMethod == "gpu2") {
			GPU2::ClusteringGPU2<dim> clustering{};
			membership = clustering.PerformClustering(Centroids, Points);
			CUDACHECK(cudaDeviceSynchronize());
		}
		else {
			throw std::runtime_error("Invalid computation method " + Parameters.ComputationMethod);
		}

		timerManager.PerformClusteringTimer.Stop();
		return membership;
	}

	void DisplaySummary() override {
		auto& timerManager = TimerManager::GetInstance();

		std::cout << std::endl;

		if (Parameters.ComputationMethod == "cpu") {

			std::cout << std::setw(40) << std::left << "Computing new centroids time: "
				<< timerManager.ComputeNewCentroidsTimer.TotalElapsedMiliseconds() << " ms." << std::endl;
			std::cout << std::setw(40) << std::left << "Updating centroids time: "
				<< timerManager.UpdateCentroidsTimer.TotalElapsedMiliseconds() << " ms." << std::endl;
		}

		else {
			std::cout << std::setw(40) << std::left << "Host to Device transfer time: "
				<< timerManager.Host2DeviceDataTransfer.TotalElapsedMiliseconds() << " ms." << std::endl;
			if (Parameters.ComputationMethod == "gpu1") {
				std::cout << std::setw(40) << std::left << "AoS to SoA conversion time: "
					<< timerManager.AoS2SoAKernelTimer.TotalElapsedMiliseconds() << " ms." << std::endl;
			}

			std::cout << std::setw(40) << std::left << "Computing new centroids time: "
				<< timerManager.ComputeNewCentroidsKernelTimer.TotalElapsedMiliseconds() << " ms." << std::endl;
			std::cout << std::setw(40) << std::left << "Updating centroids time: "
				<< timerManager.UpdateCentroidsKernelTimer.TotalElapsedMiliseconds() << " ms." << std::endl;

			if (Parameters.ComputationMethod == "gpu1") {
				std::cout << std::setw(40) << std::left << "SoA to AoS conversion time: "
					<< timerManager.SoA2AoSKernelTimer.TotalElapsedMiliseconds() << " ms." << std::endl;
			}

			std::cout << std::setw(40) << std::left << "Device to Host transfer time: "
				<< timerManager.Device2HostDataTransfer.TotalElapsedMiliseconds() << " ms." << std::endl;
		}

		std::cout << std::endl;
		std::cout << std::setw(40) << std::left << "Total clustering time: "
			<< timerManager.PerformClusteringTimer.TotalElapsedMiliseconds() << " ms." << std::endl;
		std::cout << std::setw(40) << std::left << "Saving data to output file time: " 
			<< timerManager.SaveDataToOutputFileTimer.TotalElapsedMiliseconds() << " ms." << std::endl;
		std::cout << std::setw(40) << std::left << "Loading data from input file time: " 
			<< timerManager.LoadDataFromInputFileTimer.TotalElapsedMiliseconds() << " ms." << std::endl;
	}

	void LoadDataFromInputFile(FILE* inputFile) override {
		std::cout << "Loading data from input file..." << std::endl;

		auto& timerManager = Timers::TimerManager::GetInstance();
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
	}

	void SaveDataToOutputFile(thrust::host_vector<size_t>& membership) override {
		auto& timerManager = Timers::TimerManager::GetInstance();
		timerManager.SaveDataToOutputFileTimer.Start();

		SaveDataToTextFile(membership);

		timerManager.SaveDataToOutputFileTimer.Stop();
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
			for (size_t i = 0; i < dim; ++i) {
				fprintf(outputFile, "%.4f ", centroid.Coordinates[i]);
			}

			fprintf(outputFile, "\n");
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