#pragma once

#include "DeviceDataGPU1.cuh"
#include "CommonDeviceTools.cuh"

namespace GPU1 {
	template<size_t dim>
	__global__
	void UpdateCentroidsKernel(DeviceRawDataGPU1<dim> deviceRawData) {
		size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

		if (tid >= deviceRawData.CentroidsCount) return;

		size_t count = deviceRawData.DeviceUpdatedCentroidsCounts[tid] == 0
			? (size_t)1
			: deviceRawData.DeviceUpdatedCentroidsCounts[tid];

		for (size_t i = 0; i < dim; i++) {
			float* coordinate = &CommonGPU::GetCoordinate(
				deviceRawData.DeviceCentroids,
				deviceRawData.CentroidsCount,
				tid, i
			);

			float* updatedCoordinate = &CommonGPU::GetCoordinate(
				deviceRawData.DeviceUpdatedCentroids,
				deviceRawData.CentroidsCount,
				tid, i
			);

			*coordinate = *updatedCoordinate / count;
			*updatedCoordinate = 0;
		}

		deviceRawData.DeviceUpdatedCentroidsCounts[tid] = 0;
	}

	template<size_t dim>
	__device__
	void ComputeNewCentroids(DeviceRawDataGPU1<dim> deviceRawData, size_t pointIndex, size_t centroidIndex) {
		for (size_t i = 0; i < dim; i++) {
			float* centroidCoordinate = &CommonGPU::GetCoordinate(
				deviceRawData.DeviceUpdatedCentroids, deviceRawData.CentroidsCount, centroidIndex, i
			);

			float pointCoordinate = CommonGPU::GetCoordinate(
				deviceRawData.DevicePoints, deviceRawData.PointsCount, pointIndex, i
			);

			atomicAdd(centroidCoordinate, pointCoordinate);
		}

		atomicAdd(&deviceRawData.DeviceUpdatedCentroidsCounts[centroidIndex], 1);
	}

	template<size_t dim>
	__global__
	void FindNearestCentroidsKernel(DeviceRawDataGPU1<dim> deviceRawData) {
		size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
		extern __shared__ float sharedMemory[];
		CommonGPU::CopyCentroidsToSharedMemory<dim>(deviceRawData, sharedMemory);

		if (tid >= deviceRawData.PointsCount) return;

		size_t nearest = CommonGPU::FindNearestCentroid<dim>(deviceRawData, tid, sharedMemory);

		CommonGPU::UpdateChanges(deviceRawData, tid, nearest);
		ComputeNewCentroids<dim>(deviceRawData, tid, nearest);
	}

	template<size_t dim>
	class ClusteringGPU1 {
	public:
		template<size_t dim>
		thrust::host_vector<size_t> PerformClustering(thrust::host_vector<Point<dim>>& hostCentroids, thrust::host_vector<Point<dim>>& hostPoints) {
			DeviceDataGPU1<dim> deviceData(hostCentroids, hostPoints);
			
			size_t changes = hostPoints.size();
			size_t iteration = 0, maxIterations = 100;

			std::cout << "Starting clustering..." << std::endl;
			std::cout << "Number of points: " << hostPoints.size() << ", Number of centroids: " << hostCentroids.size() << std::endl;

			while (iteration++ < maxIterations && changes != 0) {
				std::cout << "\n=== Iteration: " << iteration << " ===" << std::endl;

				changes = 0;

				std::cout << "-> Computing new centroids..." << std::endl;

				// pomiar czasu na znalezienie najbli¿szych centroidów
				changes = FindNearestCentroids(deviceData);

				// stop pomiaru czasu na znalezienie najbli¿szych centroidów

				std::cout << "-> Updating centroids..." << std::endl;

				std::cout << "-> Changes in membership: " << changes << std::endl;

				// pomiar czasu na aktualizacjê po³o¿enia centroidów
				UpdateCentroids(deviceData);
				// stop pomiaru czasu na aktualizacjê po³o¿enia centroidów
			}

			if (changes == 0) {
				std::cout << "\nClustering completed: No changes in membership." << std::endl;
			}
			else {
				std::cout << "\nClustering completed: Maximum number of iterations reached." << std::endl;
			}

			hostCentroids = deviceData.GetHostCentroids();
			hostPoints = deviceData.DevicePoints.ToHost();
			return deviceData.GetHostMembership();
		}

	private:
		void UpdateCentroids(DeviceDataGPU1<dim>& deviceData) {
			unsigned blockCount = static_cast<unsigned>((deviceData.DeviceCentroids.GetSize() + THREADS_IN_ONE_BLOCK - 1) / THREADS_IN_ONE_BLOCK);
			
			UpdateCentroidsKernel << <blockCount, THREADS_IN_ONE_BLOCK >> > (deviceData.ToDeviceRawData());
			
			CUDACHECK(cudaPeekAtLastError());
			CUDACHECK(cudaDeviceSynchronize());
		}

		size_t FindNearestCentroids(DeviceDataGPU1<dim>& deviceData) {
			unsigned blockCount = static_cast<unsigned>((deviceData.DevicePoints.GetSize() + THREADS_IN_ONE_BLOCK - 1) / THREADS_IN_ONE_BLOCK);
			
			FindNearestCentroidsKernel<dim> << <blockCount, THREADS_IN_ONE_BLOCK, sizeof(float) * dim * deviceData.DeviceCentroids.GetSize() >> > (deviceData.ToDeviceRawData());
			
			CUDACHECK(cudaPeekAtLastError());
			CUDACHECK(cudaDeviceSynchronize());

			return ReduceChanges(deviceData.DeviceChanges);
		}
	};
}