#pragma once

#include "DeviceDataGPU2.cuh"
#include "CommonDeviceTools.cuh"

#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/unique.h>
#include <thrust/iterator//discard_iterator.h>
#include <thrust//iterator//constant_iterator.h>
#include <thrust/execution_policy.h>

namespace GPU2
{
	template<size_t dim>
	__global__
		void UpdateCentroidsKernel(DeviceRawDataGPU2<dim> deviceRawData) {
		size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

		if (tid >= deviceRawData.DeviceReducedPointsCount) return;

		size_t centroidIndex = deviceRawData.DeviceReducedMembership[tid];

		for (size_t i = 0; i < dim; ++i) {
			float* coordinate = &CommonGPU::GetCoordinate(
				deviceRawData.DeviceCentroids,
				deviceRawData.CentroidsCount,
				centroidIndex, i
			);

			float reducedCoordinate = CommonGPU::GetCoordinate(
				deviceRawData.DeviceReducedPoints,
				deviceRawData.DeviceReducedPointsCount,
				tid, i
			);

			*coordinate = reducedCoordinate / deviceRawData.DeviceReducedPointsCounts[tid];
		}
	}

	template<size_t dim>
	__global__
		void FindNearestCentroidsKernel(DeviceRawDataGPU2<dim> deviceRawData) {
		size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
		extern __shared__ float sharedMemory[];
		CommonGPU::CopyCentroidsToSharedMemory<dim>(deviceRawData, sharedMemory);

		if (tid >= deviceRawData.PointsCount) return;

		size_t nearest = CommonGPU::FindNearestCentroid<dim>(deviceRawData, tid, sharedMemory);

		CommonGPU::UpdateChanges(deviceRawData, tid, nearest);
	}

	template<size_t dim>
	class ClusteringGPU2 {
	public:
		template<size_t dim>
		thrust::host_vector<size_t> PerformClustering(thrust::host_vector<Point<dim>>& hostCentroids, thrust::host_vector<Point<dim>>& hostPoints) {
			DeviceDataGPU2<dim> deviceData(hostCentroids, hostPoints);

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
		size_t FindNearestCentroids(DeviceDataGPU2<dim>& deviceData) {
			unsigned blockCount = static_cast<unsigned>((deviceData.DevicePoints.GetSize() + THREADS_IN_ONE_BLOCK - 1) / THREADS_IN_ONE_BLOCK);

			FindNearestCentroidsKernel<dim> << <blockCount, THREADS_IN_ONE_BLOCK, sizeof(float)* dim* deviceData.DeviceCentroids.GetSize() >> > (deviceData.ToDeviceRawData());
			return ReduceChanges(deviceData.DeviceChanges);
		}

		void UpdateCentroids(DeviceDataGPU2<dim>& deviceData) {
			thrust::sort_by_key(
				deviceData.DeviceMembership.begin(),
				deviceData.DeviceMembership.end(),
				deviceData.DevicePointsPermutation.begin()
			);

			thrust::unique_copy(
				deviceData.DeviceMembership.begin(),
				deviceData.DeviceMembership.end(),
				deviceData.DeviceReducedMembership.begin()
			);

			for (size_t i = 0; i < dim; ++i) {
				auto permutationIterator = thrust::make_permutation_iterator(
					deviceData.DevicePoints.RawAccess() + i * deviceData.DevicePoints.GetSize(),
					deviceData.DevicePointsPermutation.begin()
				);

				thrust::reduce_by_key(
					thrust::device,
					deviceData.DeviceMembership.begin(),
					deviceData.DeviceMembership.end(),
					permutationIterator,
					thrust::make_discard_iterator(),
					deviceData.DeviceReducedPoints.RawAccess() + i * deviceData.DeviceReducedPoints.GetSize()
				);
			}

			auto reducedEnd = thrust::reduce_by_key(
				deviceData.DeviceMembership.begin(),
				deviceData.DeviceMembership.end(),
				thrust::make_constant_iterator(1),
				thrust::make_discard_iterator(),
				deviceData.DeviceReducedPointsCounts.begin()
			);

			size_t reducedPointsCount = reducedEnd.second - deviceData.DeviceReducedPointsCounts.begin();
			DeviceRawDataGPU2<dim> deviceRawData = deviceData.ToDeviceRawData();
			deviceRawData.DeviceReducedPointsCount = reducedPointsCount;

			unsigned blockCount = static_cast<unsigned>((reducedPointsCount + THREADS_IN_ONE_BLOCK - 1) / THREADS_IN_ONE_BLOCK);
			UpdateCentroidsKernel << <blockCount, THREADS_IN_ONE_BLOCK >> > (deviceRawData);
		}
	};
}