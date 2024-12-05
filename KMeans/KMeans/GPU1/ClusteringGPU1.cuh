#pragma once

#include "DeviceData.cuh"
#include "../Timers/TimerManager.cuh"

#include <iomanip>

namespace GPU1 {
	// Copies centroids from global memory into shared memory for faster access
	template<size_t dim>
	__device__
	void CopyCentroidsToSharedMemory(DeviceRawData<dim>& deviceRawData, float* sharedMemory)
	{
		unsigned blockCount = static_cast<unsigned>(
			(deviceRawData.CentroidsCount + THREADS_IN_ONE_BLOCK - 1) / THREADS_IN_ONE_BLOCK
			);

		for (size_t i = 0; i < dim; ++i) {
			size_t offset = i * deviceRawData.CentroidsCount; // Offset for current dimension
			for (size_t j = 0; j < blockCount; ++j) {
				size_t centroidIndex = THREADS_IN_ONE_BLOCK * j + threadIdx.x; // Calculate thread-specific index
				if (centroidIndex >= deviceRawData.CentroidsCount) break;
				size_t index = centroidIndex + offset;
				sharedMemory[index] = deviceRawData.DeviceCentroids[index]; // Copy centroid coordinate to shared memory
			}
		}

		// Ensure all threads in the warp complete copying before proceeding
		__syncwarp();
		__syncthreads();
	}

	// Computes the squared Euclidean distance between a point and a centroid
	template<size_t dim>
	__device__
	float SquaredCentroidDistance(
		DeviceRawData<dim>& deviceRawData,
		size_t pointIndex,
		size_t centroidIndex,
		float* sharedMemory,
		float* pointCoordinates)
	{
		float distance = 0;

		for (size_t i = 0; i < dim; ++i) {
			float centroidCoordinate = GetCoordinate(
				sharedMemory,
				deviceRawData.CentroidsCount,
				centroidIndex,
				i
			);

			float difference = pointCoordinates[i] - centroidCoordinate;
			distance += difference * difference;
		}

		return distance;
	}

	// Finds the nearest centroid for a given point
	template<size_t dim>
	__device__
	size_t FindNearestCentroid(DeviceRawData<dim>& deviceRawData, size_t pointIndex, float* sharedMemory)
	{
		size_t nearest{};
		float nearestDistance = FLOAT_INFINITY;

		float pointCoordinates[dim];

		for (size_t i = 0; i < dim; ++i) {
			pointCoordinates[i] = GetCoordinate(
				deviceRawData.DevicePoints,
				deviceRawData.PointsCount,
				pointIndex,
				i
			);
		}

		for (size_t i = 0; i < deviceRawData.CentroidsCount; ++i)
		{
			float distance = SquaredCentroidDistance<dim>(deviceRawData, pointIndex, i, sharedMemory, pointCoordinates);
			if (distance < nearestDistance) {
				nearestDistance = distance;
				nearest = i;
			}
		}

		return nearest;
	}

	// Updates the coordinates of centroids based on new assignments
	template<size_t dim>
	__device__
	void ComputeNewCentroids(DeviceRawData<dim> deviceRawData, size_t pointIndex, size_t centroidIndex) 
	{
		// Accumulate point coordinates into the corresponding centroid using atomic operations
		for (size_t i = 0; i < dim; i++) {
			float* centroidCoordinate = &GetCoordinate(
				deviceRawData.DeviceUpdatedCentroids, deviceRawData.CentroidsCount, centroidIndex, i
			);

			float pointCoordinate = GetCoordinate(
				deviceRawData.DevicePoints, deviceRawData.PointsCount, pointIndex, i
			);

			// Atomically add point coordinate to centroid
			atomicAdd(centroidCoordinate, pointCoordinate);
		}

		// Atomically increment the count of points assigned to this centroid
		atomicAdd(&deviceRawData.DeviceUpdatedCentroidsCounts[centroidIndex], 1);
	}

	// Updates the membership of points to the nearest centroid
	template<size_t dim>
	__device__
	inline void UpdateChanges(DeviceRawData<dim>& deviceRawData, size_t pointIndex, size_t centroidIndex)
	{
		// Check if the membership has changed
		if (deviceRawData.DeviceMembership[pointIndex] == centroidIndex) return;

		deviceRawData.DeviceMembership[pointIndex] = centroidIndex;
		deviceRawData.DeviceChanges[pointIndex] = 1;
	}

	// Kernel for finding the nearest centroids for all points in parallel
	template<size_t dim>
	__global__
	void FindNearestCentroidsKernel(DeviceRawData<dim> deviceRawData)
	{
		size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
		extern __shared__ float sharedMemory[];
		CopyCentroidsToSharedMemory<dim>(deviceRawData, sharedMemory);

		if (tid >= deviceRawData.PointsCount) return;

		size_t nearest = FindNearestCentroid<dim>(deviceRawData, tid, sharedMemory);

		// Update membership and compute new centroids
		UpdateChanges(deviceRawData, tid, nearest);
		ComputeNewCentroids<dim>(deviceRawData, tid, nearest);
	}

	// Kernel for updating centroids after all points are assigned
	template<size_t dim>
	__global__
	void UpdateCentroidsKernel(DeviceRawData<dim> deviceRawData) {
		size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

		if (tid >= deviceRawData.CentroidsCount) return;

		size_t count = deviceRawData.DeviceUpdatedCentroidsCounts[tid] == 0
			? (size_t)1
			: deviceRawData.DeviceUpdatedCentroidsCounts[tid];

		// Update centroid coordinates by averaging
		for (size_t i = 0; i < dim; i++) {
			float* coordinate = &GetCoordinate(
				deviceRawData.DeviceCentroids,
				deviceRawData.CentroidsCount,
				tid, i
			);

			float* updatedCoordinate = &GetCoordinate(
				deviceRawData.DeviceUpdatedCentroids,
				deviceRawData.CentroidsCount,
				tid, i
			);

			*coordinate = *updatedCoordinate / count; // Update coordinate
			*updatedCoordinate = 0;					  // Reset updated coordinate
		}

		deviceRawData.DeviceUpdatedCentroidsCounts[tid] = 0; // Reset count for this centroid
	}

	template<size_t dim>
	class ClusteringGPU1 {
	public:
		// Main clustering function. Performs k-means clustering on GPU using self-made kernel functions
		template<size_t dim>
		thrust::host_vector<size_t> PerformClustering(
			thrust::host_vector<Point<dim>>& hostCentroids, 
			thrust::host_vector<Point<dim>>& hostPoints) 
		{
			auto& timerManager = Timers::TimerManager::GetInstance();

			size_t k = hostCentroids.size();
			size_t N = hostPoints.size();
			
			// Transfer data from host to device
			// Create buffers for intermediate results
			DeviceData<dim> deviceData(hostCentroids, hostPoints);

			size_t changes = N;
			size_t iteration = 0, maxIterations = 100;

			std::cout << std::endl << "Starting clustering..." << std::endl;

			// Main clustering loop
			while (iteration++ < maxIterations && changes != 0) {
				std::cout << "\n=== Iteration: " << iteration << " ===" << std::endl;

				changes = 0;

				std::cout << " -> Computing new centroids..." << std::endl;

				timerManager.ComputeNewCentroidsKernelTimer.Start();

				// Assign points to the nearest centroids
				// Count changes in membership
				changes = FindNearestCentroids(deviceData);

				timerManager.ComputeNewCentroidsKernelTimer.Stop();

				std::cout << std::setw(35) << std::left << "    Elapsed time: " 
					<< timerManager.ComputeNewCentroidsKernelTimer.ElapsedMiliseconds() << " ms" << std::endl;
				std::cout << std::setw(35) << std::left << "    Changes in membership: " 
					<< changes << std::endl;
				std::cout << " -> Updating centroids..." << std::endl;

				timerManager.UpdateCentroidsKernelTimer.Start();

				// Update centroids based on assigned points
				UpdateCentroids(deviceData);

				timerManager.UpdateCentroidsKernelTimer.Stop();

				std::cout << std::setw(35) << std::left << "    Elapsed time: " 
					<< timerManager.UpdateCentroidsKernelTimer.ElapsedMiliseconds() << " ms" << std::endl;
			}

			if (changes == 0) {
				std::cout << "\nClustering completed: No changes in membership." << std::endl;
			}
			else {
				std::cout << "\nClustering completed: Maximum number of iterations reached." << std::endl;
			}

			// Transfer data from device back to host
			hostCentroids = deviceData.GetHostCentroids();
			hostPoints = deviceData.DevicePoints.ToHost();
			return deviceData.GetHostMembership();
		}

	private:
		size_t ReduceChanges(thrust::device_vector<size_t>& deviceChanges) 
		{
			size_t reduced = thrust::reduce(deviceChanges.begin(), deviceChanges.end(), 0);
			thrust::fill(deviceChanges.begin(), deviceChanges.end(), 0);
			return reduced;
		}

		void UpdateCentroids(DeviceData<dim>& deviceData) 
		{
			unsigned blockCount = static_cast<unsigned>(
				(deviceData.DeviceCentroids.GetSize() + THREADS_IN_ONE_BLOCK - 1) / THREADS_IN_ONE_BLOCK
			);
			
			UpdateCentroidsKernel << <blockCount, THREADS_IN_ONE_BLOCK >> > (deviceData.ToDeviceRawData());
			
			CUDACHECK(cudaPeekAtLastError());
			CUDACHECK(cudaDeviceSynchronize());
		}

		size_t FindNearestCentroids(DeviceData<dim>& deviceData) 
		{
			unsigned blockCount = static_cast<unsigned>(
				(deviceData.DevicePoints.GetSize() + THREADS_IN_ONE_BLOCK - 1) / THREADS_IN_ONE_BLOCK
			);
			
			FindNearestCentroidsKernel<dim> << <
				blockCount, 
				THREADS_IN_ONE_BLOCK, 
				sizeof(float) * dim * deviceData.DeviceCentroids.GetSize() 
			>> > (deviceData.ToDeviceRawData());
			
			CUDACHECK(cudaPeekAtLastError());
			CUDACHECK(cudaDeviceSynchronize());

			return ReduceChanges(deviceData.DeviceChanges);
		}
	};
}