#pragma once

#include "Point.cuh"
#include "DeviceRawDataGPU.cuh"
#include "DevicePointsCollection.cuh"

#include <cfloat>

#define FLOAT_INFINITY FLT_MAX

namespace CommonGPU 
{
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
	__device__
	float SquaredCentroidDistance(DeviceRawDataGPU<dim>& deviceRawData, size_t pointIndex, size_t centroidIndex, float* sharedMemory, float* pointCoordinates) {
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

	size_t ReduceChanges(thrust::device_vector<size_t>& deviceChanges) {
		size_t reduced = thrust::reduce(deviceChanges.begin(), deviceChanges.end(), 0);
		thrust::fill(deviceChanges.begin(), deviceChanges.end(), 0);
		return reduced;
	}

	template<size_t dim>
	__device__
	void CopyCentroidsToSharedMemory(DeviceRawDataGPU<dim>& deviceRawData, float* sharedMemory) {
		unsigned blockCount = static_cast<unsigned>((deviceRawData.CentroidsCount + THREADS_IN_ONE_BLOCK - 1) / THREADS_IN_ONE_BLOCK);

		for (size_t i = 0; i < dim; ++i) {
			size_t offset = i * deviceRawData.CentroidsCount;
			for (size_t j = 0; j < blockCount; ++j) {
				size_t centroidIndex = THREADS_IN_ONE_BLOCK * j + threadIdx.x;
				if (centroidIndex >= deviceRawData.CentroidsCount) break;
				size_t index = centroidIndex + offset;
				sharedMemory[index] = deviceRawData.DeviceCentroids[index];
			}
		}

		__syncwarp();
		__syncthreads();
	}

	template<size_t dim>
	__device__
	inline void UpdateChanges(DeviceRawDataGPU<dim>& deviceRawData, size_t pointIndex, size_t centroidIndex) {
		if (deviceRawData.DeviceMembership[pointIndex] == centroidIndex) return;
		deviceRawData.DeviceMembership[pointIndex] = centroidIndex;
		deviceRawData.DeviceChanges[pointIndex] = 1;
	}

	template<size_t dim>
	__device__
	size_t FindNearestCentroid(DeviceRawDataGPU<dim>& deviceRawData, size_t pointIndex, float* sharedMemory) {
		size_t nearest{};
		float nearestDistance = FLOAT_INFINITY;

		float pointCoordinates[dim];

		for (size_t i = 0; i < dim; ++i) {
			pointCoordinates[i] = GetCoordinate(
				deviceRawData.DevicePoints,
				deviceRawData.PointsCount,
				deviceRawData.DevicePointsPermutation[pointIndex],
				i
			);
		}

		for (size_t i = 0; i < deviceRawData.CentroidsCount; ++i) {
			float distance = SquaredCentroidDistance<dim>(deviceRawData, pointIndex, i, sharedMemory, pointCoordinates);
			if (distance < nearestDistance) {
				nearestDistance = distance;
				nearest = i;
			}
		}

		return nearest;
	}
}