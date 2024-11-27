#pragma once

#include "DeviceDataGPU1.cuh"

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
			float* coordinate = &DevicePointsCollection<dim>::GetCoordinate(
				deviceRawData.DeviceCentroids,
				deviceRawData.CentroidsCount,
				tid, i
			);

			float* updatedCoordinate = &DevicePointsCollection<dim>::GetCoordinate(
				deviceRawData.DeviceUpdatedCentroids,
				deviceRawData.CentroidsCount,
				tid, i
			);

			*coordinate = *updatedCoordinate;
			*updatedCoordinate = 0;
		}

		deviceRawData.DeviceUpdatedCentroidsCounts[tid] = 0;
	}

	template<size_t dim>
	class ClusteringGPU1 {
	public:

	private:
		void UpdateCentroids(DeviceDataGPU1<dim>& deviceData) {
			size_t blockCount = (deviceData.DeviceCentroids.GetSize() + THREADS_IN_ONE_BLOCK - 1) / THREADS_IN_ONE_BLOCK;
			UpdateCentroidsKernel << <blockCount, THREADS_IN_ONE_BLOCK >> > (deviceData.ToDeviceRawData());
		}

	};
}