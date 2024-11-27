#pragma once

#include "DevicePointsCollection.cuh"
#include "DeviceRawDataGPU.cuh"

using namespace CommonGPU;

namespace GPU1
{
	template<size_t dim>
	struct DeviceRawDataGPU1 : public DeviceRawDataGPU<dim> {
		float* DeviceUpdatedCentroids{};
		size_t* DeviceUpdatedCentroidsCounts{};
	};
}