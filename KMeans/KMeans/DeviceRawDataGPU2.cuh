#pragma once

#include "DevicePointsCollection.cuh"
#include "DeviceRawDataGPU.cuh"

using namespace CommonGPU;

namespace GPU2
{
	template<size_t dim>
	struct DeviceRawDataGPU2 : public DeviceRawDataGPU<dim> {
		float* DeviceReducedPoints{};
		size_t* DeviceReducedMembership{};
		size_t* DeviceReducedPointsCounts{};
		size_t DeviceReducedPointsCount{};
	};
}