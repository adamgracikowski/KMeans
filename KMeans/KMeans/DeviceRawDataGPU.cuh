#pragma once

namespace CommonGPU
{
	template<size_t dim>
	struct DeviceRawDataGPU {
		size_t PointsCount{};
		float* DevicePoints{};
		size_t CentroidsCount{};
		float* DeviceCentroids{};
		size_t* DeviceMembership{};
		float* DeviceChanges{};
		size_t* DevicePointsPermutation{};
	};
}