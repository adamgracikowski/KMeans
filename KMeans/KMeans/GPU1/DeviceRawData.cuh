#pragma once

namespace GPU1
{
	template<size_t dim>
	struct DeviceRawData {
		size_t PointsCount{};
		float* DevicePoints{};
		size_t CentroidsCount{};
		float* DeviceCentroids{};
		size_t* DeviceMembership{};
		size_t* DeviceChanges{};
		float* DeviceUpdatedCentroids{};
		unsigned* DeviceUpdatedCentroidsCounts{};
	};
}