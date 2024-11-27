#pragma once

#include "DeviceRawDataGPU1.cuh"

using namespace CommonGPU;

namespace GPU1
{
	template<size_t dim>
	struct DeviceDataGPU1 {
		DevicePointsCollection<dim> DevicePoints;
		DevicePointsCollection<dim> DeviceCentroids;
		DevicePointsCollection<dim> DeviceUpdatedCentroids;
		thrust::device_vector<size_t> DeviceUpdatedCentroidsCounts;
		thrust::device_vector<size_t> DeviceMembership;
		thrust::device_vector<size_t> DeviceChanges;
		thrust::device_vector<size_t> DevicePointsPermutation;

		DeviceDataGPU1(thrust::host_vector<Point<dim>>& hostCentroids, thrust::host_vector<Point<dim>>& hostPoints) :
			DevicePoints(hostPoints),
			DeviceCentroids(hostCentroids),
			DeviceUpdatedCentroids(hostCentroids.size()),
			DeviceUpdatedCentroidsCounts(hostCentroids.size()),
			DeviceMembership(hostPoints.size()),
			DeviceChanges(hostPoints.size()),
			DevicePointsPermutation(hostPoints.size())
		{
			thrust::copy_n(thrust::make_counting_iterator(0), DevicePoints.size(), DevicePointsPermutation.begin());
		}

		DeviceRawDataGPU1<dim> ToDeviceRawData() {
			DeviceRawDataGPU1<dim> result{};

			result.PointsCount = DevicePoints.GetSize();
			result.DevicePoints = DevicePoints.RawAccess();
			result.CentroidsCount = DeviceCentroids.GetSize();
			result.DeviceCentroids = DeviceCentroids.RawAccess();
			result.DeviceUpdatedCentroidsCounts = thrust::raw_pointer_cast(DeviceUpdatedCentroidsCounts.data());
			result.DeviceUpdatedCentroids = DeviceUpdatedCentroids.RawAccess();
			result.DeviceMembership = thrust::raw_pointer_cast(DeviceMembership.data());
			result.DeviceChanges = thrust::raw_pointer_cast(DeviceChanges.data());
			result.DevicePointsPermutation = thrust::raw_pointer_cast(DevicePointsPermutation.data());

			return result;
		}

		thrust::host_vector<size_t> GetHostMembership() {
			return static_cast<thrust::host_vector<size_t>>(DeviceMembership);
		}

		thrust::host_vector<Point<dim>> GetHostCentroids() {
			return DeviceCentroids.ToHost();
		}
	};
}