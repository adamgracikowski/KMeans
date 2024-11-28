#pragma once

#include "DeviceRawDataGPU2.cuh"

using namespace CommonGPU;

namespace GPU2
{
	template<size_t dim>
	struct DeviceDataGPU2 {
		thrust::host_vector<Point<dim>>* HostPoints;
		thrust::host_vector<Point<dim>>* HostCentroids;
		DevicePointsCollection<dim> DevicePoints;
		DevicePointsCollection<dim> DeviceCentroids;
		thrust::device_vector<size_t> DeviceMembership;
		thrust::device_vector<size_t> DeviceChanges;

		DevicePointsCollection<dim> DeviceReducedPoints;
		thrust::device_vector<size_t> DeviceReducedMembership;
		thrust::device_vector<size_t> DeviceReducedPointsCounts;
		thrust::device_vector<size_t> DevicePointsPermutation;

		DeviceDataGPU2(thrust::host_vector<Point<dim>>& hostCentroids, thrust::host_vector<Point<dim>>& hostPoints) : 
			DeviceCentroids(hostCentroids),
			DevicePoints(hostPoints),
			DeviceMembership(hostPoints.size()),
			DeviceChanges(hostPoints.size()),
			DeviceReducedPoints(hostCentroids.size()),
			DeviceReducedMembership(hostCentroids.size()),
			DeviceReducedPointsCounts(hostCentroids.size()),
			DevicePointsPermutation(hostPoints.size()),
			HostPoints{ &hostPoints },
			HostCentroids{ &hostCentroids }
		{
			thrust::copy_n(thrust::make_counting_iterator(0), hostPoints.size(), DevicePointsPermutation.begin());
		}

		DeviceRawDataGPU2<dim> ToDeviceRawData() {
			DeviceRawDataGPU2<dim> result{};

			result.PointsCount = DevicePoints.GetSize();
			result.DevicePoints = DevicePoints.RawAccess();
			result.CentroidsCount = DeviceCentroids.GetSize();
			result.DeviceCentroids = DeviceCentroids.RawAccess();
			result.DeviceMembership = thrust::raw_pointer_cast(DeviceMembership.data());
			result.DeviceChanges = thrust::raw_pointer_cast(DeviceChanges.data());
			result.DeviceReducedPoints = DeviceReducedPoints.RawAccess();
			result.DeviceReducedMembership = thrust::raw_pointer_cast(DeviceReducedMembership.data());
			result.DeviceReducedPointsCounts = thrust::raw_pointer_cast(DeviceReducedPointsCounts.data());
			result.DevicePointsPermutation = thrust::raw_pointer_cast(DevicePointsPermutation.data());

			return result;
		}

		thrust::host_vector<size_t> GetHostMembership() {
			thrust::device_vector<size_t> deviceReversedPermutation(DeviceMembership.size());
			thrust::scatter(
				DeviceMembership.begin(), 
				DeviceMembership.end(), 
				deviceReversedPermutation.begin(), 
				deviceReversedPermutation.begin()
			);

			return static_cast<thrust::host_vector<size_t>>(deviceReversedPermutation);
		}

		thrust::host_vector<Point<dim>> GetHostCentroids() {
			return DeviceCentroids.ToHost();
		}
	};
}