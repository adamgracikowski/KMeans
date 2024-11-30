#pragma once

#include "DeviceDataGPU2.cuh"
#include "CommonDeviceTools.cuh"

#include <thrust/sequence.h>
#include "thrust/iterator/permutation_iterator.h"
#include <thrust/transform.h>

#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/unique.h>
#include <thrust//iterator//constant_iterator.h>
#include <thrust/execution_policy.h>

#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scatter.h>
#include "thrust/distance.h"

namespace GPU2
{
	//template<size_t dim>
	//__global__
	//	void UpdateCentroidsKernel(DeviceRawDataGPU2<dim> deviceRawData) {
	//	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	//	if (tid >= deviceRawData.DeviceReducedPointsCount) return;

	//	size_t centroidIndex = deviceRawData.DeviceReducedMembership[tid];

	//	for (size_t i = 0; i < dim; ++i) {
	//		float* coordinate = &CommonGPU::GetCoordinate(
	//			deviceRawData.DeviceCentroids,
	//			deviceRawData.CentroidsCount,
	//			centroidIndex, i
	//		);

	//		float reducedCoordinate = CommonGPU::GetCoordinate(
	//			deviceRawData.DeviceReducedPoints,
	//			deviceRawData.DeviceReducedPointsCount,
	//			tid, i
	//		);

	//		*coordinate = reducedCoordinate / deviceRawData.DeviceReducedPointsCounts[tid];
	//	}
	//}

	//template<size_t dim>
	//__global__
	//	void FindNearestCentroidsKernel(DeviceRawDataGPU2<dim> deviceRawData) {
	//	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	//	extern __shared__ float sharedMemory[];
	//	CommonGPU::CopyCentroidsToSharedMemory<dim>(deviceRawData, sharedMemory);

	//	if (tid >= deviceRawData.PointsCount) return;

	//	size_t nearest = CommonGPU::FindNearestCentroid<dim>(deviceRawData, tid, sharedMemory);

	//	CommonGPU::UpdateChanges(deviceRawData, tid, nearest);
	//}

	//template<size_t dim>
	//struct AssignPointsToCentroids {
	//	const Point<dim>* devicePoints;
	//	const Point<dim>* deviceCentroids;
	//	size_t k;

	//	AssignPointsToCentroids(
	//		const thrust::device_vector<Point<dim>>& points,
	//		const thrust::device_vector<Point<dim>>& centroids) :
	//		devicePoints(thrust::raw_pointer_cast(points.data())),
	//		deviceCentroids(thrust::raw_pointer_cast(centroids.data())),
	//		k(centroids.size()) {
	//	}

	//	__device__
	//		thrust::tuple<size_t, bool> operator()(size_t pointIndex) const {
	//		const Point<dim>& point = devicePoints[pointIdx];
	//		float nearestDistance = FLOAT_INFINITY;
	//		size_t nearest = {};

	//		for (size_t i = 0; i < k; ++i) {
	//			float distance = Point<dim>::SquareDistance(point, deviceCentroids[i]);
	//			if (distance < nearestDistance) {
	//				nearestDistance = distance;
	//				nearest = i;
	//			}
	//		}

	//		size_t oldAssignment = pointIdx;
	//		bool membershipChanged = (nearest != oldAssignment);

	//		return thrust::make_tuple(nearest, membershipChanged);
	//	}
	//};

	//template<size_t dim>
	//struct PointSum {
	//	__device__ Point<dim> operator()(const Point<dim>& a, const Point<dim>& b) const {
	//		Point<dim> result = a;
	//		result += b;
	//		return result;
	//	}
	//};

	//template<size_t dim>
	//struct ComputeCentroid {
	//	__device__ Point<dim> operator()(const Point<dim>& summedPoint, size_t count) const {
	//		Point<dim> result;
	//		for (int i = 0; i < dim; ++i) {
	//			result.Coordinates[i] = summedPoint.Coordinates[i] / count;
	//		}
	//		return result;
	//	}
	//};

	//template <size_t dim>
	//struct AssignToCentroid {
	//	const Point<dim>* centroids;
	//	size_t numCentroids;

	//	AssignToCentroid(const Point<dim>* centroids, size_t numCentroids)
	//		: centroids(centroids), numCentroids(numCentroids) {
	//	}

	//	__device__
	//		size_t operator()(const Point<dim>& point) const {
	//		size_t closestCentroid = 0;
	//		float minDistance = Point<dim>::SquareDistance(point, centroids[0]);
	//		for (size_t j = 1; j < numCentroids; ++j) {
	//			float distance = Point<dim>::SquareDistance(point, centroids[j]);
	//			if (distance < minDistance) {
	//				closestCentroid = j;
	//				minDistance = distance;
	//			}
	//		}
	//		return closestCentroid;
	//	}
	//};

	template<size_t dim>
	struct AssignPointsToCentroids {
		const Point<dim>* devicePoints;
		const Point<dim>* deviceCentroids;
		size_t numCentroids;

		AssignPointsToCentroids(const thrust::device_vector<Point<dim>>& points,
			const thrust::device_vector<Point<dim>>& centroids)
			: devicePoints(thrust::raw_pointer_cast(points.data())),
			deviceCentroids(thrust::raw_pointer_cast(centroids.data())),
			numCentroids(centroids.size()) {
		}

		__device__ size_t operator()(size_t pointIdx) const {
			const Point<dim>& point = devicePoints[pointIdx];
			float minDistance = FLOAT_INFINITY;
			size_t bestCentroid = 0;

			for (size_t centroidIdx = 0; centroidIdx < numCentroids; ++centroidIdx) {
				float dist = Point<dim>::SquareDistance(point, deviceCentroids[centroidIdx]);
				if (dist < minDistance) {
					minDistance = dist;
					bestCentroid = centroidIdx;
				}
			}

			return bestCentroid;
		}
	};

	template<size_t dim>
	struct PointSum {
		__device__ Point<dim> operator()(Point<dim>& a, Point<dim>& b) const {
			Point<dim> result{};
			for (int i = 0; i < dim; ++i) {
				result.Coordinates[i] = a.Coordinates[i] + b.Coordinates[i];
			}

			return result;
		}
	};

	template<size_t dim>
	struct ComputeCentroid {
		__device__ Point<dim> operator()(Point<dim>& summedPoint, size_t count) const {
			Point<dim> result;
			count = count == 0 ? 1 : count;
			for (int i = 0; i < dim; ++i) {
				result.Coordinates[i] = summedPoint.Coordinates[i] / count;
			}
			return result;
		}
	};

	struct NotEqual {
		__device__
			bool operator()(const thrust::tuple<size_t, size_t>& t) const {
			return thrust::get<0>(t) != thrust::get<1>(t);
		}
	};

	template<size_t dim>
	class ClusteringGPU2 {
	public:
		template<size_t dim>
		thrust::host_vector<size_t> PerformClustering(
			thrust::host_vector<Point<dim>>& hostCentroids,
			thrust::host_vector<Point<dim>>& hostPoints)
		{
			size_t k = hostCentroids.size();
			size_t N = hostPoints.size();

			std::cout << "here" << std::endl;

			thrust::device_vector<Point<dim>> deviceCentroids = hostCentroids;
			thrust::device_vector<Point<dim>> devicePoints = hostPoints;

			std::cout << "here" << std::endl;

			thrust::device_vector<Point<dim>> sums(k);
			thrust::device_vector<size_t> counts(k, 0);
			thrust::device_vector<size_t> centroidsToUpdate(k, 0);
			thrust::device_vector<size_t> permutation(N, 0);
			std::cout << "here" << std::endl;

			thrust::device_vector<size_t> membership(N, 0);
			thrust::device_vector<size_t> previousMembership(N, 0);
			std::cout << "here" << std::endl;

			size_t changes = N;
			size_t iteration = 0, maxIterations = 100;

			while (iteration++ < maxIterations && changes != 0) {
				std::cout << "\n=== Iteration: " << iteration << " ===" << std::endl;

				thrust::sequence(permutation.begin(), permutation.end(), 0);

				thrust::transform(
					permutation.begin(),
					permutation.end(),
					membership.begin(),
					AssignPointsToCentroids<dim>(devicePoints, deviceCentroids)
				);

				//{
				//	thrust::host_vector<size_t> h_membership = membership;
				//	thrust::host_vector<size_t> h_permutation = permutation;
				//	std::cout << "Membership: ";
				//	for (size_t i = 0; i < N; ++i) std::cout << h_membership[i] << " ";
				//	std::cout << "\nPermutation: ";
				//	for (size_t i = 0; i < N; ++i) std::cout << h_permutation[i] << " ";
				//	std::cout << std::endl;
				//}


				changes = thrust::count_if(
					thrust::make_zip_iterator(thrust::make_tuple(membership.begin(), previousMembership.begin())),
					thrust::make_zip_iterator(thrust::make_tuple(membership.end(), previousMembership.end())),
					NotEqual()
				);

				std::cout << std::setw(35) << std::left << "    Changes in membership: " << changes << std::endl;

				if (changes == 0 || iteration == maxIterations) break;

				thrust::copy(
					membership.begin(),
					membership.end(),
					previousMembership.begin()
				);

				std::cout << "here" << std::endl;


				thrust::sort_by_key(
					membership.begin(),
					membership.end(),
					permutation.begin()
				);

				//{
				//	thrust::host_vector<size_t> h_membership = membership;
				//	thrust::host_vector<size_t> h_permutation = permutation;
				//	std::cout << "Membership: ";
				//	for (size_t i = 0; i < N; ++i) std::cout << h_membership[i] << " ";
				//	std::cout << "\nPermutation: ";
				//	for (size_t i = 0; i < N; ++i) std::cout << h_permutation[i] << " ";
				//	std::cout << std::endl;
				//}

				std::cout << "here" << std::endl;


				auto end1 = thrust::reduce_by_key(
					membership.begin(),
					membership.end(),
					thrust::make_permutation_iterator(devicePoints.begin(), permutation.begin()),
					centroidsToUpdate.begin(),
					sums.begin(),
					thrust::equal_to<size_t>()
				);

				std::cout << "here" << std::endl;

				auto end2 = thrust::reduce_by_key(
					membership.begin(),
					membership.end(),
					thrust::make_constant_iterator(1),
					centroidsToUpdate.begin(),
					counts.begin(),
					thrust::equal_to<size_t>()
				);

				size_t howMany = thrust::distance(centroidsToUpdate.begin(), end2.first);
				std::cout << "here howMany = " << howMany << std::endl; // po tym jest coœ Ÿle

				//{
				//	//thrust::host_vector<Point<dim>> h_sums = sums;
				//	thrust::host_vector<size_t> h_counts = counts;
				//	thrust::host_vector<size_t> h_membership = membership;

				//	//std::cout << "Sums: ";
				//	//for (size_t i = 0; i < k; ++i) std::cout << h_sums[i] << " ";
				//	std::cout << "\nCounts: ";
				//	for (size_t i = 0; i < k; ++i) std::cout << h_counts[i] << " ";
				//	std::cout << std::endl;
				//	std::cout << "Membership: ";
				//	for (size_t i = 0; i < N; ++i) std::cout << h_membership[i] << " ";
				//	std::cout << std::endl;
				//	thrust::host_vector<size_t> h_centroidsToUpdate = centroidsToUpdate;
				//	std::cout << "Centroids to update: ";
				//	for (size_t i = 0; i < k; ++i) std::cout << h_centroidsToUpdate[i] << " ";
				//	std::cout << std::endl;
				//}


				std::cout << "here" << std::endl;

				//{
				//	thrust::host_vector<size_t> h_centroidsToUpdate = centroidsToUpdate;
				//	std::cout << "Centroids to update: ";
				//	for (size_t i = 0; i < howMany; ++i) std::cout << h_centroidsToUpdate[i] << " ";
				//	std::cout << std::endl;
				//}

				thrust::transform(
					sums.begin(),
					sums.begin() + howMany,
					counts.begin(),
					sums.begin(),
					ComputeCentroid<dim>()
				);
				std::cout << "here" << std::endl;

				thrust::scatter(
					sums.begin(),
					sums.begin() + howMany,
					centroidsToUpdate.begin(),
					deviceCentroids.begin()
				);
			}

			thrust::copy(deviceCentroids.begin(), deviceCentroids.end(), hostCentroids.begin());

			return thrust::host_vector<size_t>(membership);
		}
	};
	//template<size_t dim>
	//thrust::host_vector<size_t> PerformClustering(thrust::host_vector<Point<dim>>& hostCentroids, thrust::host_vector<Point<dim>>& hostPoints) {
	//	
	//	size_t k = hostCentroids.size();
	//	size_t N = hostPoints.size();
	//	
	//	thrust::device_vector<Point<dim>> devicePoints = hostPoints;
	//	thrust::device_vector<Point<dim>> deviceCentroids = hostCentroids;

	//	thrust::device_vector<size_t> deviceMemberships(N, 0);
	//	thrust::device_vector<size_t> deviceChanges(N, 0);
	//	thrust::device_vector<size_t> deviceCentroidsCounts(k, 0);

	//	size_t changes = N;
	//	size_t iteration = 0, maxIterations = 100;

	//	std::cout << std::endl << "Starting clustering..." << std::endl;

	//	while (iteration++ < maxIterations && changes != 0) {
	//		std::cout << "\n=== Iteration: " << iteration << " ===" << std::endl;

	//		changes = 0;

	//		thrust::transform(
	//			devicePoints.begin(),
	//			devicePoints.end(),
	//			deviceMemberships.begin(),
	//			AssignToCentroid<dim>(thrust::raw_pointer_cast(deviceCentroids.data()), k)
	//		);
	//	}

	//	
	//	
	//	return thrust::host_vector<size_t>{};
	//	//DeviceDataGPU2<dim> deviceData(hostCentroids, hostPoints);

	//	//size_t changes = hostPoints.size();
	//	//size_t iteration = 0, maxIterations = 100;

	//	//std::cout << "Starting clustering..." << std::endl;
	//	//std::cout << "Number of points: " << hostPoints.size() << ", Number of centroids: " << hostCentroids.size() << std::endl;

	//	//while (iteration++ < maxIterations && changes != 0) {
	//	//	std::cout << "\n=== Iteration: " << iteration << " ===" << std::endl;

	//	//	changes = 0;

	//	//	std::cout << "-> Computing new centroids..." << std::endl;

	//	//	// pomiar czasu na znalezienie najbli¿szych centroidów
	//	//	changes = FindNearestCentroids(deviceData);
	//	//	// stop pomiaru czasu na znalezienie najbli¿szych centroidów

	//	//	std::cout << "-> Updating centroids..." << std::endl;

	//	//	std::cout << "-> Changes in membership: " << changes << std::endl;

	//	//	// pomiar czasu na aktualizacjê po³o¿enia centroidów
	//	//	UpdateCentroids(deviceData);
	//	//	// stop pomiaru czasu na aktualizacjê po³o¿enia centroidów
	//	//}

	//	//if (changes == 0) {
	//	//	std::cout << "\nClustering completed: No changes in membership." << std::endl;
	//	//}
	//	//else {
	//	//	std::cout << "\nClustering completed: Maximum number of iterations reached." << std::endl;
	//	//}

	//	//hostCentroids = deviceData.GetHostCentroids();
	//	//hostPoints = deviceData.DevicePoints.ToHost();
	//	//return deviceData.GetHostMembership();
	//}

	/*size_t FindNearestCentroids(DeviceDataGPU2<dim>& deviceData) {
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
	}*/
}