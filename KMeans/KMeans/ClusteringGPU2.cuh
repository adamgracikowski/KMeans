#pragma once

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
	template<size_t dim>
	struct AssignPointsToCentroids {
		const Point<dim>* devicePoints;
		const Point<dim>* deviceCentroids;
		size_t k;

		AssignPointsToCentroids(
			const thrust::device_vector<Point<dim>>& devicePoints,
			const thrust::device_vector<Point<dim>>& deviceCentroids) : 
			devicePoints(thrust::raw_pointer_cast(devicePoints.data())),
			deviceCentroids(thrust::raw_pointer_cast(deviceCentroids.data())),
			k(deviceCentroids.size()) {	}

		__device__ size_t operator()(size_t pointIndex) const 
		{
			const Point<dim>& point = devicePoints[pointIndex];
			float nearestDistance = FLOAT_INFINITY;
			size_t nearest = 0;

			for (size_t i = 0; i < k; ++i) {
				float distance = Point<dim>::SquareDistance(point, deviceCentroids[i]);
				if (distance < nearestDistance) {
					nearestDistance = distance;
					nearest = i;
				}
			}

			return nearest;
		}
	};

	template<size_t dim>
	struct SumPoints {
		__device__ Point<dim> operator()(Point<dim>& firstPoint, Point<dim>& secondPoint) const 
		{
			Point<dim> result{};

			for (int i = 0; i < dim; ++i) {
				result.Coordinates[i] = firstPoint.Coordinates[i] + secondPoint.Coordinates[i];
			}

			return result;
		}
	};

	template<size_t dim>
	struct ComputeCentroid {
		__device__ Point<dim> operator()(Point<dim>& summedPoint, size_t count) const 
		{
			Point<dim> result;

			count = count == 0 ? 1 : count;

			for (int i = 0; i < dim; ++i) {
				result.Coordinates[i] = summedPoint.Coordinates[i] / count;
			}

			return result;
		}
	};

	struct CheckIfNotEqual {
		__device__
			bool operator()(const thrust::tuple<size_t, size_t>& t) const 
		{
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
			auto& timerManager = Timers::TimerManager::GetInstance();

			size_t k = hostCentroids.size();
			size_t N = hostPoints.size();

			timerManager.Host2DeviceDataTransfer.Start();

			thrust::device_vector<Point<dim>> deviceCentroids = hostCentroids;
			thrust::device_vector<Point<dim>> devicePoints = hostPoints;

			timerManager.Host2DeviceDataTransfer.Stop();

			thrust::device_vector<Point<dim>> deviceSums(k);
			thrust::device_vector<size_t> deviceCounts(k, 0);
			thrust::device_vector<size_t> deviceCentroidsToUpdate(k, 0);
			thrust::device_vector<size_t> devicePermutation(N, 0);

			thrust::device_vector<size_t> deviceMembership(N, 0);
			thrust::device_vector<size_t> devicePreviousMembership(N, 0);

			size_t changes = N;
			size_t iteration = 0, maxIterations = 100;

			std::cout << std::endl << "Starting clustering..." << std::endl;

			while (iteration++ < maxIterations && changes != 0) {
				std::cout << "\n=== Iteration: " << iteration << " ===" << std::endl;

				changes = 0;

				std::cout << " -> Computing new centroids..." << std::endl;

				timerManager.ComputeNewCentroidsKernelTimer.Start();

				thrust::sequence(devicePermutation.begin(), devicePermutation.end(), 0);

				thrust::transform(
					devicePermutation.begin(),
					devicePermutation.end(),
					deviceMembership.begin(),
					AssignPointsToCentroids<dim>(devicePoints, deviceCentroids)
				);

				changes = thrust::count_if(
					thrust::make_zip_iterator(thrust::make_tuple(deviceMembership.begin(), devicePreviousMembership.begin())),
					thrust::make_zip_iterator(thrust::make_tuple(deviceMembership.end(), devicePreviousMembership.end())),
					CheckIfNotEqual()
				);

				timerManager.ComputeNewCentroidsKernelTimer.Stop();

				std::cout << std::setw(35) << std::left << "    Elapsed time: " 
					<< timerManager.ComputeNewCentroidsKernelTimer.ElapsedMiliseconds() << " ms" << std::endl;
				std::cout << std::setw(35) << std::left << "    Changes in membership: " 
					<< changes << std::endl;

				if (changes == 0 || iteration == maxIterations) break;

				std::cout << " -> Updating centroids..." << std::endl;

				timerManager.UpdateCentroidsKernelTimer.Start();

				thrust::copy(
					deviceMembership.begin(),
					deviceMembership.end(),
					devicePreviousMembership.begin()
				);

				thrust::sort_by_key(
					deviceMembership.begin(),
					deviceMembership.end(),
					devicePermutation.begin()
				);

				thrust::reduce_by_key(
					deviceMembership.begin(),
					deviceMembership.end(),
					thrust::make_permutation_iterator(
						devicePoints.begin(), 
						devicePermutation.begin()
					),
					deviceCentroidsToUpdate.begin(),
					deviceSums.begin(),
					thrust::equal_to<size_t>()
				);

				auto updatedEnd = thrust::reduce_by_key(
					deviceMembership.begin(),
					deviceMembership.end(),
					thrust::make_constant_iterator(1),
					deviceCentroidsToUpdate.begin(),
					deviceCounts.begin(),
					thrust::equal_to<size_t>()
				);

				size_t howManyUpdated = thrust::distance(
					deviceCentroidsToUpdate.begin(), 
					updatedEnd.first
				);

				thrust::transform(
					deviceSums.begin(),
					deviceSums.begin() + howManyUpdated,
					deviceCounts.begin(),
					deviceSums.begin(),
					ComputeCentroid<dim>()
				);

				thrust::scatter(
					deviceSums.begin(),
					deviceSums.begin() + howManyUpdated,
					deviceCentroidsToUpdate.begin(),
					deviceCentroids.begin()
				);

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

			timerManager.Device2HostDataTransfer.Start();

			thrust::copy(
				deviceCentroids.begin(), 
				deviceCentroids.end(), 
				hostCentroids.begin()
			);

			thrust::host_vector<size_t> hostMembership(deviceMembership);

			timerManager.Device2HostDataTransfer.Stop();

			return hostMembership;
		}
	};
}