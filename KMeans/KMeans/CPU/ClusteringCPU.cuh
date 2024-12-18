#pragma once

#include "../Timers/TimerManager.cuh"
#include "../Point.cuh"

#include <iomanip>

using namespace Timers;
using namespace DataStructures;

namespace CPU
{
	template<size_t dim>
	class ClusteringCPU {
	public:
		thrust::host_vector<size_t> PerformClustering(
			thrust::host_vector<Point<dim>>& centroids,
			thrust::host_vector<Point<dim>>& points);

	private:
		size_t FindNearestCentroid(
			Point<dim> point,
			thrust::host_vector<Point<dim>>& centroids);

		void ComputeNewCentroids(
			thrust::host_vector<Point<dim>>& points,
			thrust::host_vector<Point<dim>>& centroids,
			thrust::host_vector<Point<dim>>& updatedCentroids,
			thrust::host_vector<size_t>& updatedCounts,
			thrust::host_vector<size_t>& membership,
			size_t& changes);

		void UpdateCentroids(
			thrust::host_vector<Point<dim>>& centroids,
			thrust::host_vector<Point<dim>>& updatedCentroids,
			thrust::host_vector<size_t>& updatedCounts);
	};

	// Main clustering function. Performs k-means clustering on CPU
	template<size_t dim>
	thrust::host_vector<size_t> ClusteringCPU<dim>::PerformClustering(
		thrust::host_vector<Point<dim>>& centroids,
		thrust::host_vector<Point<dim>>& points)
	{
		auto& timerManager = Timers::TimerManager::GetInstance();

		// Buffers for intermediate results
		thrust::host_vector<Point<dim>> updatedCentroids(centroids.size());
		thrust::host_vector<size_t> updatedCounts(centroids.size());
		thrust::host_vector<size_t> membership(points.size());

		size_t changes = points.size();
		size_t iteration = 0, maxIterations = 100;

		std::cout << std::endl << "Starting clustering..." << std::endl;

		// Main clustering loop
		while (iteration++ < maxIterations && changes != 0) {
			std::cout << "\n=== Iteration: " << iteration << " ===" << std::endl;

			changes = 0;

			std::cout << " -> Computing new centroids..." << std::endl;

			// Assign points to the nearest centroids and count changes
			timerManager.ComputeNewCentroidsTimer.Start();
			ComputeNewCentroids(points, centroids, updatedCentroids, updatedCounts, membership, changes);
			timerManager.ComputeNewCentroidsTimer.Stop();

			std::cout << std::setw(35) << std::left << "    Elapsed time: "
				<< timerManager.ComputeNewCentroidsTimer.ElapsedMiliseconds() << " ms" << std::endl;
			std::cout << std::setw(35) << std::left << "    Changes in membership: "
				<< changes << std::endl;
			std::cout << " -> Updating centroids..." << std::endl;

			// Update centroids based on assigned points
			timerManager.UpdateCentroidsTimer.Start();
			UpdateCentroids(centroids, updatedCentroids, updatedCounts);
			timerManager.UpdateCentroidsTimer.Stop();

			std::cout << std::setw(35) << std::left << "    Elapsed time: "
				<< timerManager.UpdateCentroidsTimer.ElapsedMiliseconds() << " ms" << std::endl;
		}

		if (changes == 0) {
			std::cout << "\nClustering completed: No changes in membership." << std::endl;
		}
		else {
			std::cout << "\nClustering completed: Maximum number of iterations reached." << std::endl;
		}

		return membership;
	}

	template<size_t dim>
	size_t ClusteringCPU<dim>::FindNearestCentroid(
		Point<dim> point,
		thrust::host_vector<Point<dim>>& centroids)
	{
		size_t nearest{};
		auto nearestDistance = std::numeric_limits<float>::infinity();
		auto k = centroids.size();

		for (size_t i = 0; i < k; ++i) {
			auto distance = Point<dim>::SquareDistance(point, centroids[i]);
			if (distance < nearestDistance) {
				nearestDistance = distance;
				nearest = i;
			}
		}

		return nearest;
	}

	template<size_t dim>
	void ClusteringCPU<dim>::ComputeNewCentroids(
		thrust::host_vector<Point<dim>>& points,
		thrust::host_vector<Point<dim>>& centroids,
		thrust::host_vector<Point<dim>>& updatedCentroids,
		thrust::host_vector<size_t>& updatedCounts,
		thrust::host_vector<size_t>& membership,
		size_t& changes)
	{
		auto N = points.size();

		for (size_t i = 0; i < N; ++i) {
			auto nearest = FindNearestCentroid(points[i], centroids);

			if (membership[i] != nearest) {
				membership[i] = nearest;
				changes++;
			}

			updatedCentroids[nearest] += points[i];
			updatedCounts[nearest]++;
		}
	}

	template<size_t dim>
	void ClusteringCPU<dim>::UpdateCentroids(
		thrust::host_vector<Point<dim>>& centroids,
		thrust::host_vector<Point<dim>>& updatedCentroids,
		thrust::host_vector<size_t>& updatedCounts)
	{
		auto k = centroids.size();
		for (size_t i = 0; i < k; ++i) {
			for (size_t j = 0; j < dim; ++j) {
				centroids[i].Coordinates[j] = updatedCentroids[i].Coordinates[j] /
					std::max(updatedCounts[i], (size_t)1);
				updatedCentroids[i].Coordinates[j] = 0.0f;
			}

			updatedCounts[i] = 0;
		}
	}
}