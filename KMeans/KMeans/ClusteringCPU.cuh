#pragma once

#include "HostTimer.cuh"
#include "Point.cuh"

using namespace Timers;
using namespace DataStructures;

namespace CPU
{
	template<size_t dim>
	class ClusteringCPU {
		HostTimer ComputeNewCentroidsTimer{};
		HostTimer UpdateCentroidsTimer{};

	public:

		thrust::host_vector<size_t> PerformClustering(
			thrust::host_vector<Point<dim>>& centroids,
			thrust::host_vector<Point<dim>>& points)
		{
			thrust::host_vector<Point<dim>> updatedCentroids(centroids.size());
			thrust::host_vector<size_t> updatedCounts(centroids.size());
			thrust::host_vector<size_t> membership(points.size());

			size_t changes = points.size();
			size_t iteration = 0, maxIterations = 100;

			std::cout << "Starting clustering..." << std::endl;
			std::cout << "Number of points: " << points.size() << ", Number of centroids: " << centroids.size() << std::endl;

			while (iteration++ < maxIterations && changes != 0) {
				std::cout << "\n=== Iteration: " << iteration << " ===" << std::endl;

				changes = 0;

				std::cout << "-> Computing new centroids..." << std::endl;

				ComputeNewCentroidsTimer.Start();
				ComputeNewCentroids(points, centroids, updatedCentroids, updatedCounts, membership, changes);
				ComputeNewCentroidsTimer.Stop();

				std::cout << "   Computation time: " << ComputeNewCentroidsTimer.ElapsedMiliseconds() << " ms" << std::endl;

				std::cout << "-> Updating centroids..." << std::endl;

				UpdateCentroidsTimer.Start();
				UpdateCentroids(centroids, updatedCentroids, updatedCounts);
				UpdateCentroidsTimer.Stop();

				std::cout << "   Update time: " << UpdateCentroidsTimer.ElapsedMiliseconds() << " ms" << std::endl;

				std::cout << "-> Changes in membership: " << changes << std::endl;
			}

			if (changes == 0) {
				std::cout << "\nClustering completed: No changes in membership." << std::endl;
			}
			else {
				std::cout << "\nClustering completed: Maximum number of iterations reached." << std::endl;
			}

			return membership;
		}


	private:
		size_t FindNearestCentroid(
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

		void ComputeNewCentroids(
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

		void UpdateCentroids(
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
	};
}