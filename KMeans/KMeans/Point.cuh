#pragma once

#include "cuda_runtime.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define HOST_DEVICE __host__ __device__
#define THREADS_IN_ONE_BLOCK 1024

namespace DataStructures 
{
	template<size_t dim>
	struct Point {
	public:
		float Coordinates[dim];

		HOST_DEVICE
		Point() {
			for (int i = 0; i < dim; i++) {
				Coordinates[i] = 0.0;
			}
		}

		HOST_DEVICE
		static float SquareDistance(const Point<dim>& p, const Point<dim>& q) {
			float distance = 0.0;
			for (int i = 0; i < dim; i++) {
				float difference = p.Coordinates[i] - q.Coordinates[i];
				distance += difference * difference;
			}

			return distance;
		}

		HOST_DEVICE
		Point<dim>& operator+=(const Point<dim>& other) {
			for (int i = 0; i < dim; i++) {
				this->Coordinates[i] += other.Coordinates[i];
			}

			return *this;
		}
	};
}