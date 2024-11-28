#pragma once

#include "CommonDeviceTools.cuh"

using namespace DataStructures;

namespace CommonGPU 
{
	template<size_t dim>
	class DevicePointsCollection {
	private:
	public:
		size_t Size{};
		float* DevicePoints{};

	public:
		DevicePointsCollection(size_t size)
		{
			Size = size;
			cudaMalloc(&DevicePoints, sizeof(Point<dim>) * Size);
			cudaMemset(DevicePoints, 0, sizeof(Point<dim>) * Size);
		}

		DevicePointsCollection(thrust::host_vector<Point<dim>>& points)
		{
			FromHost(points);
		}

		~DevicePointsCollection() {
			cudaFree(DevicePoints);
		}

		void operator=(thrust::host_vector<Point<dim>>& points) {
			FromHost(points);
		}

		size_t GetSize() {
			return Size;
		}

		float* RawAccess() {
			return DevicePoints;
		}

		thrust::host_vector<Point<dim>> ToHost() {
			float* deviceAoS{};

			cudaMalloc(&deviceAoS, sizeof(Point<dim>) * Size);

			unsigned blockCount = static_cast<unsigned>((Size + THREADS_IN_ONE_BLOCK - 1) / THREADS_IN_ONE_BLOCK);

			// pomiar czasu na konwersjê pomiêdzy SoA -> AoS
			SoA2AoSKernel<dim> << < blockCount, THREADS_IN_ONE_BLOCK >> > (DevicePoints, deviceAoS, Size);
			// stiop pomiar czasu na konwersjê pomiêdzy SoA -> AoS

			thrust::host_vector<Point<dim>> points(Size);

			// pomiar czasu na transfer danych z gpu na cpu
			cudaMemcpy(points.data(), deviceAoS, sizeof(Point<dim>) * Size, cudaMemcpyDeviceToHost);
			// stop pomiaru czasu na transfer danych z gpu na cpu

			cudaFree(deviceAoS);

			return points;
		}

	private:
		void FromHost(thrust::host_vector<Point<dim>>& points) {
			Size = points.size();

			float* deviceAoS{};

			cudaMalloc(&DevicePoints, sizeof(Point<dim>) * Size);
			cudaMalloc(&deviceAoS, sizeof(Point<dim>) * Size);

			// pomiar czasu na transfer danych z cpu na gpu
			cudaMemcpy(deviceAoS, points.data(), sizeof(Point<dim>) * Size, cudaMemcpyHostToDevice);
			// stop pomiaru czasu na transfer danych z cpu na gpu

			unsigned blockCount = static_cast<unsigned>((Size + THREADS_IN_ONE_BLOCK - 1) / THREADS_IN_ONE_BLOCK);

			// pomiar czasu na konwersjê pomiêdzy AoS -> SoA
			AoS2SoAKernel<dim> << <blockCount, THREADS_IN_ONE_BLOCK >> > (deviceAoS, DevicePoints, Size);
			// stop pomiaru czasu na konwersjê pomiêdzy AoS -> SoA

			cudaFree(deviceAoS);
		}
	};

	__device__
	float& GetCoordinate(float* devicePoints, size_t size, size_t pointIndex, size_t dimensionIdx) {
		return devicePoints[pointIndex + dimensionIdx * size];
	}
}