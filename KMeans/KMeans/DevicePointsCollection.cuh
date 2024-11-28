#pragma once

#include "CommonDeviceTools.cuh"
#include "CudaCheck.cuh"

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
			CUDACHECK(cudaMalloc(&DevicePoints, sizeof(Point<dim>) * Size));
			CUDACHECK(cudaMemset(DevicePoints, 0, sizeof(Point<dim>) * Size));
		}

		DevicePointsCollection(thrust::host_vector<Point<dim>>& points)
		{
			FromHost(points);
		}

		~DevicePointsCollection() {
			CUDACHECK(cudaFree(DevicePoints));
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

			CUDACHECK(cudaMalloc(&deviceAoS, sizeof(Point<dim>) * Size));

			unsigned blockCount = static_cast<unsigned>((Size + THREADS_IN_ONE_BLOCK - 1) / THREADS_IN_ONE_BLOCK);

			// pomiar czasu na konwersjê pomiêdzy SoA -> AoS
			SoA2AoSKernel<dim> << < blockCount, THREADS_IN_ONE_BLOCK >> > (DevicePoints, deviceAoS, Size);
			CUDACHECK(cudaPeekAtLastError());
			// stiop pomiar czasu na konwersjê pomiêdzy SoA -> AoS

			thrust::host_vector<Point<dim>> points(Size);

			// pomiar czasu na transfer danych z gpu na cpu
			CUDACHECK(cudaMemcpy(points.data(), deviceAoS, sizeof(Point<dim>) * Size, cudaMemcpyDeviceToHost));
			// stop pomiaru czasu na transfer danych z gpu na cpu

			CUDACHECK(cudaFree(deviceAoS));

			return points;
		}

	private:
		void FromHost(thrust::host_vector<Point<dim>>& points) {
			Size = points.size();

			float* deviceAoS{};

			CUDACHECK(cudaMalloc(&DevicePoints, sizeof(Point<dim>) * Size));
			CUDACHECK(cudaMalloc(&deviceAoS, sizeof(Point<dim>) * Size));

			// pomiar czasu na transfer danych z cpu na gpu
			CUDACHECK(cudaMemcpy(deviceAoS, points.data(), sizeof(Point<dim>) * Size, cudaMemcpyHostToDevice));
			// stop pomiaru czasu na transfer danych z cpu na gpu

			unsigned blockCount = static_cast<unsigned>((Size + THREADS_IN_ONE_BLOCK - 1) / THREADS_IN_ONE_BLOCK);

			// pomiar czasu na konwersjê pomiêdzy AoS -> SoA
			AoS2SoAKernel<dim> << <blockCount, THREADS_IN_ONE_BLOCK >> > (deviceAoS, DevicePoints, Size);
			CUDACHECK(cudaPeekAtLastError());
			CUDACHECK(cudaDeviceSynchronize());
			// stop pomiaru czasu na konwersjê pomiêdzy AoS -> SoA

			CUDACHECK(cudaFree(deviceAoS));
		}
	};

	__device__
	float& GetCoordinate(float* devicePoints, size_t size, size_t pointIndex, size_t dimensionIdx) {
		return devicePoints[pointIndex + dimensionIdx * size];
	}
}