#pragma once

#include "../Point.cuh"
#include "../CudaCheck.cuh"
#include "../Timers/TimerManager.cuh"

#include <cfloat>

#define FLOAT_INFINITY FLT_MAX

using namespace DataStructures;

namespace GPU1 
{
	// Kernel to convert data from Array of Structures (AoS) to Structure of Arrays (SoA)
	// Converts data layout: [x1, y1, z1, x2, y2, z2] -> [x1, x2, y1, y2, z1, z2]
	template<size_t dim>
	__global__
	void AoS2SoAKernel(float* deviceAoS, float* deviceSoA, size_t length)
	{
		size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

		if (tid >= length) return;

		for (size_t i = 0; i < dim; ++i) {
			deviceSoA[tid + i * length] = deviceAoS[tid * dim + i];
		}
	}

	// Kernel to convert data from Structure of Arrays (SoA) to Array of Structures (AoS).
	// Converts data layout: [x1, x2, y1, y2, z1, z2] -> [x1, y1, z1, x2, y2, z2]
	template<size_t dim>
	__global__
	void SoA2AoSKernel(float* deviceSoA, float* deviceAoS, size_t length)
	{
		size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

		if (tid >= length) return;

		for (size_t i = 0; i < dim; ++i) {
			deviceAoS[tid * dim + i] = deviceSoA[tid + i * length];
		}
	}

	// Helper function to access a specific coordinate of a point stored in SoA format
	__device__
	float& GetCoordinate(float* devicePoints, size_t size, size_t pointIndex, size_t dimensionIdx)
	{
		return devicePoints[pointIndex + dimensionIdx * size];
	}

	// Class to manage points stored on the GPU in a Structure of Arrays (SoA) format
	template<size_t dim>
	class DevicePointsCollection {
	private:
	public:
		size_t Size{};
		float* DevicePoints{}; // Pointer to the GPU memory for points in SoA format

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

		// Copies points from the device (SoA format) back to the host (AoS format)
		thrust::host_vector<Point<dim>> ToHost() {
			auto& timerManager = Timers::TimerManager::GetInstance();
			
			float* deviceAoS{}; // Temporary memory for AoS format on the device

			CUDACHECK(cudaMalloc(&deviceAoS, sizeof(Point<dim>) * Size));

			unsigned blockCount = static_cast<unsigned>((Size + THREADS_IN_ONE_BLOCK - 1) / THREADS_IN_ONE_BLOCK);

			timerManager.SoA2AoSKernelTimer.Start();

			// Launch the SoA to AoS conversion kernel
			SoA2AoSKernel<dim> << < blockCount, THREADS_IN_ONE_BLOCK >> > (DevicePoints, deviceAoS, Size);
			CUDACHECK(cudaPeekAtLastError());
			CUDACHECK(cudaDeviceSynchronize());

			timerManager.SoA2AoSKernelTimer.Stop();

			thrust::host_vector<Point<dim>> points(Size);

			timerManager.Device2HostDataTransfer.Start();

			CUDACHECK(cudaMemcpy(points.data(), deviceAoS, sizeof(Point<dim>) * Size, cudaMemcpyDeviceToHost));

			timerManager.Device2HostDataTransfer.Stop();

			CUDACHECK(cudaFree(deviceAoS));

			return points;
		}

	private:
		// Transfers points from the host (AoS format) to the device (SoA format)
		void FromHost(thrust::host_vector<Point<dim>>& points) {
			auto& timerManager = Timers::TimerManager::GetInstance();

			Size = points.size();

			float* deviceAoS{}; // Temporary memory for AoS format on the device

			CUDACHECK(cudaMalloc(&DevicePoints, sizeof(Point<dim>) * Size));
			CUDACHECK(cudaMalloc(&deviceAoS, sizeof(Point<dim>) * Size));

			timerManager.Host2DeviceDataTransfer.Start();

			// Copy points from the host to the AoS memory on the device
			CUDACHECK(cudaMemcpy(deviceAoS, points.data(), sizeof(Point<dim>) * Size, cudaMemcpyHostToDevice));

			timerManager.Host2DeviceDataTransfer.Stop();

			unsigned blockCount = static_cast<unsigned>((Size + THREADS_IN_ONE_BLOCK - 1) / THREADS_IN_ONE_BLOCK);

			timerManager.AoS2SoAKernelTimer.Start();

			// Launch the AoS to SoA conversion kernel
			AoS2SoAKernel<dim> << <blockCount, THREADS_IN_ONE_BLOCK >> > (deviceAoS, DevicePoints, Size);
			CUDACHECK(cudaPeekAtLastError());
			CUDACHECK(cudaDeviceSynchronize());

			timerManager.AoS2SoAKernelTimer.Stop();

			CUDACHECK(cudaFree(deviceAoS));
		}
	};
}