#pragma once

#include "CommonDeviceTools.cuh"
#include "CudaCheck.cuh"
#include "TimerManager.cuh"

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
			auto& timerManager = Timers::TimerManager::GetInstance();
			
			float* deviceAoS{};

			CUDACHECK(cudaMalloc(&deviceAoS, sizeof(Point<dim>) * Size));

			unsigned blockCount = static_cast<unsigned>((Size + THREADS_IN_ONE_BLOCK - 1) / THREADS_IN_ONE_BLOCK);

			timerManager.SoA2AoSKernelTimer.Start();

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
		void FromHost(thrust::host_vector<Point<dim>>& points) {
			auto& timerManager = Timers::TimerManager::GetInstance();

			Size = points.size();

			float* deviceAoS{};

			CUDACHECK(cudaMalloc(&DevicePoints, sizeof(Point<dim>) * Size));
			CUDACHECK(cudaMalloc(&deviceAoS, sizeof(Point<dim>) * Size));

			timerManager.Host2DeviceDataTransfer.Start();

			CUDACHECK(cudaMemcpy(deviceAoS, points.data(), sizeof(Point<dim>) * Size, cudaMemcpyHostToDevice));

			timerManager.Host2DeviceDataTransfer.Stop();

			unsigned blockCount = static_cast<unsigned>((Size + THREADS_IN_ONE_BLOCK - 1) / THREADS_IN_ONE_BLOCK);

			timerManager.AoS2SoAKernelTimer.Start();

			AoS2SoAKernel<dim> << <blockCount, THREADS_IN_ONE_BLOCK >> > (deviceAoS, DevicePoints, Size);
			CUDACHECK(cudaPeekAtLastError());
			CUDACHECK(cudaDeviceSynchronize());

			timerManager.AoS2SoAKernelTimer.Stop();

			CUDACHECK(cudaFree(deviceAoS));
		}
	};

	__device__
	float& GetCoordinate(float* devicePoints, size_t size, size_t pointIndex, size_t dimensionIdx) {
		return devicePoints[pointIndex + dimensionIdx * size];
	}
}