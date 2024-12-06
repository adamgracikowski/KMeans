#pragma once

#include "Timer.cuh"
#include "../CudaCheck.cuh"

#include <driver_types.h>
#include <cuda_runtime_api.h>

namespace Timers
{
	class DeviceTimer : public Timer {
	private:
		cudaEvent_t StartEvent{ nullptr }; // CUDA event used to mark the start of a timing interval
		cudaEvent_t StopEvent{ nullptr };  // CUDA event used to mark the end of a timing interval
		float MilisecondsElapsed{};
		float TotalMilisecondsElapsed{};

	public:
		~DeviceTimer() override;

		void Start() override;
		void Stop() override;
		float ElapsedMiliseconds() override;
		float TotalElapsedMiliseconds() override;
		void Reset() override;
	private:
		void InitCudaEvents();
		void DestroyCudaEvents();
	};
}