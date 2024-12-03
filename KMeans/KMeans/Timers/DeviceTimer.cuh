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
		~DeviceTimer() {
			Reset();
		}

		void Start() override {
			// Ensure old events are destroyed before creating new ones
			DestroyCudaEvents();
			InitCudaEvents();
			CUDACHECK(cudaEventRecord(StartEvent));
		}

		void Stop() override {
			if (StartEvent == nullptr || StopEvent == nullptr)
				return;

			CUDACHECK(cudaEventRecord(StopEvent));
			CUDACHECK(cudaEventSynchronize(StopEvent)); // Wait until the stop event is complete
			CUDACHECK(cudaEventElapsedTime(&MilisecondsElapsed, StartEvent, StopEvent));

			TotalMilisecondsElapsed += MilisecondsElapsed;
		}

		float ElapsedMiliseconds() override {
			return MilisecondsElapsed;
		}

		float TotalElapsedMiliseconds() override {
			return TotalMilisecondsElapsed;
		}

		void Reset() override {
			DestroyCudaEvents();
			MilisecondsElapsed = 0;
			TotalMilisecondsElapsed = 0;
		}

	private:
		void InitCudaEvents() {
			if (StartEvent == nullptr) {
				CUDACHECK(cudaEventCreate(&StartEvent));
			}
			if (StopEvent == nullptr) {
				CUDACHECK(cudaEventCreate(&StopEvent));
			}
		}

		void DestroyCudaEvents() {
			if (StartEvent != nullptr) {
				CUDACHECK(cudaEventDestroy(StartEvent));
				StartEvent = nullptr;
			}
			if (StopEvent != nullptr) {
				CUDACHECK(cudaEventDestroy(StopEvent));
				StopEvent = nullptr;
			}
		}
	};
}