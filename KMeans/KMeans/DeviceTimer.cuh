#pragma once

#include "Timer.cuh"
#include "CudaCheck.cuh"

#include <driver_types.h>
#include <cuda_runtime_api.h>

namespace Timers
{
	class DeviceTimer : public Timer {
	private:
		cudaEvent_t StartEvent{ nullptr };
		cudaEvent_t StopEvent{ nullptr };

		size_t StartCounter{};
		float MilisecondsElapsed{};
		float TotalMilisecondsElapsed{};
		bool IsTotalMilisecondsElapsedUpdated{};

	public:

		void Start() override {
			if (StartCounter > 0) {
				UpdateTotalMilisecondsElapsed();
				DestroyEvents();
			}

			InitCudaEvents();
			StartCounter++;

			CUDACHECK(cudaEventRecord(StartEvent));
		}
	 
		void Stop() override {
			CUDACHECK(cudaEventRecord(StopEvent));
			IsTotalMilisecondsElapsedUpdated = false;
		}

		float ElapsedMiliseconds() override {
			if (!IsTotalMilisecondsElapsedUpdated) {
				UpdateTotalMilisecondsElapsed();
				IsTotalMilisecondsElapsedUpdated = true;
			}

			return TotalMilisecondsElapsed;
		}

		void Reset() override {
			TotalMilisecondsElapsed = 0;

			if (StartCounter == 0) return;

			DestroyEvents();
			StartCounter = 0;
			IsTotalMilisecondsElapsedUpdated = false;
		}

		~DeviceTimer() override {
			if (StartCounter == 0) return;

			StartCounter = 0;
			DestroyEvents();
		}

	private:
		void InitCudaEvents() {
			CUDACHECK(cudaEventCreate(&StartEvent));
			CUDACHECK(cudaEventCreate(&StopEvent));
		}

		void DestroyEvents() {
			if (StartEvent != nullptr) {
				CUDACHECK(cudaEventDestroy(StartEvent));
				StartEvent = nullptr;
			}
			if (StopEvent != nullptr) {
				CUDACHECK(cudaEventDestroy(StopEvent));
				StopEvent = nullptr;
			}
		}

		void UpdateTotalMilisecondsElapsed() {
			CUDACHECK(cudaEventElapsedTime(&MilisecondsElapsed, StartEvent, StopEvent));
			TotalMilisecondsElapsed += MilisecondsElapsed;
		}
	};
}