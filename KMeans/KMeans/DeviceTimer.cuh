#pragma once

#include "Timer.cuh"
#include <driver_types.h>
#include <cuda_runtime_api.h>

namespace Timers
{
	class DeviceTimer : public Timer {
	private:
		cudaEvent_t StartEvent{ nullptr };
		cudaEvent_t StopEvent{ nullptr };

		size_t StartCounter{};
		float ElapsedMiliseconds{};
		float TotalElapsedMiliseconds{};
		bool IsTotalElapsedMilisecondsUpdated{};

	public:

		void Start() override {
			if (StartCounter > 0) {
				UpdateTotalElapsedMiliseconds();
				DestroyEvents();
			}

			InitCudaEvents();
			StartCounter++;

			cudaEventRecord(StartEvent);
		}
	 
		void Stop() override {
			cudaEventRecord(StopEvent);
			IsTotalElapsedMilisecondsUpdated = false;
		}

		float ElapsedMiliseconds() override {
			if (!IsTotalElapsedMilisecondsUpdated) {
				UpdateTotalElapsedMiliseconds();
				IsTotalElapsedMilisecondsUpdated = true;
			}

			return TotalElapsedMiliseconds;
		}

		void Reset() override {
			TotalElapsedMiliseconds = 0;

			if (StartCounter == 0) return;

			DestroyEvents();
			StartCounter = 0;
			IsTotalElapsedMilisecondsUpdated = false;
		}

		~DeviceTimer() override {
			if (StartCounter == 0) return;

			StartCounter = 0;
			DestroyEvents();
		}

	private:
		void InitCudaEvents() {
			cudaEventCreate(&StartEvent);
			cudaEventCreate(&StopEvent);
		}

		void DestroyEvents() {
			if (StartEvent) {
				cudaEventDestroy(StartEvent);
				StartEvent = nullptr;
			}
			if (StopEvent) {
				cudaEventDestroy(StopEvent);
				StopEvent = nullptr;
			}
		}

		void UpdateTotalElapsedMiliseconds() {
			cudaEventElapsedTime(&ElapsedMiliseconds, StartEvent, StopEvent);
			TotalElapsedMiliseconds += ElapsedMiliseconds;
		}

	};
}