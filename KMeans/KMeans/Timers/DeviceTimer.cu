#include "DeviceTimer.cuh"

namespace Timers
{
	void DeviceTimer::Start()
	{
		// Ensure old events are destroyed before creating new ones
		DestroyCudaEvents();
		InitCudaEvents();
		CUDACHECK(cudaEventRecord(StartEvent));
	}

	void DeviceTimer::Stop() {
		if (StartEvent == nullptr || StopEvent == nullptr)
			return;

		CUDACHECK(cudaEventRecord(StopEvent));
		CUDACHECK(cudaEventSynchronize(StopEvent)); // Wait until the stop event is complete
		CUDACHECK(cudaEventElapsedTime(&MilisecondsElapsed, StartEvent, StopEvent));

		TotalMilisecondsElapsed += MilisecondsElapsed;
	}

	float DeviceTimer::ElapsedMiliseconds() {
		return MilisecondsElapsed;
	}

	float DeviceTimer::TotalElapsedMiliseconds() {
		return TotalMilisecondsElapsed;
	}

	void DeviceTimer::Reset() {
		DestroyCudaEvents();
		MilisecondsElapsed = 0;
		TotalMilisecondsElapsed = 0;
	}

	void DeviceTimer::InitCudaEvents() {
		if (StartEvent == nullptr) {
			CUDACHECK(cudaEventCreate(&StartEvent));
		}
		if (StopEvent == nullptr) {
			CUDACHECK(cudaEventCreate(&StopEvent));
		}
	}

	void DeviceTimer::DestroyCudaEvents() {
		if (StartEvent != nullptr) {
			CUDACHECK(cudaEventDestroy(StartEvent));
			StartEvent = nullptr;
		}
		if (StopEvent != nullptr) {
			CUDACHECK(cudaEventDestroy(StopEvent));
			StopEvent = nullptr;
		}
	}

	DeviceTimer::~DeviceTimer() {
		Reset();
	}
}