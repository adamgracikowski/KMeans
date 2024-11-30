#pragma once

#include "HostTimer.cuh"
#include "DeviceTimer.cuh"

namespace Timers
{
	class TimerManager {
	private:
		TimerManager() = default;
		TimerManager(const TimerManager&) = delete;
		TimerManager& operator=(const TimerManager&) = delete;

		~TimerManager() {
			SoA2AoSKernelTimer.Reset();
			Host2DeviceDataTransfer.Reset();
			Device2HostDataTransfer.Reset();
			AoS2SoAKernelTimer.Reset();
			ComputeNewCentroidsKernelTimer.Reset();
			UpdateCentroidsKernelTimer.Reset();
		}

	public:
		HostTimer LoadDataFromInputFileTimer{};
		HostTimer SaveDataToOutputFileTimer{};
		HostTimer PerformClusteringTimer{};
		HostTimer ComputeNewCentroidsTimer{};
		HostTimer UpdateCentroidsTimer{};

		DeviceTimer SoA2AoSKernelTimer{};
		DeviceTimer AoS2SoAKernelTimer{};
		DeviceTimer Host2DeviceDataTransfer{};
		DeviceTimer Device2HostDataTransfer{};
		DeviceTimer ComputeNewCentroidsKernelTimer{};
		DeviceTimer UpdateCentroidsKernelTimer{};

		static TimerManager& GetInstance() {
			static TimerManager instance;
			return instance;
		}
	};
}