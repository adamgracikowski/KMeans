#pragma once

#include "HostTimer.cuh"

namespace Timers
{
	class HostTimerManager {
	private:
		HostTimerManager() = default;
		~HostTimerManager() = default;

		HostTimerManager(const HostTimerManager&) = delete;
		HostTimerManager& operator=(const HostTimerManager&) = delete;

	public:
		HostTimer LoadDataFromInputFileTimer{};
		HostTimer SaveDataToOutputFileTimer{};
		HostTimer PerformClusteringTimer{};
		HostTimer ComputeNewCentroidsTimer{};
		HostTimer UpdateCentroidsTimer{};

		static HostTimerManager& GetInstance() {
			static HostTimerManager instance;
			return instance;
		}
	};
}