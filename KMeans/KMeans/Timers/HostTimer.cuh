#pragma once

#include "Timer.cuh"

#include <chrono>

namespace Timers 
{
	class HostTimer : public Timer {
	private:
		std::chrono::steady_clock::time_point StartTimePoint;
		std::chrono::microseconds TotalElapsedMicroseconds;
		std::chrono::microseconds ElapsedMicroseconds;

	public:
		HostTimer();

		void Start() override;
		void Stop() override;
		float ElapsedMiliseconds() override;
		float TotalElapsedMiliseconds() override;
		void Reset() override;
		~HostTimer() override;
	};
}