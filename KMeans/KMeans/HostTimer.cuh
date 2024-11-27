#pragma once

#include "Timer.cuh"
#include <chrono>

namespace Timers 
{
	class HostTimer : public Timer {
	private:
		std::chrono::steady_clock::time_point StartTimePoint;
		std::chrono::microseconds TotalElapsedMicroseconds;

	public:
		HostTimer() : StartTimePoint(std::chrono::steady_clock::time_point::min()),
			TotalElapsedMicroseconds(std::chrono::microseconds::zero()) {
		}

		void Start() override {
			StartTimePoint = std::chrono::steady_clock::now();
		}

		void Stop() override {
			auto stopTimePoint = std::chrono::steady_clock::now();
			TotalElapsedMicroseconds += std::chrono::duration_cast<std::chrono::microseconds>(
				stopTimePoint - StartTimePoint
			);
		}

		float ElapsedMiliseconds() override {
			return TotalElapsedMicroseconds.count() / 1000.0f;
		}

		void Reset() override {
			StartTimePoint = std::chrono::steady_clock::time_point::min();
			TotalElapsedMicroseconds = std::chrono::microseconds::zero();
		}

		~HostTimer() override = default;
	};
}