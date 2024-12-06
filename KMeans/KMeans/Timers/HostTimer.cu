#include "HostTimer.cuh"

namespace Timers
{
	HostTimer::HostTimer() : StartTimePoint(std::chrono::steady_clock::time_point::min()),
		TotalElapsedMicroseconds(std::chrono::microseconds::zero()),
		ElapsedMicroseconds(std::chrono::microseconds::zero()) {
	}

	void HostTimer::Start() {
		StartTimePoint = std::chrono::steady_clock::now();
	}

	void HostTimer::Stop() {
		auto stopTimePoint = std::chrono::steady_clock::now();

		ElapsedMicroseconds = std::chrono::duration_cast<std::chrono::microseconds>(
			stopTimePoint - StartTimePoint
		);

		TotalElapsedMicroseconds += ElapsedMicroseconds;
	}

	float HostTimer::ElapsedMiliseconds() {
		return ElapsedMicroseconds.count() / 1000.0f;
	}

	float HostTimer::TotalElapsedMiliseconds() {
		return TotalElapsedMicroseconds.count() / 1000.0f;
	}

	void HostTimer::Reset() {
		StartTimePoint = std::chrono::steady_clock::time_point::min();
		ElapsedMicroseconds = std::chrono::microseconds::zero();
		TotalElapsedMicroseconds = std::chrono::microseconds::zero();
	}

	HostTimer::~HostTimer() = default;
}