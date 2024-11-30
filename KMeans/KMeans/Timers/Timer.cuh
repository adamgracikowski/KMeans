#pragma once

namespace Timers
{
	class Timer {
	public:
		virtual void Start() = 0;
		virtual void Stop() = 0;
		virtual float ElapsedMiliseconds() = 0;
		virtual float TotalElapsedMiliseconds() = 0;
		virtual void Reset() = 0;
		virtual ~Timer() = default;
	};
}