#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "ProgramManagers/ParseProgramParameters.cuh"
#include "ProgramManagers/CreateManagerInstance.cuh"
#include "Visualization/Visualization.cuh"

#include <cstdio>
#include <stdexcept>

#include <vector>
#include <iostream>
#include <cstdlib>

int main(int argc, char* argv[])
{
	auto parameters = ParseProgramParameters(argc, argv);

	if (!parameters.Success) {
		return 1;
	}

	std::unique_ptr<IProgramManager> manager{};

	try
	{
		manager = CreateManagerInstance(parameters);

		auto membership = manager->StartComputation();

		manager->SaveDataToOutputFile(membership);
		manager->DisplaySummary();

		// Start visualization
		if (manager->GetDimension() == 3) {
			auto glmPoints = manager->GetGlmPoints();

			if (Visualize(glmPoints, membership)) {
				std::cerr << "Visualization has failed." << std::endl;
			}
		}
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return 1;
	}
}