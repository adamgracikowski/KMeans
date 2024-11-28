
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "ParseProgramParameters.cuh"
#include "CreateManagerInstance.cuh"

#include <iostream>
#include <cstdio>
#include <stdexcept>

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
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what();
		return 1;
	}
}