#include "ParseProgramParameters.cuh"

ProgramParameters ParseProgramParameters(int argc, char* argv[]) {
	ProgramParameters parameters{};

	if (argc != 5) {
		std::cerr << "Usage: KMeans data_format computation_method input_file output_file" << std::endl;
		return parameters;
	}

	parameters.DataFormat = argv[1];
	parameters.ComputationMethod = argv[2];
	parameters.InputFile = argv[3];
	parameters.OutputFile = argv[4];

	if (parameters.DataFormat != "txt" && parameters.DataFormat != "bin") {
		std::cerr << "Invalid data format. Use 'txt' or 'bin'." << std::endl;
		return parameters;
	}

	if (parameters.ComputationMethod != "cpu" && parameters.ComputationMethod != "gpu1" && parameters.ComputationMethod != "gpu2") {
		std::cerr << "Invalid computation method. Use 'cpu', 'gpu1' or 'gpu2'." << std::endl;
	}

	parameters.Success = true;
	return parameters;
}