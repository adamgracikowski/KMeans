#pragma once

#include "ProgramManager.cuh"

#include <memory>

// Macro to generate a case statement for handling a specific dimension M
// It creates an instance of ProgramManager<M> with parameters, loads data, and then returns it
#define GENERATE_CASE(M)														  \
    case M: {																	  \
        auto instance = std::make_unique<ProgramManager<M>>(N, d, k, parameters); \
		instance->LoadDataFromInputFile(inputFile);								  \
        return instance;														  \
    }

// Factory function to create an instance of IProgramManager based on input parameters
std::unique_ptr<IProgramManager> CreateManagerInstance(ProgramParameters parameters) {
	FILE* inputFile{};
	int N, d, k;

	// Handle input file in text format
	if (parameters.DataFormat == "txt") {
		if (fopen_s(&inputFile, parameters.InputFile.c_str(), "r") != 0) {
			throw std::runtime_error("Could not open " + parameters.InputFile);
		}

		if (fscanf_s(inputFile, "%d %d %d", &N, &d, &k) != 3) {
			fclose(inputFile);
			throw std::runtime_error("Error while reading the header of " + parameters.InputFile);
		}
	}
	// Handle input file in binary format
	else if (parameters.DataFormat == "bin") {
		if (fopen_s(&inputFile, parameters.InputFile.c_str(), "rb") != 0) {
			throw std::runtime_error("Could not open " + parameters.InputFile);
		}

		if (fread(&N, sizeof(int), 1, inputFile) != 1 ||
			fread(&d, sizeof(int), 1, inputFile) != 1 ||
			fread(&k, sizeof(int), 1, inputFile) != 1) {
			fclose(inputFile);
			throw std::runtime_error("Error while reading the header of " + parameters.InputFile);
		}
	}
	// Throw an error for unsupported data formats
	else {
		throw std::runtime_error("Invalid format " + parameters.DataFormat);
	}

	switch (d) {
		GENERATE_CASE(1);
		GENERATE_CASE(2);
		GENERATE_CASE(3);
		GENERATE_CASE(4);
		GENERATE_CASE(5);
		GENERATE_CASE(6);
		GENERATE_CASE(7);
		GENERATE_CASE(8);
		GENERATE_CASE(9);
		GENERATE_CASE(10);
		GENERATE_CASE(11);
		GENERATE_CASE(12);
		GENERATE_CASE(13);
		GENERATE_CASE(14);
		GENERATE_CASE(15);
		GENERATE_CASE(16);
		GENERATE_CASE(17);
		GENERATE_CASE(18);
		GENERATE_CASE(19);
		GENERATE_CASE(20);
	default:
		fclose(inputFile);
		throw std::runtime_error("Unhandled dimension");
	}
}