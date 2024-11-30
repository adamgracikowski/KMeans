#pragma once

#include <string>

struct ProgramParameters {
	std::string DataFormat{};
	std::string ComputationMethod{};
	std::string InputFile{};
	std::string OutputFile{};
	bool Success{};
};