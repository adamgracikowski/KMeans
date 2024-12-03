#pragma once

#include "cuda_runtime.h"
#include <cstdio>
#include <cstdlib>

#define CUDACHECK(err) do { cuda_check((err), __FILE__, __LINE__); } while(false)

// https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
inline void cuda_check(cudaError_t error_code, const char* file, int line)
{
	if (error_code != cudaSuccess)
	{
		fprintf(stderr, "CUDA Error %d: %s. In file '%s' on line %d\n", error_code, cudaGetErrorString(error_code), file, line);
		fflush(stderr);
		exit(error_code);
	}
}