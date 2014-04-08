#pragma once
#include "vector_functions.h"

#include <vector> 
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ __constant__ float3 colarr[18]={{1.0f,0.0f,0.0f},
	{0.0f,1.0f,0.0f},
	{0.0f,0.0f,1.0f},
	{1.0f,0.0f,1.0f},
	{0.0f,1.0f,1.0f},
	{1.0f,1.0f,0.0f},
	{1.0f,0.5f,0.0f},
	{0.5f,1.0f,0.0f},
	{1.0f,0.0f,0.5f},
	{0.5f,0.5f,0.5f},
	{1.0f,0.7f,0.7f},
	{1.0f,1.0f,0.2f},
	{0.33f,0.25f,0.4f},
	{0.8f,1.0f,0.24f},
	{0.0f,0.66f,0.5f},
	{0.1f,0.231f,0.13f},
	{0.87f,0.5f,1.0f},
	{0.2f,0.2f,1.0f}};