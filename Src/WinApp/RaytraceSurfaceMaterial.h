#pragma once

#include <vector> 
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "RaytraceConstantBuffer.h"
#include "KernelHelper.h"
#include "RaytraceSurfaceMaterial.h"

struct Material
{
	float4 diffuse;
	float4 specular; // r,g,b,glossiness
	float reflection;
};