#pragma once

// =======================================================================================
//                                    Light Structures
// =======================================================================================

#include <vector> 
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "RaytraceConstantBuffer.h"
#include "KernelHelper.h"
#include "RaytraceSurfaceMaterial.h"

struct SurfaceLightingData
{
	float4 diffuseColor;
	float4 specularColor;
};

struct Light
{
	float diffusePower;
	float specularPower;
	float4 vec; // direction(w=0) or position(w=1)
	float4 diffuseColor;
	float4 specularColor;
};