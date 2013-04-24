#pragma once
#include "RaytraceSurfaceMaterial.h"

struct Intersection
{
	float4 normal;
	float4 pos;
	Material surface; // don't allow changing the material's values from intersection struct
	float dist;
};

struct Ray
{
	float4 dir;
	float4 origin;
};