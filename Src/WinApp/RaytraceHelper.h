#pragma once

#define R_CH 0
#define G_CH 1
#define B_CH 2
#define A_CH 3

#include <iostream> 
#include <vector> 
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_math.h>
#include "RaytraceConstantBuffer.h"
#include "KernelHelper.h"
#include "RaytraceHelper.h"

__device__ unsigned int float4ToInt(float4* rgba)
{ 
    return ((unsigned int)(rgba->w*255.0f)<<24) | 
			((unsigned int)(rgba->z*255.0f)<<16) | 
			((unsigned int)(rgba->y*255.0f)<<8) | 
			(unsigned int)(rgba->x*255.0f);
}

__device__ float4* mat4mul(float* mat4,const float4* in_vec, float4* out_res)
{
	out_res->x = dot( *in_vec,  make_float4(mat4[0],mat4[1],mat4[2],   mat4[3]));
	out_res->y = dot( *in_vec,  make_float4(mat4[4],mat4[5],mat4[6],   mat4[7]));
	out_res->z = dot( *in_vec,  make_float4(mat4[8],mat4[9],mat4[10],  mat4[11]));
	out_res->w = dot( *in_vec,  make_float4(mat4[12],mat4[13],mat4[14],mat4[15]));
	return out_res;
}

// optimization for when w is not needed
__device__ float4* mat4mul_ignoreW(float* mat4,const float4* in_vec, float4* out_res)
{
	out_res->x = dot( *in_vec,  make_float4(mat4[0],mat4[1],mat4[2],   mat4[3]));
	out_res->y = dot( *in_vec,  make_float4(mat4[4],mat4[5],mat4[6],   mat4[7]));
	out_res->z = dot( *in_vec,  make_float4(mat4[8],mat4[9],mat4[10],  mat4[11]));
	out_res->w = 0.0f;
	return out_res;
}

// squared length of vector
__device__ float squaredLen(const float4* in_vec)
{
	return in_vec->x*in_vec->x + in_vec->y*in_vec->y + in_vec->z*in_vec->z;
}
