#pragma once

#define __CUDACC__

#pragma comment(lib, "cudart") 
#include <vector> 
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector_types.h>

__device__ float4 sampleTextureRGB(texture<uchar4, 2, cudaReadModeElementType> p_texture, float2 p_uv)
{
	//unsigned char blue = tex2D(p_texture, (3 * p_uv.x) , p_uv.y);
	//unsigned char green = tex2D(p_texture, (3 * p_uv.x) + 1, p_uv.y);
	//unsigned char red = tex2D(p_texture, (3 * p_uv.x) + 2, p_uv.y);
	uchar4 cols=tex2D(p_texture,p_uv.x,p_uv.y);

	float4 texCol=make_float4((float)cols.x*UCHAR_COLOR_TO_FLOAT,
		(float)cols.y*UCHAR_COLOR_TO_FLOAT,
		(float)cols.z*UCHAR_COLOR_TO_FLOAT,0.0f);
	return texCol;
}