#pragma once
__device__ __constant__ RaytraceConstantBuffer cb[1];


texture<float4, 2, cudaReadModeElementType> tex;
