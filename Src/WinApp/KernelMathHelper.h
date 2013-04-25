#pragma once

#define R_CH 0
#define G_CH 1
#define B_CH 2
#define A_CH 3
#define __CUDACC__

#include <iostream> 
#include <vector> 
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#include "RaytraceConstantBuffer.h"

#pragma comment(lib, "cudart") 

__device__ float fminf(float a, float b)
{
	return a < b ? a : b;
}
__device__ float fmaxf(float a, float b)
{
	return a > b ? a : b;
}
__device__ float rsqrtf(float x)
{
	return 1.0f/__fsqrt_rd(x);
	//return 1.0f / sqrtf(x);
}

__device__ float cu_dot(float2 a, float2 b)
{
	return a.x * b.x + a.y * b.y;
}
__device__ float cu_dot(float3 a, float3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ float cu_dot(float4 a, float4 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__device__ unsigned int float4ToInt(float4* rgba)
{ 
    return ((unsigned int)(rgba->w*255.0f)<<24) | 
			((unsigned int)(rgba->z*255.0f)<<16) | 
			((unsigned int)(rgba->y*255.0f)<<8) | 
			(unsigned int)(rgba->x*255.0f);
}

__device__ float4* mat4mul(float* mat4,const float4* in_vec, float4* out_res)
{
	out_res->x = cu_dot( *in_vec,  make_float4(mat4[0],mat4[1],mat4[2],   mat4[3]));
	out_res->y = cu_dot( *in_vec,  make_float4(mat4[4],mat4[5],mat4[6],   mat4[7]));
	out_res->z = cu_dot( *in_vec,  make_float4(mat4[8],mat4[9],mat4[10],  mat4[11]));
	out_res->w = cu_dot( *in_vec,  make_float4(mat4[12],mat4[13],mat4[14],mat4[15]));
	return out_res;
}

// optimization for when w is not needed
__device__ float4* mat4mul_ignoreW(float* mat4,const float4* in_vec, float4* out_res)
{
	out_res->x = cu_dot( *in_vec,  make_float4(mat4[0],mat4[1],mat4[2],   mat4[3]));
	out_res->y = cu_dot( *in_vec,  make_float4(mat4[4],mat4[5],mat4[6],   mat4[7]));
	out_res->z = cu_dot( *in_vec,  make_float4(mat4[8],mat4[9],mat4[10],  mat4[11]));
	out_res->w = 0.0f;
	return out_res;
}

// squared length of vector
__device__ float squaredLen(const float4* in_vec)
{
	return in_vec->x*in_vec->x + in_vec->y*in_vec->y + in_vec->z*in_vec->z;
}

////////////////////////////////////////////////////////////////////////////////
// negate
////////////////////////////////////////////////////////////////////////////////
__device__ float2 operator-(float2 &a)
{
	return make_float2(-a.x, -a.y);
}
__device__ float3 operator-(float3 &a)
{
	return make_float3(-a.x, -a.y, -a.z);
}
__device__ float4 operator-(float4 &a)
{
	return make_float4(-a.x, -a.y, -a.z, -a.w);
}

////////////////////////////////////////////////////////////////////////////////
// addition
////////////////////////////////////////////////////////////////////////////////

__device__ float2 operator+(float2 a, float2 b)
{
	return make_float2(a.x + b.x, a.y + b.y);
}
__device__ void operator+=(float2 &a, float2 b)
{
	a.x += b.x;
	a.y += b.y;
}
__device__ float2 operator+(float2 a, float b)
{
	return make_float2(a.x + b, a.y + b);
}
__device__ float2 operator+(float b, float2 a)
{
	return make_float2(a.x + b, a.y + b);
}
__device__ void operator+=(float2 &a, float b)
{
	a.x += b;
	a.y += b;
}
//
__device__ float3 operator+(float3 a, float3 b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ void operator+=(float3 &a, float3 b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}
__device__ float3 operator+(float3 a, float b)
{
	return make_float3(a.x + b, a.y + b, a.z + b);
}
__device__ void operator+=(float3 &a, float b)
{
	a.x += b;
	a.y += b;
	a.z += b;
}
__device__ float3 operator+(float b, float3 a)
{
	return make_float3(a.x + b, a.y + b, a.z + b);
}
//
__device__ float4 operator+(float4 a, float4 b)
{
	return make_float4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
__device__ void operator+=(float4 &a, float4 b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	a.w += b.w;
}
__device__ float4 operator+(float4 a, float b)
{
	return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}
__device__ float4 operator+(float b, float4 a)
{
	return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}
__device__ void operator+=(float4 &a, float b)
{
	a.x += b;
	a.y += b;
	a.z += b;
	a.w += b;
}

////////////////////////////////////////////////////////////////////////////////
// subtract
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator-(float2 a, float2 b)
{
	return make_float2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(float2 &a, float2 b)
{
	a.x -= b.x;
	a.y -= b.y;
}
inline __host__ __device__ float2 operator-(float2 a, float b)
{
	return make_float2(a.x - b, a.y - b);
}
inline __host__ __device__ float2 operator-(float b, float2 a)
{
	return make_float2(b - a.x, b - a.y);
}
inline __host__ __device__ void operator-=(float2 &a, float b)
{
	a.x -= b;
	a.y -= b;
}
//
inline __host__ __device__ float3 operator-(float3 a, float3 b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ void operator-=(float3 &a, float3 b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
}
inline __host__ __device__ float3 operator-(float3 a, float b)
{
	return make_float3(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ float3 operator-(float b, float3 a)
{
	return make_float3(b - a.x, b - a.y, b - a.z);
}
inline __host__ __device__ void operator-=(float3 &a, float b)
{
	a.x -= b;
	a.y -= b;
	a.z -= b;
}
//
inline __host__ __device__ float4 operator-(float4 a, float4 b)
{
	return make_float4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}
inline __host__ __device__ void operator-=(float4 &a, float4 b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	a.w -= b.w;
}
inline __host__ __device__ float4 operator-(float4 a, float b)
{
	return make_float4(a.x - b, a.y - b, a.z - b,  a.w - b);
}
inline __host__ __device__ void operator-=(float4 &a, float b)
{
	a.x -= b;
	a.y -= b;
	a.z -= b;
	a.w -= b;
}

////////////////////////////////////////////////////////////////////////////////
// multiply
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator*(float2 a, float2 b)
{
	return make_float2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ void operator*=(float2 &a, float2 b)
{
	a.x *= b.x;
	a.y *= b.y;
}
inline __host__ __device__ float2 operator*(float2 a, float b)
{
	return make_float2(a.x * b, a.y * b);
}
inline __host__ __device__ float2 operator*(float b, float2 a)
{
	return make_float2(b * a.x, b * a.y);
}
inline __host__ __device__ void operator*=(float2 &a, float b)
{
	a.x *= b;
	a.y *= b;
}

inline __host__ __device__ int2 operator*(int2 a, int2 b)
{
	return make_int2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ void operator*=(int2 &a, int2 b)
{
	a.x *= b.x;
	a.y *= b.y;
}
inline __host__ __device__ int2 operator*(int2 a, int b)
{
	return make_int2(a.x * b, a.y * b);
}
inline __host__ __device__ int2 operator*(int b, int2 a)
{
	return make_int2(b * a.x, b * a.y);
}
inline __host__ __device__ void operator*=(int2 &a, int b)
{
	a.x *= b;
	a.y *= b;
}

inline __host__ __device__ uint2 operator*(uint2 a, uint2 b)
{
	return make_uint2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ void operator*=(uint2 &a, uint2 b)
{
	a.x *= b.x;
	a.y *= b.y;
}
inline __host__ __device__ uint2 operator*(uint2 a, uint b)
{
	return make_uint2(a.x * b, a.y * b);
}
inline __host__ __device__ uint2 operator*(uint b, uint2 a)
{
	return make_uint2(b * a.x, b * a.y);
}
inline __host__ __device__ void operator*=(uint2 &a, uint b)
{
	a.x *= b;
	a.y *= b;
}

inline __host__ __device__ float3 operator*(float3 a, float3 b)
{
	return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ void operator*=(float3 &a, float3 b)
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
}
inline __host__ __device__ float3 operator*(float3 a, float b)
{
	return make_float3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ float3 operator*(float b, float3 a)
{
	return make_float3(b * a.x, b * a.y, b * a.z);
}
inline __host__ __device__ void operator*=(float3 &a, float b)
{
	a.x *= b;
	a.y *= b;
	a.z *= b;
}

inline __host__ __device__ int3 operator*(int3 a, int3 b)
{
	return make_int3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ void operator*=(int3 &a, int3 b)
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
}
inline __host__ __device__ int3 operator*(int3 a, int b)
{
	return make_int3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ int3 operator*(int b, int3 a)
{
	return make_int3(b * a.x, b * a.y, b * a.z);
}
inline __host__ __device__ void operator*=(int3 &a, int b)
{
	a.x *= b;
	a.y *= b;
	a.z *= b;
}

inline __host__ __device__ uint3 operator*(uint3 a, uint3 b)
{
	return make_uint3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ void operator*=(uint3 &a, uint3 b)
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
}
inline __host__ __device__ uint3 operator*(uint3 a, uint b)
{
	return make_uint3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ uint3 operator*(uint b, uint3 a)
{
	return make_uint3(b * a.x, b * a.y, b * a.z);
}
inline __host__ __device__ void operator*=(uint3 &a, uint b)
{
	a.x *= b;
	a.y *= b;
	a.z *= b;
}

inline __host__ __device__ float4 operator*(float4 a, float4 b)
{
	return make_float4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}
inline __host__ __device__ void operator*=(float4 &a, float4 b)
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	a.w *= b.w;
}
inline __host__ __device__ float4 operator*(float4 a, float b)
{
	return make_float4(a.x * b, a.y * b, a.z * b,  a.w * b);
}
inline __host__ __device__ float4 operator*(float b, float4 a)
{
	return make_float4(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline __host__ __device__ void operator*=(float4 &a, float b)
{
	a.x *= b;
	a.y *= b;
	a.z *= b;
	a.w *= b;
}

inline __host__ __device__ int4 operator*(int4 a, int4 b)
{
	return make_int4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}
inline __host__ __device__ void operator*=(int4 &a, int4 b)
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	a.w *= b.w;
}
inline __host__ __device__ int4 operator*(int4 a, int b)
{
	return make_int4(a.x * b, a.y * b, a.z * b,  a.w * b);
}
inline __host__ __device__ int4 operator*(int b, int4 a)
{
	return make_int4(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline __host__ __device__ void operator*=(int4 &a, int b)
{
	a.x *= b;
	a.y *= b;
	a.z *= b;
	a.w *= b;
}

inline __host__ __device__ uint4 operator*(uint4 a, uint4 b)
{
	return make_uint4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}
inline __host__ __device__ void operator*=(uint4 &a, uint4 b)
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	a.w *= b.w;
}
inline __host__ __device__ uint4 operator*(uint4 a, uint b)
{
	return make_uint4(a.x * b, a.y * b, a.z * b,  a.w * b);
}
inline __host__ __device__ uint4 operator*(uint b, uint4 a)
{
	return make_uint4(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline __host__ __device__ void operator*=(uint4 &a, uint b)
{
	a.x *= b;
	a.y *= b;
	a.z *= b;
	a.w *= b;
}

////////////////////////////////////////////////////////////////////////////////
// normalize
////////////////////////////////////////////////////////////////////////////////

__device__ float2 normalize(float2 v)
{
	float invLen = rsqrtf(cu_dot(v, v));
	return v * invLen;
}
__device__ float3 normalize(float3 v)
{
	float invLen = rsqrtf(cu_dot(v, v));
	return v * invLen;
}
__device__ float4 normalize(float4 v)
{
	float invLen = rsqrtf(cu_dot(v, v));
	return v * invLen;
}

////////////////////////////////////////////////////////////////////////////////
// length
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float length(float2 v)
{
	//return sqrtf(cu_dot(v, v));
	return __fsqrt_rd(cu_dot(v, v));
}
inline __host__ __device__ float length(float3 v)
{
	//return sqrtf(cu_dot(v, v));
	return __fsqrt_rd(cu_dot(v, v));
}
inline __host__ __device__ float length(float4 v)
{
	//return sqrtf(cu_dot(v, v));
	return __fsqrt_rd(cu_dot(v, v));
}

////////////////////////////////////////////////////////////////////////////////
// fmod
////////////////////////////////////////////////////////////////////////////////

__device__ float2 fmodf(float2 a, float2 b)
{
	return make_float2(fmodf(a.x, b.x), fmodf(a.y, b.y));
}
__device__ float3 fmodf(float3 a, float3 b)
{
	return make_float3(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z));
}
__device__ float4 fmodf(float4 a, float4 b)
{
	return make_float4(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z), fmodf(a.w, b.w));
}

////////////////////////////////////////////////////////////////////////////////
// reflect
// - returns reflection of incident ray I around surface normal N
// - N should be normalized, reflected vector's length is equal to length of I
////////////////////////////////////////////////////////////////////////////////

__device__ float3 reflect(float3 i, float3 n)
{
	return i - 2.0f * n * cu_dot(n,i);
}

////////////////////////////////////////////////////////////////////////////////
// cross product
////////////////////////////////////////////////////////////////////////////////

__device__ float3 cu_cross(float3 a, float3 b)
{
	return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}