#ifndef INTERSECT_TRIANGLE_H
#define INTERSECT_TRIANGLE_H

#include <vector> 
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector_types.h>


#include "KernelMathHelper.h"
#include "RaytraceLighting.h"
#include "Primitives.h"
#include "Ray.h"
#include "DeviceResources.h"
#include "KernelTextureHelper.h"


using std::vector; 

#define TRIEPSILON 1.0f/100000.0f

// Compute barycentric coordinates (u, v, w) for
// point p with respect to triangle (a, b, c)
// Cramer's rule
__device__ float3 Barycentric(float3& p_point, float3& a, float3& b, float3& c)
{
	float3 v0 = b - a, v1 = c - a, v2 = p_point - a;
	float inv_den = 1.0f / (v0.x * v1.y - v1.x * v0.y);
	float v = (v2.x * v1.y - v1.x * v2.y) * inv_den;
	float w = (v0.x * v2.y - v2.x * v0.y) * inv_den;
	float u = 1.0f - v - w;
	return make_float3(u,v,w);
//  float3 v0 = b - a, v1 = c - a, v2 = p_point - a;
//  float d00 = cu_dot(v0, v0);
//  float d01 = cu_dot(v0, v1);
//  float d11 = cu_dot(v1, v1);
//  float d20 = cu_dot(v2, v0);
//  float d21 = cu_dot(v2, v1);
//  float invDenom = 1.0 / (d00 * d11 - d01 * d01);
//  float v = (d11 * d20 - d01 * d21) * invDenom;
//  float w = (d00 * d21 - d01 * d20) * invDenom;
//  float u = 1.0f - v - w;
//  return make_float3(u,v,w);
}

__device__ float3 InterpolateUV(float3& p_barycentric, float3& p_uvA, float3& p_uvB, float3& p_uvC)
{
	float3 uv = p_uvA * p_barycentric.x + p_uvB * p_barycentric.y + p_uvC * p_barycentric.z;
	return uv;
}

__device__ float2 InterpolateUV(float2& p_hitUv, float3& p_uvA, float3& p_uvB, float3& p_uvC)
{
	//float2 uvA=make_float2(p_uvA.x,p_uvA.y);
	//float2 uvB=make_float2(p_uvB.x,p_uvB.y); 
	//float2 uvC=make_float2(p_uvC.x,p_uvC.y);
	//float2 uv = uvA * p_hitUv.x + uvB * p_hitUv.y + uvC * 1.0f;
	//return uv;
	float2 uvA=make_float2(p_uvA.x,p_uvA.y);
	float2 uvB=make_float2(p_uvB.x,p_uvB.y); 
	float2 uvC=make_float2(p_uvC.x,p_uvC.y);
	float2 uv = make_float2(uvA.x + p_hitUv.x * (uvB.x-uvA.x) + p_hitUv.x * (uvC.x-uvA.x),
							uvA.y + p_hitUv.y * (uvB.y-uvA.y) + p_hitUv.y * (uvC.y-uvA.y));
	return uv;
}

__device__ bool IntersectTriangle(const Tri* in_tri, const Ray* in_ray, Intersection* inout_intersection, bool storeResult)
{
	bool result = false;
	float3 vert0 = in_tri->vertices[0];
	float3 vert1 = in_tri->vertices[1];
	float3 vert2 = in_tri->vertices[2];
	float3 edge1 = vert1 - vert0;
	float3 edge2 = vert2 - vert0;
	float3 dir = make_float3(in_ray->dir.x,in_ray->dir.y,in_ray->dir.z);
	float3 orig = make_float3(in_ray->origin.x,in_ray->origin.y,in_ray->origin.z);
	float3 q = cu_cross(dir,edge2);
	float a = cu_dot(edge1,q);		// determinant
	if (a > -TRIEPSILON && a < TRIEPSILON) return false; // return if determinant a is close to zero

	float f = 1.0f/a;
	float3 s = orig - vert0;
	float u = f*cu_dot(s,q);
	if (u<0.0) return false;			// return if u-coord of ray is outside the edge of the triangle

	float3 r = cu_cross(s,edge1);
	float v = f*cu_dot(dir,r);
	if (v < 0.0f || u+v > 1.0f) return false; // return if v- or u+v-coord of ray is outside

	float t = f*cu_dot(edge2,r);
	if(t > 0.001f && t < inout_intersection->dist)
	{
		if (storeResult)
		{
			inout_intersection->dist=t;
			inout_intersection->surface=in_tri->mat;
			inout_intersection->pos=in_ray->origin+t*in_ray->dir;
			float3 normvec = cu_normalize(cu_cross(edge1,edge2));
			inout_intersection->normal=make_float4(normvec.x,normvec.y,normvec.z,inout_intersection->normal.w);
		}
		result = true;
	}
	return result;
}


__device__ bool IntersectTriangle(const float3* vertArr, const float3* uvArr, const float3* normArr, 
								  unsigned int i0, unsigned int i1, unsigned int i2, 
								  Material* mat,
								  const Ray* in_ray, Intersection* inout_intersection, bool storeResult)
{
	bool result = false;
	float3 vert0 = vertArr[i0];
	float3 vert1 = vertArr[i1];
	float3 vert2 = vertArr[i2];
	float3 n0 = normArr[i0];
	float3 n1 = normArr[i1];
	float3 n2 = normArr[i2];
	float3 uv0 = uvArr[i0];
	float3 uv1 = uvArr[i1];
	float3 uv2 = uvArr[i2];
	float3 edge1 = vert1 - vert0;
	float3 edge2 = vert2 - vert0;
	float3 dir = make_float3(in_ray->dir.x,in_ray->dir.y,in_ray->dir.z);
	float3 orig = make_float3(in_ray->origin.x,in_ray->origin.y,in_ray->origin.z);
	float3 q = cu_cross(dir,edge2);
	float a = cu_dot(edge1,q);		// determinant
	if (a > -TRIEPSILON && a < TRIEPSILON) return false; // return if determinant a is close to zero

	float f = 1.0f/a;
	float3 s = orig - vert0;
	float u = f*cu_dot(s,q);
	if (u<0.0) return false;			// return if u-coord of ray is outside the edge of the triangle

	float3 r = cu_cross(s,edge1);
	float v = f*cu_dot(dir,r);
	if (v < 0.0f || u+v > 1.0f) return false; // return if v- or u+v-coord of ray is outside

	float t = f*cu_dot(edge2,r);
	if(t > 0.001f && t < inout_intersection->dist)
	{
		if (storeResult)
		{
			inout_intersection->dist=t;
			inout_intersection->surface=*mat;
			float4 triPos=in_ray->origin+t*in_ray->dir;
			inout_intersection->pos=triPos;

			/*float3 barycentric=Barycentric(make_float3(u,v,0.0f),
										   vert0,vert1,vert2);				*/
			float3 uv=InterpolateUV(make_float3((1.0f-u-v),u,v),uv0,uv1,uv2);			//???
			float4 texCol=tex2D(tex,uv.x,uv.y);
				//make_float4(uv1.x,uv1.y,uv1.z,1.0f);
				//tex2D(tex,uv.x,uv.y);
				//sampleTextureRGB(tex,make_float2(uv.x,uv.y));
			inout_intersection->surface.diffuse = texCol;
			//inout_intersection->surface.diffuse += make_float4(0.0f,u,v,0.0f);
				//texCol;

			float3 normvec = cu_normalize(n0+u*(n1-n0)+v*(n2-n0));
			inout_intersection->normal=make_float4(normvec.x,normvec.y,normvec.z,inout_intersection->normal.w);
		}
		result = true;
	}
	return result;
}

#endif