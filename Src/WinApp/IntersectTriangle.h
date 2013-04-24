#pragma once
#include "KernelRaytraceCommon.h"
#include "KernelRaytraceLight.h"



#define TRIEPSILON 1.0f/100000.0f

__device__ bool IntersectTriangle(const Tri* in_tri, const Ray* in_ray, Intersection* inout_intersection, bool storeResult)
{
	bool result = false;
	float4 edge1 = in_tri->vertices[1] - in_tri->vertices[0];
	float4 edge2 = in_tri->vertices[2] - in_tri->vertices[0];
	float4 q = cross(in_ray->dir,edge2);
	float a = dot(edge1,q);		// determinant
	if (a > -TRIEPSILON && a < TRIEPSILON) return false; // return if determinant a is close to zero

	float f = 1.0f/a;
	float4 s = in_ray->origin - in_tri->vertices[0];
	float u = f*dot(s,q);
	if (u<0.0) return false;			// return if u-coord of ray is outside the edge of the triangle

	float4 r = cross(s,edge1);
	float v = f*dot(in_ray->dir,r);
	if (v < 0.0f || u+v > 1.0f) return false; // return if v- or u+v-coord of ray is outside

	float t = f*dot(edge2,r);
	if(t > 0.001f && t < inout_intersection->dist)
	{
		if (storeResult)
		{
			inout_intersection->dist=t;
			inout_intersection->surface=in_tri->mat;
			inout_intersection->pos=in_ray->origin+t*in_ray->dir;
			inout_intersection->normal=fast_normalize(cross(edge1,edge2));
		}
		result = true;
	}
	return result;
}
