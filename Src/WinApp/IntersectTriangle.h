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


using std::vector; 

#define TRIEPSILON 1.0f/100000.0f

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

__device__ bool IntersectTriangle(const float3* vertArr, const float3* normArr, 
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
			inout_intersection->pos=in_ray->origin+t*in_ray->dir;

			float3 normvec = cu_normalize(n0+u*(n1-n0)+v*(n2-n0));
			inout_intersection->normal=make_float4(normvec.x,normvec.y,normvec.z,inout_intersection->normal.w);
		}
		result = true;
	}
	return result;
}

#endif