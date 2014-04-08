#ifndef INTERSECT_PLANE_H
#define INTERSECT_PLANE_H

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



__device__ bool IntersectPlane(const Plane* in_plane, const Ray* in_ray, Intersection* inout_intersection, bool storeResult, float p_time)
{
	// plane intersection
	float t = (in_plane->distance - cu_dot(in_ray->origin,in_plane->normal)) / cu_dot(in_ray->dir,in_plane->normal);
	if(t > 0.001f && t < inout_intersection->dist)
	{
		if (storeResult)
		{
			inout_intersection->dist = t; 
			inout_intersection->surface=in_plane->mat;
			inout_intersection->pos=in_ray->origin+t*in_ray->dir;

			float i=max(0.0f,1.0f-0.01f*(cu_length(inout_intersection->pos)));
			float sini=(1.0f+sin(100.0f*(1.0f-i*i*i)-p_time*2.0f))*0.5f,cosi=(1.0f-cos(100.0f*(1.0f-i*i*i)-p_time*2.0))*0.5f;
			float4 waven=cu_normalize(in_plane->normal+0.01f*i*make_float4(sini,0.0f,cosi,0.0f));
			inout_intersection->surface.specular *= i*i*(2.0f+sini+cosi)*0.2f;
			inout_intersection->normal=waven;
		}
		return true;
	}
	return false;
}

#endif