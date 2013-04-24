#pragma once
#include "KernelRaytraceCommon.h"
#include "KernelRaytraceLight.h"



__device__ bool IntersectPlane(const Plane* in_plane, const Ray* in_ray, Intersection* inout_intersection, bool storeResult)
{
	// sphere intersection
	float t = (in_plane->distance - dot(in_ray->origin,in_plane->normal)) / dot(in_ray->dir,in_plane->normal);
	if(t > 0.001f && t < inout_intersection->dist)
	{
		if (storeResult)
		{
			inout_intersection->dist = t; 
			inout_intersection->surface=in_plane->mat;
			inout_intersection->pos=in_ray->origin+t*in_ray->dir;
			inout_intersection->normal=in_plane->normal;
		}
		return true;
	}
	return false;
}
