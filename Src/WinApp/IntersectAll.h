#ifndef INTERSECT_ALL_H
#define INTERSECT_ALL_H


#include <iostream> 
#include <vector> 
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector_types.h>

#include "RaytraceConstantBuffer.h"
#include "KernelMathHelper.h"
#include "Scene.h"
#include "Ray.h"
#include "IntersectionInfo.h"

#include "IntersectSphere.h"
#include "IntersectPlane.h"
#include "IntersectTriangle.h"
#include "IntersectBox.h"


// =======================================================================================
//                                      IntersectAll
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Brief
///        
/// # IntersectAll
/// 
/// 25-4-2013 Jarl Larsson
///---------------------------------------------------------------------------------------

__device__ bool IntersectAll(const Scene* in_scene, const Ray* in_ray, Intersection* inout_intersection, bool breakOnFirst,bool previousResult)
{
	bool result=previousResult;
	bool storeResults = !breakOnFirst;

    #pragma unroll MAXSPHERES
	for (int i=0;i<MAXSPHERES;i++)
	{
		result|=IntersectSphere(&(in_scene->sphere[i]), in_ray, inout_intersection, storeResults);
		if (result && breakOnFirst) 
			return true;
	}	// for each sphere	

	#pragma unroll MAXPLANES 
	for (int i=0;i<MAXPLANES;i++)
	{
		result|=IntersectPlane(&(in_scene->plane[i]), in_ray, inout_intersection,storeResults);
		if (result && breakOnFirst) 
			return true;
	}	// for each plane


	#pragma unroll MAXTRIS
	for (int i=0;i<MAXTRIS;i++)
	{
		result|=IntersectTriangle(&(in_scene->tri[i]), in_ray, inout_intersection,storeResults);
		if (result && breakOnFirst) 
			return true;

	}	// for each tri


	#pragma unroll MAXBOXES
	for (int i=0;i<MAXBOXES;i++)
	{
		result|=IntersectBox(&(in_scene->box[i]), in_ray, inout_intersection,storeResults);
		if (result && breakOnFirst) 
			return true;
	}	// for each box
	

	
	// debug for lights
	if (!breakOnFirst)
	{
		Sphere t;
		for (int i=0;i<MAXLIGHTS-1;i++)
		{
			t.pos = in_scene->light[i].vec;
			t.rad = 0.5f;
			t.mat.diffuse = in_scene->light[i].diffuseColor;
			t.mat.specular = make_float4(1.0f, 1.0f, 1.0f,1.0f);
			t.mat.reflection = 0.0f;
			IntersectSphere(&(t), in_ray, inout_intersection,storeResults);
		}	// debug for each light
	}

	return result;
}

__device__ bool MarchAll(const Ray* in_ray, Intersection* inout_intersection, bool breakOnFirst, bool previousResult)
{
	bool result=previousResult;
	bool storeResults = !breakOnFirst;

	// Planet field
	Sphere f;

	f.pos = make_float4(0.0f,0.0f,0.0f,1.0f);
	f.rad = 0.5f;
	f.mat.diffuse = make_float4(1.0f, 0.0f, 1.0f,1.0f);
	f.mat.specular = make_float4(0.0f, 0.0f, 0.0f,0.0f);
	f.mat.reflection = 0.0f;


	result|=MarchSphere(&f, in_ray, inout_intersection, storeResults);
	
	return true;
}

#endif