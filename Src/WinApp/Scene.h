#ifndef RAYTRACE_SCENE_H
#define RAYTRACE_SCENE_H

#include <vector> 
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector_types.h>
#include "RaytraceDefines.h"
#include "Primitives.h"
#include "LightStructures.h"
#include "RaytraceSurfaceMaterial.h"


#pragma comment(lib, "cudart") 

using std::vector; 


// =======================================================================================
//                                      Scene
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Brief
///        
/// # Scene
/// 
/// 25-4-2013 Jarl Larsson
///---------------------------------------------------------------------------------------


struct Scene
{	
	int numTris;
	Light light[MAXLIGHTS];
	Sphere sphere[MAXSPHERES];
	Plane plane[MAXPLANES];
	Tri tri[MAXTRIS];
	Box box[MAXBOXES];
	float3 meshVerts[MAXMESHLOCAL_VERTSBIN];
	int meshVerts[MAXMESHLOCAL_VERTSBIN];
};

#endif