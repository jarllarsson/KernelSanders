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
	int numTris, numVerts, numIndices;
	Light light[MAXLIGHTS];
	Sphere sphere[MAXSPHERES];
	Plane plane[MAXPLANES];
	Tri tri[MAXTRIS];
	Box box[MAXBOXES];
	float3 meshVerts[MAXMESHLOCAL_VERTSBIN];
	float3 meshNorms[MAXMESHLOCAL_VERTSBIN];
	unsigned int meshIndices[MAXMESHLOCAL_INDICESBIN];
};

#endif