#ifndef RAYTRACE_SCENE_H
#define RAYTRACE_SCENE_H

#include "RaytraceSetup.h"
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
	Light light[AMOUNTOFLIGHTS];
	//Sphere sphere[AMOUNTOFSPHERES];
	//Plane plane[AMOUNTOFPLANES];
	//Tri tri[AMOUNTOFTRIS];
	Box box[AMOUNTOFBOXES];
};

#endif