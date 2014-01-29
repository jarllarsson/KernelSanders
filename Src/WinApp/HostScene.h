#pragma once

#include "HostLightStructures.h"
#include "HostMaterial.h"
#include "HostPrimitives.h"
#include "RaytraceDefines.h"
// =======================================================================================
//                                      HostScene
// =======================================================================================

struct Scene
{	
	Light light[AMOUNTOFLIGHTS];
	Sphere sphere[AMOUNTOFSPHERES];
	Plane plane[AMOUNTOFPLANES];
	Tri tri[AMOUNTOFTRIS];
	Box box[AMOUNTOFBOXES];
};

