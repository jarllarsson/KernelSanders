#pragma once

#include "HostLightStructures.h"
#include "HostMaterial.h"
#include "HostPrimitives.h"
#include <vector>

using namespace std;
// =======================================================================================
//                                      HostScene
// =======================================================================================

struct HScene
{	
	/*Light light[MAXLIGHTS];
	Sphere sphere[MAXSPHERES];
	Plane plane[MAXPLANES];
	Tri tri[MAXTRIS];
	Box box[MAXBOXES];*/

	vector<HLight> light;
	vector<HSphere> sphere;
	vector<HPlane> plane;
	vector<HTriPart> tri;
	vector<HBox> box;
};

