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
	enum OBJTYPE
	{
		LIGHT,
		SPHERE,
		PLANE,
		TRI,
		BOX
	};
	/*Light light[MAXLIGHTS];
	Sphere sphere[MAXSPHERES];
	Plane plane[MAXPLANES];
	Tri tri[MAXTRIS];
	Box box[MAXBOXES];*/

	HScene();

	vector<HLight> light;
	vector<HSphere> sphere;
	vector<HPlane> plane;
	vector<HTriPart> tri;
	vector<HBox> box;
	// flags
	void setDirty(OBJTYPE p_objType, bool p_status=true);
	bool isDirty(OBJTYPE p_objType);

private:
	bool lightDirty;
	bool sphereDirty;
	bool planeDirty;
	bool triDirty;
	bool boxDirty;
};

