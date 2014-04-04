#pragma once

#include "HostLightStructures.h"
#include "HostMaterial.h"
#include "HostPrimitives.h"
#include <vector>
#include "KDNode.h"
#include "KDBounds.h"

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
		BOX,
		MESH
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
	// Mesh
	vector<glm::vec3> meshVerts;
	vector<glm::vec2> meshUVs;
	vector<unsigned int> meshIndices;
	vector<glm::vec3> meshNorms;
	// KD
	vector<KDNode> KDnode;	
	vector<KDLeaf> KDleaves;
	vector<unsigned int> KDindices;
	KDBounds KDRootBounds;

	// flags
	void setDirty(OBJTYPE p_objType, bool p_status=true);
	bool isDirty(OBJTYPE p_objType);

private:
	bool lightDirty;
	bool sphereDirty;
	bool planeDirty;
	bool triDirty;
	bool boxDirty;
	bool meshDirty;
};

