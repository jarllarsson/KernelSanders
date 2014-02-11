#pragma once

// =======================================================================================
//                                    Host Primitives
// =======================================================================================

#include <glm/glm.hpp>
#include "HostMaterial.h"


struct HBox
{
	glm::vec4 pos;		    // The center point of the box in world coordinates
	glm::vec4 sides[3];		// normalized side directions u,v,w
	float hlengths[3];	// positive half-lengths from box center
	HMaterial mat;
	float pad;
};


struct HTri
{
	glm::vec3 vertices[3];
	HMaterial mat;
};

struct HTriPart
{
	glm::vec3 vertices[3];
};

struct HMesh
{
	unsigned int polyCount;
	HTriPart* polygons;
	HMaterial mat;
};

struct HPlane
{
	float distance;
	glm::vec4 normal;
	HMaterial mat;
	float pad[3];
};


struct HSphere
{
	glm::vec4 pos;
	float rad;
	HMaterial mat;
	float pad[3];
};
