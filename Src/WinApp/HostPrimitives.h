#pragma once

// =======================================================================================
//                                    Host Primitives
// =======================================================================================

#include <glm/glm.hpp>
#include "HostMaterial.h"


struct Box
{
	glm::vec4 pos;		    // The center point of the box in world coordinates
	glm::vec4 sides[3];		// normalized side directions u,v,w
	float hlengths[3];	// positive half-lengths from box center
	Material mat;
	float pad;
};


struct Tri
{
	glm::vec4 vertices[3];
	Material mat;
};

struct TriPart
{
	glm::vec4 vertices[3];
};

struct Mesh
{
	unsigned int polyCount;
	TriPart* polygons;
	Material mat;
};

struct Plane
{
	float distance;
	glm::vec4 normal;
	Material mat;
	float pad[3];
};


struct Sphere
{
	glm::vec4 pos;
	float rad;
	Material mat;
	float pad[3];
};
