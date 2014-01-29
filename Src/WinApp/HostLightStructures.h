#pragma once
#include <glm/glm.hpp>
// =======================================================================================
//                                      HostLightStructures
// =======================================================================================



struct SurfaceLightingData
{
	glm::vec4 diffuseColor;
	glm::vec4 specularColor;
};

struct Light
{
	float diffusePower;
	float specularPower;
	glm::vec4 vec; // direction(w=0) or position(w=1)
	glm::vec4 diffuseColor;
	glm::vec4 specularColor;
};
