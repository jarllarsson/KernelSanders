#pragma once
#include <glm/glm.hpp>
// =======================================================================================
//                                      HostLightStructures
// =======================================================================================



struct HSurfaceLightingData
{
	glm::vec4 diffuseColor;
	glm::vec4 specularColor;
};

struct HLight
{
	float diffusePower;
	float specularPower;
	glm::vec4 vec; // direction(w=0) or position(w=1)
	glm::vec4 diffuseColor;
	glm::vec4 specularColor;
};
