#pragma once
#include <cuda.h>
#include <builtin_types.h>
#include <cuda_runtime_api.h>
#include <cuda_d3d11_interop.h>
#include <device_launch_parameters.h>
#include "Texture.h"
#include <vector>

using namespace std;
// =======================================================================================
//                                  InteropResourceMapping
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Contains an array of references to textures and corresponding cuda resources
///        
/// # InteropResourceMapping
/// 
/// 21-4-2013 Jarl Larsson
///---------------------------------------------------------------------------------------


struct InteropResourceMapping
{
	vector<Texture*> m_texture;
	vector<cudaGraphicsResource*> m_textureResource;
};