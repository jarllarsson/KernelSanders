#ifndef RAYTRACER_CONSTANTBUFFER_H
#define RAYTRACER_CONSTANTBUFFER_H

#define RAYTRACEDRAWMODE_REGULAR 0
#define RAYTRACEDRAWMODE_BLOCKDBG 1

// =======================================================================================
//                                   RaytraceConstantBuffer
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	A constant buffer for the raytracer kernel
///        
/// # RaytraceConstantBuffer
/// 
/// 22-4-2013 Jarl Larsson
///---------------------------------------------------------------------------------------

struct RaytraceConstantBuffer
{
	int m_drawMode;
	float a;
	float b;
	float c;
	float d;
};

#endif