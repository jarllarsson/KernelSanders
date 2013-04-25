#ifndef RAYTRACER_CONSTANTBUFFER_H
#define RAYTRACER_CONSTANTBUFFER_H

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
	float a;
	float b;
	float c;
	float d;
};

#endif