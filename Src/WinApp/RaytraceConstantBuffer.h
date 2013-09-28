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
	float m_rayDirScaleX;
	float m_rayDirScaleY;
	float m_time;
	float m_cameraRotationMat[16];
	float m_camPos[4];
};

#endif