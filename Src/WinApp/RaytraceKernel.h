#pragma once

#include "IKernelHandler.h"
#include "RaytraceConstantBuffer.h"
#include <cuda.h>
#include <builtin_types.h>
#include <cuda_runtime_api.h>
#include <cuda_d3d11_interop.h>
#include <vector>
#include "HostScene.h"

using namespace std;



struct RaytraceKernelData : public KernelData
{
	int m_width; int m_height;
	// Render Texture
	cudaGraphicsResource*		m_textureResource;
	void*						m_textureLinearMemDevice;
	cudaArray*					m_textureView;
	size_t*						m_pitch;
	RaytraceConstantBuffer*		m_cb;
	// Scene data
	HScene*						m_hostScene;
	void**						m_vertsLinearMemDeviceRef;
	void**						m_normsLinearMemDeviceRef;
	void**						m_indicesLinearMemDeviceRef;
	void**						m_trisLinearMemDeviceRef;
};

// =======================================================================================
//                                   RaytraceKernel
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Kernel handler for raytracer
///        
/// # RaytraceKernel
/// 
/// 21-4-2013 Jarl Larsson
///---------------------------------------------------------------------------------------

class RaytraceKernel : public IKernelHandler
{
public:
	RaytraceKernel();
	virtual ~RaytraceKernel();


	virtual void SetPerKernelArgs();

	virtual void Execute(KernelData* p_data, float p_dt);
protected:
private:

};