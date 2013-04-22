#pragma once
#include <d3d11.h>
#include "InteropResourceMapping.h"
#include "RaytraceKernel.h"
#include <vector>

using namespace std;

// =======================================================================================
//                                      KernelDevice
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Manager of kernels
///        
/// # KernelDevice
/// 
/// 21-4-2013 Jarl Larsson
///---------------------------------------------------------------------------------------

class KernelDevice
{
public:
	enum KernelJob
	{
		J_RAYTRACEWORLD,
	};

	KernelDevice(void* p_device);
	virtual ~KernelDevice();

	void registerGBuffer(vector<void*> p_buffer);

	void update(float dt);

	void executeKernelJob( float p_dt, KernelJob p_jobId );
protected:
private:
	int m_width, m_height;
	RaytraceKernel* m_raytracer;

	bool assertCudaResult(cudaError_t p_result);

	InteropResourceMapping	m_gbufferHandle;
	ID3D11Device*			m_device;
};