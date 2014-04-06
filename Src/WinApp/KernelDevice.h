#pragma once
#include <d3d11.h>
#include "InteropResourceMapping.h"
#include "RaytraceKernel.h"
#include <vector>
#include "TempController.h"
#include "HostSceneManager.h"

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

	KernelDevice(void* p_device, MeasurementBin* p_measurer);
	virtual ~KernelDevice();

	void registerCanvas(void** p_texture);
	void registerSceneMgr(HostSceneManager* p_sceneMgr);

	void update(float p_dt, TempController* p_tmpCam, int p_drawMode, int p_shadowMode );

	void executeKernelJob( float p_dt, KernelJob p_jobId );
protected:
private:
	int m_width, m_height;
	RaytraceKernel* m_raytracer;

	// Rendering and interop (DX and CUDA shared)
	InteropResourceMapping	m_gbufferHandle;
	ID3D11Device*			m_device;

	// CUDA device memory	
	// Constant
	RaytraceConstantBuffer  m_cb;
	// Global, storage between calls
	void* m_vertArray;
	void* m_uvsArray;
	void* m_normsArray;
	void* m_indicesArray;
	void* m_trisArray;
	//
	void* m_nodesArray;
	void* m_leafArray;
	void* m_nodeIndicesArray;

	// Geometry
	HostSceneManager* m_sceneMgr;

};