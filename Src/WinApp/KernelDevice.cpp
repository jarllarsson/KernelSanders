#include "KernelDevice.h"
#include "KernelException.h"

#include <cuda.h>
#include <builtin_types.h>
#include <cuda_runtime_api.h>
#include <cuda_d3d11_interop.h>
#include <DebugPrint.h>
#include "KernelHelper.h"
#include <glm\gtc\type_ptr.hpp>
#include "RaytraceDefines.h"


KernelDevice::KernelDevice( void* p_device )
{
	ZeroMemory(&m_cb,sizeof(m_cb));
	m_width=1; m_height=1;
	m_device=(ID3D11Device*)p_device;
	cudaError_t res = cudaD3D11SetDirect3DDevice(m_device);
	if (!KernelHelper::assertCudaResult(res))
		throw KernelException(res,"Error registering d3d-device",__FILE__,__FUNCTION__,__LINE__);

	ZeroMemory(&m_cb,sizeof(RaytraceConstantBuffer));
	m_cb.m_drawMode = RAYTRACEDRAWMODE_REGULAR;

	m_vertArray=NULL;
	m_uvsArray=NULL;
	m_normsArray=NULL;
	m_indicesArray=NULL;
	m_trisArray=NULL;
	m_nodesArray=NULL;
	m_leafArray=NULL;
	m_nodeIndicesArray=NULL;



	m_raytracer = new RaytraceKernel();
}

KernelDevice::~KernelDevice()
{

	cudaError_t res = cudaGraphicsUnregisterResource( m_gbufferHandle.m_textureResource);
	KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);
	res = cudaFree(m_gbufferHandle.m_textureLinearMem);
	KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);

	// Global memory
	res = cudaFree(m_vertArray);
	KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);
	res = cudaFree(m_uvsArray);
	KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);
	res = cudaFree(m_normsArray);
	KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);
	res = cudaFree(m_indicesArray);
	KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);
	res = cudaFree(m_trisArray);
	KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);
	res = cudaFree(m_nodesArray);
	KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);
	res = cudaFree(m_leafArray);
	KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);
	res = cudaFree(m_nodeIndicesArray);
	KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);
	delete m_raytracer;
}

void KernelDevice::update( float p_dt, TempController* p_tmpCam, int p_drawMode, int p_shadowMode )
{
	// CONST BUFFER DATA
	// time
	m_cb.m_time += p_dt;
	
	// settings
	m_cb.m_shadowMode=p_shadowMode;
	m_cb.m_drawMode=p_drawMode;

	// camera
	memcpy(&m_cb.m_camPos, glm::value_ptr(p_tmpCam->getPos()), sizeof(m_cb.m_camPos));
	memcpy(&m_cb.m_cameraRotationMat, glm::value_ptr(p_tmpCam->getRotationMatrix()), sizeof(m_cb.m_cameraRotationMat));
	glm::vec2 rayscale=p_tmpCam->getFovXY();
	m_cb.m_rayDirScaleX = rayscale.x;
	m_cb.m_rayDirScaleY = rayscale.y;
	// ======================================

	Texture* texBuf = *m_gbufferHandle.m_texture/*[i]*/;
	if (texBuf->isDirty())
	{
		cudaError_t res;

		m_width=texBuf->m_width;
		m_height=texBuf->m_height;


		// ***********
		res = cudaFree(m_gbufferHandle.m_textureLinearMem/*[i]*/);
		KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);
		// ***********

		res = cudaGraphicsD3D11RegisterResource( &(m_gbufferHandle.m_textureResource),
			texBuf->m_textureBuffer,
			cudaGraphicsRegisterFlagsNone);

		KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);

		// ***********

		res = cudaMallocPitch(&m_gbufferHandle.m_textureLinearMem, 
			&m_gbufferHandle.m_pitch, m_width * sizeof(float) * 4, (size_t)m_height);
		KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);
		res = cudaMemset(m_gbufferHandle.m_textureLinearMem, 1, m_gbufferHandle.m_pitch * m_height);	
		KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);
		// ***********


		texBuf->unsetDirtyFlag();
	}

}

void KernelDevice::executeKernelJob( float p_dt, KernelJob p_jobId )
{
	switch (p_jobId)
	{
	case J_RAYTRACEWORLD:
		{
			RaytraceKernelData blob;
			blob.m_width=m_width; blob.m_height=m_height;
			blob.m_textureResource = m_gbufferHandle.m_textureResource;
			blob.m_textureLinearMemDevice = m_gbufferHandle.m_textureLinearMem;
			blob.m_pitch = &m_gbufferHandle.m_pitch;

			// Cuda memory
			blob.m_cb=&m_cb;
			blob.m_vertsLinearMemDeviceRef			= &m_vertArray;
			blob.m_normsLinearMemDeviceRef			= &m_normsArray;
			blob.m_uvsLinearMemDeviceRef			= &m_uvsArray;
			blob.m_indicesLinearMemDeviceRef		= &m_indicesArray;
			blob.m_trisLinearMemDeviceRef			= &m_trisArray;
			blob.m_nodesLinearMemDeviceRef			= &m_nodesArray;
			blob.m_nodeLeavesLinearMemDeviceRef		= &m_leafArray;
			blob.m_nodeIndicesLinearMemDeviceRef	= &m_nodeIndicesArray;

			// Scene desc
			blob.m_hostScene = m_sceneMgr->getScenePtr();


			m_raytracer->Execute((KernelData*)&blob,p_dt);
			break;
		}
		
	}
}


void KernelDevice::registerCanvas( void** p_texture )
{

	m_gbufferHandle.m_texture = (Texture**)p_texture;

	Texture* texBuf = *m_gbufferHandle.m_texture/*[i]*/;
	cudaError_t res = cudaGraphicsD3D11RegisterResource( &(m_gbufferHandle.m_textureResource/*[i]*/),
		texBuf->m_textureBuffer,
		cudaGraphicsRegisterFlagsNone);

	m_width=texBuf->m_width;
	m_height=texBuf->m_height;


	if (!KernelHelper::assertCudaResult(res))
		throw KernelException(res,"Error registering GBuffer texture",__FILE__,__FUNCTION__,__LINE__);


	res = cudaMallocPitch(&m_gbufferHandle.m_textureLinearMem, &m_gbufferHandle.m_pitch, m_width * sizeof(float) * 4, (size_t)m_height);
	if (!KernelHelper::assertCudaResult(res))
		throw KernelException(res,"Error registering GBuffer texture",__FILE__,__FUNCTION__,__LINE__);
	res = cudaMemset(m_gbufferHandle.m_textureLinearMem, 1, m_gbufferHandle.m_pitch * m_height);	
	if (!KernelHelper::assertCudaResult(res))
		throw KernelException(res,"Error registering GBuffer texture",__FILE__,__FUNCTION__,__LINE__);


	texBuf->unsetDirtyFlag();


}


void KernelDevice::registerSceneMgr( HostSceneManager* p_sceneMgr )
{
	m_sceneMgr=p_sceneMgr;
}



