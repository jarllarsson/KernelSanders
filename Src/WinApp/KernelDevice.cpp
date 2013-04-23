#include "KernelDevice.h"
#include "KernelException.h"

#include <cuda.h>
#include <builtin_types.h>
#include <cuda_runtime_api.h>
#include <cuda_d3d11_interop.h>
#include <DebugPrint.h>
#include "KernelHelper.h"


KernelDevice::KernelDevice( void* p_device )
{
	ZeroMemory(&cb,sizeof(cb));
	m_width=1; m_height=1;
	m_device=(ID3D11Device*)p_device;
	cudaError_t res = cudaD3D11SetDirect3DDevice(m_device);
	if (!KernelHelper::assertCudaResult(res))
		throw KernelException(res,"Error registering d3d-device",__FILE__,__FUNCTION__,__LINE__);

	m_raytracer = new RaytraceKernel();
}

KernelDevice::~KernelDevice()
{
// 	unsigned int gbufSz = static_cast<unsigned int>(m_gbufferHandle.m_texture.size());
// 	for (unsigned int i=0;i<m_gbufferHandle.m_texture.size();i++)
// 	{
		cudaError_t res = cudaGraphicsUnregisterResource( m_gbufferHandle.m_textureResource/*[i]*/);
		KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);
		res = cudaFree(m_gbufferHandle.m_textureLinearMem/*[i]*/);
		KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);
/*	}*/

	delete m_raytracer;
}

void KernelDevice::update( float dt )
{

	// Handle change in textures
	//unsigned int gbufSz = static_cast<unsigned int>(m_gbufferHandle.m_texture.size());
	//for (unsigned int i=0;i<gbufSz;i++)
	//{
		Texture* texBuf = *m_gbufferHandle.m_texture/*[i]*/;
		if (texBuf->isDirty())
		{
			cudaError_t res;
// 			if (i==0)
// 			{
				m_width=texBuf->m_width;
				m_height=texBuf->m_height;
			/*}*/
// 			res = cudaGraphicsUnregisterResource( m_gbufferHandle.m_textureResource);		
// 			KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);

			// ***********
			res = cudaFree(m_gbufferHandle.m_textureLinearMem/*[i]*/);
			KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);
			// ***********
			
			res = cudaGraphicsD3D11RegisterResource( &(m_gbufferHandle.m_textureResource/*[i]*/),
																 texBuf->m_textureBuffer,
																 cudaGraphicsRegisterFlagsNone);

			KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);

			// ***********

			res = cudaMallocPitch(&m_gbufferHandle.m_textureLinearMem/*[i]*/, &m_gbufferHandle.m_pitch, m_width * sizeof(float) * 4, (size_t)m_height);
			KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);
			res = cudaMemset(m_gbufferHandle.m_textureLinearMem/*[i]*/, 1, m_gbufferHandle.m_pitch * m_height);	
			KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);
			// ***********


			texBuf->unsetDirtyFlag();
		}
	//}
}

void KernelDevice::executeKernelJob( float p_dt, KernelJob p_jobId )
{
	switch (p_jobId)
	{
	case J_RAYTRACEWORLD:
		{
			cb.b += p_dt;

			RaytraceKernelData blob;
			blob.m_width=m_width; blob.m_height=m_height;
			blob.m_textureResource = m_gbufferHandle.m_textureResource;
			blob.m_textureLinearMem = m_gbufferHandle.m_textureLinearMem;
			blob.m_pitch = &m_gbufferHandle.m_pitch;
			blob.m_cb=&cb;

			m_raytracer->Execute((KernelData*)&blob,p_dt);
			break;
		}
		
	}
}


void KernelDevice::registerCanvas( void** p_texture )
{
	//unsigned int gbufSz = static_cast<unsigned int>(p_buffer.size());
	//for (unsigned int i=0;i<gbufSz;i++)
	//{
// 		m_gbufferHandle.m_texture.push_back((Texture*)p_buffer[i]);
// 		m_gbufferHandle.m_textureResource.push_back(NULL);
		m_gbufferHandle.m_texture = (Texture**)p_texture;
		
		Texture* texBuf = *m_gbufferHandle.m_texture/*[i]*/;
		cudaError_t res = cudaGraphicsD3D11RegisterResource( &(m_gbufferHandle.m_textureResource/*[i]*/),
															  texBuf->m_textureBuffer,
															  cudaGraphicsRegisterFlagsNone);
// 		if (i==0)
// 		{
			m_width=texBuf->m_width;
			m_height=texBuf->m_height;
		/*}*/
		
		if (!KernelHelper::assertCudaResult(res))
			throw KernelException(res,"Error registering GBuffer texture",__FILE__,__FUNCTION__,__LINE__);

		// ***********
// 		if (i==0)
// 		{
			res = cudaMallocPitch(&m_gbufferHandle.m_textureLinearMem, &m_gbufferHandle.m_pitch, m_width * sizeof(float) * 4, (size_t)m_height);
			if (!KernelHelper::assertCudaResult(res))
				throw KernelException(res,"Error registering GBuffer texture",__FILE__,__FUNCTION__,__LINE__);
			res = cudaMemset(m_gbufferHandle.m_textureLinearMem, 1, m_gbufferHandle.m_pitch * m_height);	
			if (!KernelHelper::assertCudaResult(res))
				throw KernelException(res,"Error registering GBuffer texture",__FILE__,__FUNCTION__,__LINE__);
		/*}*/
		
		// ***********

		texBuf->unsetDirtyFlag();
	//}

}


