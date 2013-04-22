#include "KernelDevice.h"
#include "KernelException.h"

#include <cuda.h>
#include <builtin_types.h>
#include <cuda_runtime_api.h>
#include <cuda_d3d11_interop.h>
#include <DebugPrint.h>


KernelDevice::KernelDevice( void* p_device )
{
	m_width=1; m_height=1;
	m_device=(ID3D11Device*)p_device;
	cudaError_t res = cudaD3D11SetDirect3DDevice(m_device);
	if (!assertCudaResult(res))
		throw KernelException(res,"Error registering d3d-device",__FILE__,__FUNCTION__,__LINE__);

	m_raytracer = new RaytraceKernel();
}

KernelDevice::~KernelDevice()
{
	unsigned int gbufSz = m_gbufferHandle.m_texture.size();
	for (unsigned int i=0;i<m_gbufferHandle.m_texture.size();i++)
	{
		cudaError_t res = cudaGraphicsUnregisterResource( m_gbufferHandle.m_textureResource[i]);
		if (!assertCudaResult(res))
			DEBUGPRINT(( string("\n"+string(cudaGetErrorString(res))).c_str() ));
	}

	delete m_raytracer;
}

void KernelDevice::update( float dt )
{

	// Handle change in textures
	unsigned int gbufSz = m_gbufferHandle.m_texture.size();
	for (unsigned int i=0;i<gbufSz;i++)
	{
		Texture* texBuf = m_gbufferHandle.m_texture[i];
		if (texBuf->isDirty())
		{
			if (i==0)
			{
				m_width=texBuf->m_width;
				m_height=texBuf->m_height;
			}

			cudaError_t res = cudaGraphicsD3D11RegisterResource( &(m_gbufferHandle.m_textureResource[i]),
																 m_gbufferHandle.m_texture[i]->m_textureBuffer,
																 cudaGraphicsRegisterFlagsNone);
			if (!assertCudaResult(res))
				DEBUGPRINT(( string("\n"+string(cudaGetErrorString(res))).c_str() ));
			texBuf->unsetDirtyFlag();
		}
	}
}

void KernelDevice::executeKernelJob( float p_dt, KernelJob p_jobId )
{
	switch (p_jobId)
	{
	case J_RAYTRACEWORLD:
		{
			RaytraceConstantBuffer cb;
			RaytraceKernelData blob;
			blob.m_width=m_width; blob.m_height=m_height;
			blob.m_textureResource = &m_gbufferHandle.m_textureResource;
			blob.m_cb=&cb;

			m_raytracer->Execute((KernelData*)&cb,p_dt);
			break;
		}
		
	}
}

bool KernelDevice::assertCudaResult( cudaError_t p_result )
{
	return p_result==cudaSuccess;
}

void KernelDevice::registerGBuffer( vector<void*> p_buffer )
{
	unsigned int gbufSz = m_gbufferHandle.m_texture.size();
	for (unsigned int i=0;i<gbufSz;i++)
	{


		m_gbufferHandle.m_texture.push_back((Texture*)p_buffer[i]);
		m_gbufferHandle.m_textureResource.push_back(NULL);
		Texture* texBuf = m_gbufferHandle.m_texture[i];
		cudaError_t res = cudaGraphicsD3D11RegisterResource( &(m_gbufferHandle.m_textureResource[i]),
															  texBuf->m_textureBuffer,
															  cudaGraphicsRegisterFlagsNone);
		if (i==0)
		{
			m_width=texBuf->m_width;
			m_height=texBuf->m_height;
		}
		
		if (!assertCudaResult(res))
			throw KernelException(res,"Error registering GBuffer texture["+toString(i)+"]",__FILE__,__FUNCTION__,__LINE__);
		texBuf->unsetDirtyFlag();
	}

}


