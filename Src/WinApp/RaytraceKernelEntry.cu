#ifndef RAYTRACEKERNELENTRY_CU
#define RAYTRACEKERNELENTRY_CU

// =======================================================================================
//                                  RaytraceKernelEntry
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Entry point from host to kernel(device)
///        
/// # RaytraceKernelEntry
/// 
/// 2012->2013 Jarl Larsson
///---------------------------------------------------------------------------------------

#include <vector> 
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "RaytraceConstantBuffer.h"
#include "KernelHelper.h"

// device specific
#include "DeviceResources.h"
#include "Raytracer.h"
#include <RawTexture.h>
 
#pragma comment(lib, "cudart") 

 
using std::vector; 




__global__ void RaytraceKernel(unsigned char *p_outSurface, 
							   const int p_width, const int p_height, const size_t p_pitch,
							   float3* p_verts,float3* p_uvs,float3* p_norms,unsigned int p_numVerts,
							   unsigned int* p_indices,unsigned int p_numIndices,
							   float3 p_kdExtents, float3 p_kdPos,
							   TriPart* p_tris, unsigned int p_numTris,
							   DKDNode* p_nodes, DKDLeaf* p_leaflist, unsigned int* p_nodeIndices,
							   unsigned int p_numNodes,unsigned int p_numLeaves,unsigned int p_numNodeIndices)
{
    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;
    float *pixel;

    // discard pixel if outside
    if (x >= p_width || y >= p_height) return;

    // get a pointer to the pixel at (x,y)
    pixel = (float *)(p_outSurface + y*p_pitch) + 4*x;

	Raytrace(pixel,x,y, p_width, p_height, 
			 p_verts,p_uvs,p_norms,p_numVerts,
			 p_indices,p_numIndices,
			 p_kdExtents,p_kdPos,
			 p_tris, p_numTris,
			 p_nodes, p_leaflist, p_nodeIndices,
			 p_numNodes,p_numLeaves,p_numNodeIndices);
}
 
// Executes CUDA kernel 
extern "C" void RunRaytraceKernel(void* p_cb,void *surface,
			int width, int height, int pitch,
			void* p_verts,void* p_uvs,void* p_norms,unsigned int p_numVerts,
			void* p_indices,unsigned int p_numIndices,
			RawTexture* p_texture,
			void* p_kdExtents, void* p_kdPos,
			void* p_tris, unsigned int p_numTris,
			void* p_nodes, void* p_leaflist, void* p_nodeIndices,
			unsigned int p_numNodes,unsigned int p_numLeaves,unsigned int p_numNodeIndices) 
{ 
	// copy to constant buffer
	cudaError_t res = cudaMemcpyToSymbol(cb, p_cb, sizeof(RaytraceConstantBuffer));
	KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);

	// Allocate texture
	//float4* input;
	int ww=656, hh=480;
    // input = new float4[ww*hh];
    // for(int i = 0; i < ww*hh; i++)
    // {
	//  	// r
    //     input[i].x = /*(unsigned char)(256.0f**/(float)i/(float)(ww*hh)/*)*/;
	//  	// g
	//  	input[i].y = /*(unsigned char)(256.0f*(*/1.0f-((float)i/(float)(ww*hh))/*)*/;
	//  	// b
	//  	input[i].z = 128;
	//  	// a
	//  	input[i].w = 0;
    // }

	float4* texinput=(float4*)p_texture->m_data;
	ww=p_texture->m_width;
	hh=p_texture->m_height;


	// Allocate array and copy image data
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
		//cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
		//cudaCreateChannelDesc<uchar4>();
        //cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray *cuArray;
    res=cudaMallocArray(&cuArray,
                                    &channelDesc,
                                    ww,
                                    hh);
	KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);
	int s=sizeof(float4);
    res=cudaMemcpyToArray(cuArray,
                                      0,
                                      0,
                                      texinput,
                                      ww*hh*s,
                                      cudaMemcpyHostToDevice);
	KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);

    // Set texture parameters
    tex.addressMode[0] = cudaAddressModeWrap;
    tex.addressMode[1] = cudaAddressModeWrap;
    tex.filterMode = cudaFilterModeLinear;
    tex.normalized = true;    // access with normalized texture coordinates

    // Bind the array to the texture
    res=cudaBindTextureToArray(&tex, cuArray, &channelDesc);
	KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);

	// Set up dimensions
	int bd=min(16,max(8,width/40));
	if (bd<14) 
		bd=8;
	else
		bd=16;

	dim3 Db = dim3(bd, bd );   // block dimensions are fixed to be 256 threads
    dim3 Dg = dim3((width+Db.x-1)/Db.x, (height+Db.y-1)/Db.y);

	//DEBUGPRINT(( ("\n"+toString(width)+" x "+toString(height)+" @ "+toString(1000*reinterpret_cast<RaytraceConstantBuffer*>(p_cb)->b)).c_str() ));

    RaytraceKernel<<<Dg,Db>>>((unsigned char *)surface, width, height, pitch, 
							  (float3*)p_verts, (float3*) p_uvs, (float3*)p_norms,p_numVerts,
							  (unsigned int*)p_indices, p_numIndices,
							  *((float3*)p_kdExtents),*((float3*)p_kdPos),
							  (TriPart*)p_tris,p_numTris,
							  (DKDNode*)p_nodes, (DKDLeaf*)p_leaflist, (unsigned int*)p_nodeIndices,
							   p_numNodes,p_numLeaves,p_numNodeIndices);

	res = cudaDeviceSynchronize();
	KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);

	res=cudaUnbindTexture(&tex);
	KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);
    res=cudaFreeArray(cuArray);
	KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);
	//delete [] texinput;

} 

#endif