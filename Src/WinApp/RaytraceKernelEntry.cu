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
#include "Raytracer.h"
 
#pragma comment(lib, "cudart") 

 
using std::vector; 




texture<float, 2, cudaReadModeElementType> texRef;

__global__ void RaytraceKernel(unsigned char *p_outSurface, 
							   const int p_width, const int p_height, const size_t p_pitch,
							   float3* p_verts,float3* p_norms,unsigned int p_numVerts,
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
			 p_verts,p_norms,p_numVerts,
			 p_indices,p_numIndices,
			 p_kdExtents,p_kdPos,
			 p_tris, p_numTris,
			 p_nodes, p_leaflist, p_nodeIndices,
			 p_numNodes,p_numLeaves,p_numNodeIndices);
}
 
// Executes CUDA kernel 
extern "C" void RunRaytraceKernel(void* p_cb,void *surface,
			int width, int height, int pitch,
			void* p_verts,void* p_norms,unsigned int p_numVerts,
			void* p_indices,unsigned int p_numIndices,
			void* p_kdExtents, void* p_kdPos,
			void* p_tris, unsigned int p_numTris,
			void* p_nodes, void* p_leaflist, void* p_nodeIndices,
			unsigned int p_numNodes,unsigned int p_numLeaves,unsigned int p_numNodeIndices) 
{ 
	// copy to constant buffer
	cudaError_t res = cudaMemcpyToSymbol(cb, p_cb, sizeof(RaytraceConstantBuffer));
	KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);

	// Set up dimensions
	dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
    dim3 Dg = dim3((width+Db.x-1)/Db.x, (height+Db.y-1)/Db.y);

	//DEBUGPRINT(( ("\n"+toString(width)+" x "+toString(height)+" @ "+toString(1000*reinterpret_cast<RaytraceConstantBuffer*>(p_cb)->b)).c_str() ));

    RaytraceKernel<<<Dg,Db>>>((unsigned char *)surface, width, height, pitch, 
							  (float3*)p_verts, (float3*)p_norms,p_numVerts,
							  (unsigned int*)p_indices, p_numIndices,
							  *((float3*)p_kdExtents),*((float3*)p_kdPos),
							  (TriPart*)p_tris,p_numTris,
							  (DKDNode*)p_nodes, (DKDLeaf*)p_leaflist, (unsigned int*)p_nodeIndices,
							   p_numNodes,p_numLeaves,p_numNodeIndices);

	res = cudaDeviceSynchronize();
	KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);

} 

#endif