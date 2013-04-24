#include <iostream> 
#include <vector> 
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "RaytraceConstantBuffer.h"
#include "KernelHelper.h"
 
#pragma comment(lib, "cudart") 

 
using std::cerr; 
using std::cout; 
using std::endl; 
using std::exception; 
using std::vector; 
 
static const int MaxSize = 96; 



__constant__ RaytraceConstantBuffer cb[1];



texture<float, 2, cudaReadModeElementType> texRef;

__global__ void cuda_kernel_texture_2d(unsigned char *surface, int width, int height, size_t pitch)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    float *pixel;

    // in the case where, due to quantization into grids, we have
    // more threads than pixels, skip the threads which don't
    // correspond to valid pixels
    if (x >= width || y >= height) return;

    // get a pointer to the pixel at (x,y)
    pixel = (float *)(surface + y*pitch) + 4*x;

	float t=0.5f;

    // populate it
    float value_x = 0.5f + 0.5f*cos(t + 10.0f*((2.0f*x)/width  - 1.0f));
    float value_y = 0.5f + 0.5f*cos(t + 10.0f*((2.0f*y)/height - 1.0f));
    pixel[0] = (float)blockIdx.x/(float)gridDim.x /*0.1f+0.5*pixel[0] + 0.5*pow(value_x, 3.0f)*/; // red
    pixel[1] = (float)blockIdx.y/(float)gridDim.y /*0.1f+0.5*pixel[1] + 0.5*pow(value_y, 3.0f)*/; // green
    pixel[2] = 1.0f; // blue
    pixel[3] = 1; // alpha
}
 
// Executes CUDA kernel 
extern "C" void RunCubeKernel(void* p_cb,unsigned char *surface,
			int width, int height, int pitch) 
{ 
	// copy to constant buffer
	cudaError_t res = cudaMemcpyToSymbol(cb, p_cb, sizeof(p_cb));
	KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);

	// Set up dimensions
	dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
    dim3 Dg = dim3((width+Db.x-1)/Db.x, (height+Db.y-1)/Db.y);

	DEBUGPRINT(( ("\n"+toString(width)+" x "+toString(height)).c_str() ));

    cuda_kernel_texture_2d<<<Dg,Db>>>((unsigned char *)surface, width, height, pitch);

	res = cudaDeviceSynchronize();
	KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);

} 
