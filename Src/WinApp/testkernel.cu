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

// texture<float, 2, cudaReadModeElementType> colorTex;
// texture<float, 2, cudaReadModeElementType> normalTex;

// CUDA kernel: cubes each array value 
// __global__ void cubeKernel(float* result, float* data,float* surface) 
// { 
// 	int idx = threadIdx.x; 
// 	float f = data[idx]; 
// 	result[idx] = cb[0].b; 
// 
// 	//int x = blockIdx.x*blockDim.x + threadIdx.x;
//    // int y = blockIdx.y*blockDim.y + threadIdx.y;
//     //float *pixel;
// 	surface[idx]=1.0f;

    // in the case where, due to quantization into grids, we have
    // more threads than pixels, skip the threads which don't
    // correspond to valid pixels
   // if (x >= width || y >= height) return;

    // get a pointer to the pixel at (x,y)
// 	int y = 0;
// 	int x = idx;
// 	int pitch = 4*600;
//     pixel = (float *)(surface + y*pitch) + 4*x;
// 
//     // populate it
//     pixel[0] = 0.0f; // red
//     pixel[1] = 1.0f; // green
//     pixel[2] = 1.0f; // blue
//     pixel[3] = 1; // alpha
/*} */

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
    pixel[0] = 0.5*pixel[0] + 0.5*pow(value_x, 3.0f); // red
    pixel[1] = 0.5*pixel[1] + 0.5*pow(value_y, 3.0f); // green
    pixel[2] = 0.5f + 0.5f*cos(t); // blue
    pixel[3] = 1; // alpha
}
 
// Executes CUDA kernel 
extern "C" void RunCubeKernel(void* p_cb,unsigned char *surface,
			int width, int height, int pitch) 
{ 
	// copy to constant buffer
	cudaError_t res = cudaMemcpyToSymbol(cb, p_cb, sizeof(p_cb));
	KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);

	// bind array to texture
// 	cudaChannelFormatDesc colorTexFormat = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
//     res = cudaBindTextureToArray(colorTex, colorArray, colorTexFormat);
// 	KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);
 
	// Launch kernel: 1 block, 96 threads 
	// Important: Do not exceed number of threads returned by the device query, 1024 on my computer. 
	//cubeKernel<<<1, MaxSize>>>(resDev,dataDev,(float*)colorArray); 

	dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
    dim3 Dg = dim3((width+Db.x-1)/Db.x, (height+Db.y-1)/Db.y);

    cuda_kernel_texture_2d<<<Dg,Db>>>((unsigned char *)surface, width, height, pitch);

	//res = cudaUnbindTexture(colorTex);
	//KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);
} 
 
// Main entry into the program 
/*
int main(void) 
{ 
	cout << "In main." << endl; 
 
	// Create sample data 
	vector<float> data(MaxSize); 
	InitializeData(data); 
 
	// Compute cube on the device 
	vector<float> cube(MaxSize); 
	RunCubeKernel(data, cube); 
 
	// Print out results 
	cout << "Cube kernel results." << endl << endl; 
 
	for (int i = 0; i < MaxSize; ++i) 
	{ 
		cout << cube[i] << endl; 
	} 
 
	return 0; 
} 
*/