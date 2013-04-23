#include "RaytraceKernel.h"
#include <DebugPrint.h>
#include <ToString.h>


extern "C"
{
		void RunCubeKernel(void* p_cb,void* colorArray,
			int width, int height, int pitch);
}

RaytraceKernel::RaytraceKernel() : IKernelHandler()
{

}

RaytraceKernel::~RaytraceKernel()
{

}

void RaytraceKernel::SetPerKernelArgs()
{

}

void RaytraceKernel::Execute( KernelData* p_data, float p_dt )
{

	RaytraceKernelData* blob = static_cast<RaytraceKernelData*>(p_data);
	RaytraceConstantBuffer* constantBuffer = blob->m_cb;
	int width = blob->m_width;
	int height = blob->m_height;
	size_t pitch = *blob->m_pitch;

	// OLD TEST
	// __________________________________________________
// 	vector<float> data(96); 
// 	for (int i = 0; i < 96; ++i) 
// 	{ 
// 		data[i] = static_cast<float>(i); 
// 	} 
// 
// 	// Compute cube on the device 
// 	vector<float> cube(96); 
// 
// 	const size_t size = 96 * sizeof(float); 
// 
// 	// TODO: test for error 
// 	float* d; 
// 	float* r; 
// 	cudaError hr; 
// 
// 	hr = cudaMalloc(reinterpret_cast<void**>(&d), size);            // Could return 46 if device is unavailable. 
// 	if (hr == cudaErrorDevicesUnavailable) 
// 	{ 
// 		//cerr << "Close all browsers and rerun" << endl; 
// 		throw std::runtime_error("Close all browsers and rerun"); 
// 	} 
// 
// 	hr = cudaMalloc(reinterpret_cast<void**>(&r), size); 
// 	if (hr == cudaErrorDevicesUnavailable) 
// 	{ 
// 		//cerr << "Close all browsers and rerun" << endl; 
// 		throw std::runtime_error("Close all browsers and rerun"); 
// 	} 
// 
// 	// Copy data to the device 
// 	cudaMemcpy(d, &data[0], size, cudaMemcpyHostToDevice); 

	// __________________________________________________

	// Map render textures
	cudaStream_t stream = 0;
	const int resourceCount = 1; // only use color for now
		//(int)blob->m_textureResource->size();
	cudaError_t res = cudaGraphicsMapResources(resourceCount, &blob->m_textureResource/*[0]*/, stream);
	KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);
	// Get pointers
	res = cudaGraphicsSubResourceGetMappedArray(&blob->m_textureView, blob->m_textureResource/*[0]*/, 0, 0);
	KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);

	RunCubeKernel(reinterpret_cast<void*>(constantBuffer),blob->m_textureLinearMem,
		width,height,(int)pitch); 


	// copy color array to texture (device->device)
	res = cudaMemcpy2DToArray(
		blob->m_textureView, // dst array
		0, 0,    // offset
		blob->m_textureLinearMem, pitch,       // src
		width*4*sizeof(float), height, // extent
		cudaMemcpyDeviceToDevice); // kind
	KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);

	// Copy back to the host 
// 	res = cudaMemcpy(&cube[0], r, size, cudaMemcpyDeviceToHost); 
// 	KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);



	// unmap textures
	res = cudaGraphicsUnmapResources(resourceCount, &blob->m_textureResource/*[0]*/, stream);
	KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);

	// Free device memory 
// 	cudaFree(d); 
// 	cudaFree(r); 

	// Print out results 
// 	DEBUGPRINT(("\n\nCube kernel results:\n")); 
// 
// 	for (int i = 0; i < 96; ++i) 
// 	{ 
// 		DEBUGPRINT(( string(toString(cube[i])+" ").c_str() )); 
// 	} 
}

