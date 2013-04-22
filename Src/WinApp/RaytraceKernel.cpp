#include "RaytraceKernel.h"
#include <DebugPrint.h>
#include <ToString.h>


extern "C" void RunCubeKernel(float* dataDev, float* resDev);

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

	vector<float> data(96); 
	for (int i = 0; i < 96; ++i) 
	{ 
		data[i] = static_cast<float>(i); 
	} 

	// Compute cube on the device 
	vector<float> cube(96); 

	const size_t size = 96 * sizeof(float); 

	// TODO: test for error 
	float* d; 
	float* r; 
	cudaError hr; 

	hr = cudaMalloc(reinterpret_cast<void**>(&d), size);            // Could return 46 if device is unavailable. 
	if (hr == cudaErrorDevicesUnavailable) 
	{ 
		//cerr << "Close all browsers and rerun" << endl; 
		throw std::runtime_error("Close all browsers and rerun"); 
	} 

	hr = cudaMalloc(reinterpret_cast<void**>(&r), size); 
	if (hr == cudaErrorDevicesUnavailable) 
	{ 
		//cerr << "Close all browsers and rerun" << endl; 
		throw std::runtime_error("Close all browsers and rerun"); 
	} 

	// Copy data to the device 
	cudaMemcpy(d, &data[0], size, cudaMemcpyHostToDevice); 

	RunCubeKernel(d, r); 

	// Copy back to the host 
	cudaMemcpy(&cube[0], r, size, cudaMemcpyDeviceToHost); 

	// Free device memory 
	cudaFree(d); 
	cudaFree(r); 

	// Print out results 
	DEBUGPRINT(("\n\nCube kernel results:\n")); 

	for (int i = 0; i < 96; ++i) 
	{ 
		DEBUGPRINT(( (string("CUDA: ")+toString(cube[i])).c_str() )); 
	} 
}

