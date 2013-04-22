#include <iostream> 
#include <vector> 
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
 
#pragma comment(lib, "cudart") 

 
using std::cerr; 
using std::cout; 
using std::endl; 
using std::exception; 
using std::vector; 
 
static const int MaxSize = 96; 
 
// CUDA kernel: cubes each array value 
__global__ void cubeKernel(float* result, float* data) 
{ 
    int idx = threadIdx.x; 
    float f = data[idx]; 
    result[idx] = f * f * f; 
} 
 
 
// Executes CUDA kernel 
extern "C" void RunCubeKernel(float* dataDev, float* resDev) 
{ 

 
    // Launch kernel: 1 block, 96 threads 
    // Important: Do not exceed number of threads returned by the device query, 1024 on my computer. 
    cubeKernel<<<1, MaxSize>>>(resDev,dataDev); 
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