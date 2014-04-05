#ifndef DEVICE_RAYTRACER_RESOURCES
#define DEVICE_RAYTRACER_RESOURCES

__device__ __constant__ RaytraceConstantBuffer cb[1];


texture<float, 2, cudaReadModeElementType> tex;

#endif