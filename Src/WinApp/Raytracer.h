#ifndef RAYTRACER_CU
#define RAYTRACER_CU

#include <iostream> 
#include <vector> 
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "RaytraceConstantBuffer.h"
#include "KernelHelper.h"
#include "RaytraceHelper.h"
 
#pragma comment(lib, "cudart") 

 
using std::cerr; 
using std::cout; 
using std::endl; 
using std::exception; 
using std::vector; 

// =======================================================================================
//                                   Raytracer
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	The main loop for the raytracer, here a color is determined for the pixel
///        
/// # Raytracer
/// 
/// 24-4-2013 Jarl Larsson
///---------------------------------------------------------------------------------------


__device__ void Raytrace(float* p_outPixel, const int p_x, const int p_y)
{
	float4 currentColor;
	// Main raytrace loop
	do
	{
		currentColor = (float4)(0.0f,0.0f,0.0f,0.0f);
		IntersectAll(&scene,&ray,&intersection,false);			// Do the intersection tests
		
		if (intersection.dist >= 0.0f && intersection.dist<MAX_INTERSECT_DIST)
		{

			// finalColor=intersection.color*Lambert(&light,&intersection);
			dat.diffuseColor = (float4)(0.0f,0.0f,0.0f,0.0f);
			dat.specularColor = (float4)(0.0f,0.0f,0.0f,0.0f);
			float4 ambient=(float4)(0.3f, 0.3f, 0.6f,0.0f); // ambient
			// float4 ambient=(float4)(0.0f, 0.0f, 0.0f,0.0f); // ambient

			currentColor=intersection.surface.diffuse*ambient; // ambient base add (note: on do this on current colour for ambient on shadows)

			// add all lights
			for (int i=0;i<AMOUNTOFLIGHTS;i++)
			{				

				lightColor = (float4)(0.0f,0.0f,0.0f,0.0f);		
				BlinnPhong(&dat,&(scene.light[i]),&viewFrameDir,&intersection);
				lightColor+=intersection.surface.diffuse*dat.diffuseColor;
				lightColor+=intersection.surface.specular*dat.specularColor;
				// second intersection test for shadows, return true on very first hit
				intersection.dist = MAX_INTERSECT_DIST; // reset
			

				/*shadowRay.origin = intersection.pos;

				shadowRay.dir = fast_normalize( scene.light[i].vec - intersection.pos*scene.light[i].vec.w );


				if (!IntersectAll(&scene,&shadowRay,&intersection,true)) // only add light colour if no object is between current pixel and light*/
					currentColor += lightColor;


			}


			// Add color of this pixel(modified by previous pixel's reflection value) to final
			if (depth==0)
				finalColor = currentColor;
			else
				finalColor += currentColor * reflectionfactor; 



			reflectionfactor = intersection.surface.reflection;
			if (reflectionfactor>0.01f)
			{
				ray.origin = intersection.pos;
				reflect(&(ray.dir),&(intersection.normal),&(ray.dir));
				fast_normalize(ray.dir);
			}

			intersection.dist = MAX_INTERSECT_DIST;

			depth++;

		}
		else
			depth=max_depth;
		
	}  while (reflectionfactor>0.01f && depth<max_depth);

	// Set the color
	p_outPixel[R_CH] = (float)blockIdx.x/(float)gridDim.x; // red
    p_outPixel[G_CH] = (float)blockIdx.y/(float)gridDim.y; // green
    p_outPixel[B_CH] = 1.0f; // blue
    p_outPixel[A_CH] = 1; // alpha
}

#endif