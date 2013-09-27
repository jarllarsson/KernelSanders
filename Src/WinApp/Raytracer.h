#ifndef RAYTRACER_H
#define RAYTRACER_H

#include <iostream> 
#include <vector> 
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "RaytraceConstantBuffer.h"
#include "KernelHelper.h"
#include "KernelMathHelper.h"
#include "IntersectAll.h"
#include "Scene.h"
#include "Ray.h"
#include "IntersectionInfo.h"

 
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


__device__ void Raytrace(float* p_outPixel, const int p_x, const int p_y,
						 const int p_width, const int p_height)
{
	// Normalized device coordinates of pixel. (-1 to 1)
	const float u = (p_x / (float) p_width)*2.0f-1.0f;
	const float v = (p_y / (float) p_height)*2.0f-1.0f;
	// =======================================================
	//                   TEST SETUP CODE
	// =======================================================
	/*
	// define a scene
	Scene scene;

	// define some spheres
	scene.sphere[0].pos = (float4)(0.0f,0.0f,0.0f,1.0f);
	scene.sphere[0].rad = 0.1f;
	scene.sphere[0].mat.diffuse = (float4)(0.0f, 0.0f, 1.0f,1.0f);
	scene.sphere[0].mat.specular = (float4)(0.0f, 0.0f, 0.0f,0.0f);
	scene.sphere[0].mat.reflection = 0.0f;

	scene.sphere[1].pos = (float4)(2.0f,0.0f,0.0f,1.0f);
	scene.sphere[1].rad = 0.2f;
	scene.sphere[1].mat.diffuse = (float4)(0.0f, 1.0f, 0.0f,1.0f);
	scene.sphere[1].mat.specular = (float4)(1.0f, 0.0f, 0.0f,0.0f);
	scene.sphere[1].mat.reflection = 0.0f;

	for (int i=2;i<AMOUNTOFSPHERES;i++)
	{
		scene.sphere[i].pos = (float4)((float)(i%3),(float)i,(float)i,1.0f);
		scene.sphere[i].rad = i*0.1f;
		scene.sphere[i].mat.diffuse = (float4)( (float)i/(float)AMOUNTOFSPHERES, 1.0f-((float)i/(float)AMOUNTOFSPHERES), ((float)i/(float)(AMOUNTOFSPHERES*0.2f)) ,1.0f);
		scene.sphere[i].mat.specular = (float4)(1.0f, 1.0f, 1.0f,0.8f);
		scene.sphere[i].mat.reflection = (float)i/(float)AMOUNTOFSPHERES;
	}

	// define a plane
	for (int i=0;i<AMOUNTOFPLANES;i++)
	{
		scene.plane[i].distance = -1.0f;
		scene.plane[i].normal = (float4)(0.0f,1.0f,0.0f,0.0f);
		//scene.plane[i].mat.diffuse = (float4)( 71.0f/255.0f, 21.0f/255.0f, 87.0f/255.0f ,1.0f);
		scene.plane[i].mat.diffuse = (float4)( 1.0f, 1.0f, 1.0f ,1.0f);
		scene.plane[i].mat.specular = (float4)(0.1f, 0.1f, 0.1f,0.1f);
		scene.plane[i].mat.reflection = 0.0f;

	}



	// define some tris

	for (int i=0;i<AMOUNTOFTRIS;i++)
	{

#pragma unroll 3
		for (int x=0;x<3;x++)
		{

			scene.tri[i].vertices[x] = (float4)((float)i+x*0.5f, ((i%2)*2-1)*(float)(x%2)*0.5f, sin((float)(x+i)*0.5f)*-3.0f,0.0f);
		}

		scene.tri[i].mat.diffuse = (float4)( 1.0f-((float)i/(float)AMOUNTOFTRIS), (float)i/(float)AMOUNTOFTRIS, 1.0f-((float)i/(float)(AMOUNTOFTRIS*0.2f)) ,1.0f);
		scene.tri[i].mat.specular = (float4)(1.0f, 1.0f, 1.0f,0.5f);
		scene.tri[i].mat.reflection = 0.6f;

	}

	// define some boxes
	for (int i=0;i<AMOUNTOFBOXES;i++)
	{
		scene.box[i].pos = (float4)(-5.0f,10+sin((float)i)*10.0f*sin(time), i*10,0.0f) + (float4)(sin((float)i)*50.0f*(1.0f+sin(time)),
			5.0f+sin(time*0.5f)*5.0f,
			cos((float)i)*50.0f*(1.0f+sin(time)),
			0.0f);
		// float4 tesst = (float4)(1.0f,0.0f,0.0f,0.0f);
		scene.box[i].sides[0] = (float4)(1.0f,0.0f,0.0f,0.0f);  // x
		scene.box[i].sides[1] = (float4)(0.0f,1.0f,0.0f,0.0f);  // y
		scene.box[i].sides[2] = (float4)(0.0f,0.0f,1.0f,0.0f);  // z
		// mat4mul(viewMatrix,&tesst, &box[i].sides[0]);
		scene.box[i].hlengths[0] = (1+i);
		scene.box[i].hlengths[1] = (1+i);
		scene.box[i].hlengths[2] = (1+i);
		scene.box[i].mat.diffuse = (float4)( (float)(i%5)*0.5f, 1.0f-sin((float)i), ((float)i/(float)(AMOUNTOFBOXES*2.0f)) ,1.0f);
		scene.box[i].mat.specular = (float4)(0.1f, 0.1f, 0.1f,0.5f);
		scene.box[i].mat.reflection = 0.2f;
	}


	// define some lights
	for (int i=0;i<AMOUNTOFLIGHTS-1;i++)
	{
		// scene.light[i].vec = (float4)(i*5.0f*sin((1.0f+i)*time),i+sin(time),100.0f*sin(time) + i*2.0f*cos((1.0f+i)*time),1.0f);
		scene.light[i].vec = (float4)(sin((float)i)*20.0f*(1.0f+sin(time)),
			5.0f+sin(time*0.5f)*5.0f,
			cos((float)i)*20.0f*(1.0f+sin(time)),
			1.0f);
		scene.light[i].diffusePower = 2.0f*(5.0f+sin(time*0.5f)*5.0f);
		scene.light[i].specularPower = 10.0f;
		scene.light[i].diffuseColor = (float4)( ((float)i/(float)(AMOUNTOFBOXES*2.0f)), 1.0f-sin((float)i),(float)(i%5)*0.5f  ,1.0f);
		scene.light[i].specularColor = (float4)(1.0f,1.0f,1.0f,0.3f);

	}


	// Create a directional light
	scene.light[AMOUNTOFLIGHTS-1].vec = fast_normalize((float4)(0.0f,1.0f,1.0f,0.0f));
	scene.light[AMOUNTOFLIGHTS-1].diffusePower = 1.0f;
	scene.light[AMOUNTOFLIGHTS-1].specularPower = 0.5f;
	scene.light[AMOUNTOFLIGHTS-1].diffuseColor = (float4)(1.0f, 1.0f,1.0f,1.0f);
	scene.light[AMOUNTOFLIGHTS-1].specularColor = (float4)(1.0f,1.0f,1.0f,0.05f);
	*/

	// 1. Create ray
	// calculate eye ray in world space
	Ray ray;
	ray.origin = make_float4(u,v,10.0f,1.0f);
	//ray.origin = make_float4(0.0f,0.0f,0.0f,0.0f);

	//ray.origin = camPos;   

	float4 viewFrameDir = cu_normalize( make_float4(u, v, -1.3f,0.0f) );
	ray.dir = make_float4(0.0f,0.0f,-1.0f,0.0f);
	//ray.dir = viewFrameDir;
	//mat4mul_ignoreW(viewMatrix,&viewFrameDir, &ray.dir); // transform viewFrameDir with the viewMatrix to get the world space ray

	Ray shadowRay;

	// TRANSFORM

	// 2. Declare var for final color storage
	float4 finalColor = make_float4((1.0f+ray.dir.x)*0.5f,
		(1.0f+ray.dir.z)*0.5f,
		(1.0f+ray.dir.y)*0.5f,
		0.0f);

	Intersection intersection;
	intersection.dist = MAX_INTERSECT_DIST;
	intersection.surface.diffuse = make_float4(0.0f,0.0f,0.0f,0.0f);
	intersection.surface.specular = make_float4(0.0f,0.0f,0.0f,0.0f);
	intersection.surface.reflection= 0.0f;


	// Raytrace:
	float reflectionfactor = 0.0f;
	int max_depth = 4;
	int depth = 0;

	float4 currentColor;
	SurfaceLightingData dat;
	float4 lightColor = make_float4(0.0f,0.0f,0.0f,0.0f);	

	// =======================================================


	// Raytrace:
 
	// while (reflectionFactor>0 && depth<max_depth)
	// {

	// for (int i=0;i<amountOfObjects;i++)
	// {
	float dist = -1.0f;

	// sphere intersection proto
	float4 spherePos = make_float4(0.0f,0.0f,0.0f,1.0f);
	float sphereRad = 0.5f;
	float4 delta = spherePos - ray.origin;
	float B = cu_dot(ray.dir, delta);
	float D = B*B - cu_dot(delta, delta) + sphereRad * sphereRad; 
	if (D >= 0.0f) 
	{
		float t0 = B - sqrt(D); 
		float t1 = B + sqrt(D);
		if ((t0 > 0.001f) && (t0 < intersection.dist)) 
		{
			intersection.dist = t0;
		} 
		if ((t1 > 0.001f) && (t1 < intersection.dist)) 
		{
			intersection.dist = t1; 
		}
	}

	// general after each test
	// float dist = rayIntersect(ray,orig); // return -1 on miss
	/*if (dist>0.0f && (dist<intersectDistance || intersectDistance<0.0f))
	intersectDistance=dist;*/

	// }	// for each object		

	if (intersection.dist >= 0.0f && intersection.dist<MAX_INTERSECT_DIST)
	{
		finalColor=make_float4(0.0f,
			0.0f,
			1.0f,
			0.0f);
		// for (int i=0;i<amountOfLights;i++)
		// {
		// if (noObjectIsInFrontOfLight) // only add light colour if no object is between current pixel and light
		// currentColor += lightColor;
		// }
	}

	// Add color of this pixel(modified by previous pixel's reflection value) to final
	// float4 finalColor += currentColor * reflectionFactor; 

	// Modify for bounce with this reflection val 
	// reflectionFactor *= surfaceReflectionVal;

	// depth++;
	// intersectDistance = -1.0f;

	// }   // while


	// Main raytrace loop
#ifdef JHSDFJHGDSIUYTDSFKJDFS
	do
	{
		currentColor = make_float4(0.0f,0.0f,0.0f,0.0f);
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
#endif
	// Set the color
	p_outPixel[R_CH] = finalColor.x + (float)blockIdx.x/(float)gridDim.x; // red
    p_outPixel[G_CH] = finalColor.y + (float)blockIdx.y/(float)gridDim.y; // green
	p_outPixel[B_CH] = finalColor.z + 1.0f; // blue
	p_outPixel[A_CH] = finalColor.w + 1; // alpha
}

#endif