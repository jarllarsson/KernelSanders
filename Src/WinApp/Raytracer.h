#ifndef RAYTRACER_H
#define RAYTRACER_H

//#define RENDER_STARRY_SKY

#include <iostream> 
#include <vector> 
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "RaytraceConstantBuffer.h"
#include "KernelHelper.h"
#include "KernelMathHelper.h"
#include "IntersectAll.h"
#include "RaytraceShadow.h"
#include "Scene.h"
#include "Ray.h"
#include "IntersectionInfo.h"
#include "DeviceKDStructures.h"
#include "IntersectKDTree.h"
#include "DeviceResources.h"
#include "RaytraceColourPalette.h"

 
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
						 const int p_width, const int p_height,
						 float3* p_verts,float3* p_uvs,float3* p_norms,unsigned int p_numVerts,
						 unsigned int* p_indices,unsigned int p_numIndices,
						 float3 p_kdExtents, float3 p_kdPos,
						 TriPart* p_tris, unsigned int p_numTris,
						 DKDNode* p_nodes, DKDLeaf* p_leaflist, unsigned int* p_nodeIndices,
						 unsigned int p_numNodes,unsigned int p_numLeaves,unsigned int p_numNodeIndices)
{	
	// Normalized device coordinates of pixel. (-1 to 1)
	const float u = (p_x / (float) p_width)*2.0f-1.0f;
	const float v = (p_y / (float) p_height)*2.0f-1.0f;
	const int tx = blockIdx.x*blockDim.x + threadIdx.x;
	const int ty = blockIdx.y*blockDim.y + threadIdx.y;

	// Store contents of constant buffer in local mem
	float time = cb[0].m_time;
	float rayDirScaleX = cb[0].m_rayDirScaleX;
	float rayDirScaleY = cb[0].m_rayDirScaleY;
	int drawMode=cb[0].m_drawMode;
	int shadowMode=cb[0].m_shadowMode;
	float partLit=1.0f/(float)shadowMode;
	float4  camPos = make_float4(cb[0].m_camPos);
	float4x4 camRotation = make_float4x4(cb[0].m_cameraRotationMat);
	unsigned int numVerts=min(0,MAXMESHLOCAL_VERTSBIN), 
				 numIndices=min(0,MAXMESHLOCAL_INDICESBIN), 
				 numTris=min(p_numTris,MAXTRIS);
	float3 kdExtents=p_kdExtents;
	float3 kdPos=p_kdPos;
	unsigned int numNodes=p_numNodes, numLeaves=p_numLeaves, numNodeIndices=p_numNodeIndices;
	// define a scene
	Scene scene;
	scene.time=time;

	scene.numIndices=0;
	scene.numVerts=0;

	scene.kdExtents=kdExtents;
	scene.kdPos=kdPos;
	scene.nodes=p_nodes;
	scene.leaflist=p_leaflist;
	scene.nodeIndices=p_nodeIndices;
	scene.numNodes=p_numNodes;
	scene.numLeaves=p_numLeaves;
	scene.numNodeIndices=numNodeIndices;
	scene.verts=p_verts;
	scene.uvs=p_uvs;
	scene.norms=p_norms;

#pragma region unusedshared
	// Copy mesh data to local memory
	//if (numNodes>0 && numLeaves>0 && numIndices>2)
	//{
	//	extern __shared__ DKDNode		snodes[];
	//	extern __shared__ DKDLeaf		sleaves[];
	//	extern __shared__ unsigned int	snindices[];
	//	extern __shared__ float3		sverts[];
	//	extern __shared__ float3		suvs[];
	//	extern __shared__ float3		snorms[];
	//
	//	snodes[];
	//	sleaves[];
	//	snindices[];
	//	sverts[];
	//	suvs[];
	//	snorms[];
	//
	//	scene.nodes=snodes;
	//	scene.leaflist=sleaves;
	//	scene.nodeIndices=snindices;
	//	scene.verts=sverts;
	//	scene.uvs=suvs;
	//	scene.norms=snorms;
	//	//for (unsigned int i=0;i<numNodes;i++)
	//	//{
	//	//	scene.nodes[i]=p_nodes[i];
	//	//}
	//	//for (unsigned int i=0;i<numLeaves;i++)
	//	//{
	//	//	scene.leaflist[i]=p_leaflist[i];
	//	//}
	//	//for (unsigned int i=0;i<numNodeIndices;i++)
	//	//{
	//	//	scene.nodeIndices[i]=p_nodeIndices[i];
	//	//}
	//	//for (unsigned int i=0;i<numVerts;i++)
	//	//{
	//	//	scene.verts[i]=p_verts[i];
	//	//}
	//	//for (unsigned int i=0;i<numVerts;i++)
	//	//{
	//	//	scene.norms[i]=p_norms[i];
	//	//}
	//	//for (unsigned int i=0;i<numVerts;i++)
	//	//{
	//	//	scene.uvs[i]=p_uvs[i];
	//	//}
	//}
#pragma endregion
	// =======================================================
	//                   TEST SETUP CODE
	// =======================================================

	// 1. Create ray
	// calculate eye ray in world space
	Ray ray;
	//ray.origin = make_float4(u,v,10.0f,1.0f);
	ray.origin = camPos;

	//ray.origin = camPos;   

	float4 viewFrameDir = cu_normalize( make_float4(u*rayDirScaleX, -v*rayDirScaleY, 1.0f,0.0f) );
	//ray.dir = make_float4(0.0f,0.0f,-1.0f,0.0f);
	ray.dir = viewFrameDir;
	mat4mul(&camRotation,&viewFrameDir, &ray.dir); // transform viewFrameDir with the viewMatrix to get the world space ray
	Ray shadowRay;	
	float spacefade=max(0.0f,1.0f-abs(ray.origin.y)/100.0f); // for fading stuff into space when traveling upwards

	// define some spheres

	//scene.sphere[0].pos = make_float4(17.0f,15.0f,0.0f,1.0f);
	//scene.sphere[0].rad = 10.5f;
	//scene.sphere[0].mat.diffuse = make_float4(0.5f, 0.79f, 0.22f,1.0f);
	//scene.sphere[0].mat.specular = make_float4(1.0f, 1.0f, 1.0f,500.0f);
	//scene.sphere[0].mat.reflection = 0.5f;
	//
	//scene.sphere[1].pos = make_float4(15.0f,15.0f,0.0f,1.0f);
	//scene.sphere[1].rad = 10.6f;
	//scene.sphere[1].mat.diffuse = make_float4(0.0f, 1.0f, 0.0f,1.0f);
	//scene.sphere[1].mat.specular = make_float4(0.0f, 0.0f, 0.0f,0.0f);
	//scene.sphere[1].mat.reflection = 0.5f;

	int ii=0;float d=4.0f,rad=0.3f;
	for (int i=0;i<MAXSPHERES;i++)
	{	
		scene.sphere[i].pos = make_float4(sin((float)ii)*d,rad,-cos((float)ii)*d,1.0f);
		scene.sphere[i].rad = rad;
		float3 col=0.6f*colarr2[i*3];
		scene.sphere[i].mat.diffuse = make_float4(col.x,col.y,col.z,1.0f);
		scene.sphere[i].mat.specular = make_float4(0.1f, 0.1f, 0.1f,0.8f);
		scene.sphere[i].mat.reflection = 0.4f;
		ii++;
		if (ii>4)
		{
			ii=0; d=7.0f; rad=2.0f;
		}
		if (i>5) rad+=0.1f;
	}

	// define a plane
	for (int i=0;i<MAXPLANES;i++)
	{
		scene.plane[i].distance = 0.0f;
		scene.plane[i].normal = make_float4(0.0f,-1.0f,0.0f,0.0f);
		//scene.plane[i].mat.diffuse = (float4)( 71.0f/255.0f, 21.0f/255.0f, 87.0f/255.0f ,1.0f);
		scene.plane[i].mat.diffuse = make_float4( 0.15625f, 0.37641f, 0.3394f ,spacefade);
		scene.plane[i].mat.specular = 0.1f*make_float4(0.5f, 0.9f, 0.86f,0.0f);
		scene.plane[i].mat.reflection = 0.3f;
	}



	// define some tris
	scene.numTris=numTris;
	//for (unsigned int i=0;i<numTris;i++)
	//{
	//
	//	#pragma unroll 3
	//	for (int x=0;x<3;x++)
	//	{
	//		scene.tri[i].vertices[x] = p_tris[i].vertices[x];
	//			//make_float3((float)i+x*0.5f, sin(time+(float)i+x*0.01f) + ((i%2)*2-1)*(float)(x%2)*0.5f, sin((float)(x+i)*0.5f)*-3.0f);
	//	}
	//
	//	scene.tri[i].mat.diffuse = make_float4( 1.0f-((float)i/(float)MAXTRIS), (float)i/(float)MAXTRIS, 1.0f-((float)i/(float)(MAXTRIS*0.2f)) ,1.0f);
	//	scene.tri[i].mat.specular = make_float4(1.0f, 1.0f, 1.0f,0.5f);
	//	scene.tri[i].mat.reflection = 0.0f;
	//}





	// define some boxes
	d+=4.0f; rad=5.0f;
	for (int i=0;i<MAXBOXES;i++)
	{
		float sini=sin((float)i),cosi=cos((float)i);
		scene.box[i].pos = make_float4(sini*d,rad*4.0f,-cosi*d,1.0f);
			/*make_float4(-5.0f,10+sin((float)i)*10.0f*sin(time), i*10,0.0f) + make_float4(sin((float)i)*50.0f*(1.0f+sin(time)),
			5.0f+sin(time*0.5f)*5.0f,
			cos((float)i)*50.0f*(1.0f+sin(time)),
			0.0f);*/
		// float4 tesst = (float4)(1.0f,0.0f,0.0f,0.0f);
		scene.box[i].sides[0] = make_float4(cosi,0.0f,sini,0.0f);  // x
		scene.box[i].sides[1] = make_float4(0.0f,1.0f,0.0f,0.0f);  // y
		scene.box[i].sides[2] = make_float4(-sini,0.0f,cosi,0.0f);  // z
		// mat4mul(viewMatrix,&tesst, &box[i].sides[0]);
		scene.box[i].hlengths[0] = rad*0.5f;
		scene.box[i].hlengths[1] = rad*4.0f;
		scene.box[i].hlengths[2] = rad*0.5f;
		scene.box[i].mat.diffuse = make_float4(colarr2[13],1.0f);
			//make_float4( (float)(i%5)*0.5f, 1.0f-sin((float)i), ((float)i/(float)(MAXBOXES*2.0f)) ,1.0f);
		scene.box[i].mat.specular = make_float4(0.1f, 0.2f, 0.0f,0.8f);
		scene.box[i].mat.reflection = 0.2f;
	}

	// define some lights
	
	for (int i=0;i<MAXLIGHTS-2;i++)
	{
		// scene.light[i].vec = (float4)(i*5.0f*sin((1.0f+i)*time),i+sin(time),100.0f*sin(time) + i*2.0f*cos((1.0f+i)*time),1.0f);
		scene.light[i].vec = make_float4(sin(i+time*0.3f)*2.0f*((float)(i+1)*0.15f),1.99f-cos(i+time*0.3f)*2.0f,(float)(i-2)*-cos(i+time*0.3f)*1.0f,1.0f);
		scene.light[i].diffusePower = 2000.0f;
		scene.light[i].specularPower = 0.001f;
		float3 col=colarr[i];
		scene.light[i].diffuseColor = make_float4(0.001f*col.x+0.0001f,0.001f*col.y,0.001f*col.z+0.0001f,1.0f);
		scene.light[i].specularColor = make_float4(1.0f,1.0f,1.0f,0.0f);
	}
	

	// Create a directional light
	scene.light[MAXLIGHTS-2].vec = cu_normalize(make_float4(1.0f,-1.0f,0.3f,0.0f));
		//cu_normalize(make_float4(sin(time),sin(time*0.5f),cos(time),0.0f));
	scene.light[MAXLIGHTS-2].diffusePower = 0.7f;
	scene.light[MAXLIGHTS-2].specularPower = 0.7f;
	scene.light[MAXLIGHTS-2].diffuseColor = make_float4(1.0f, 0.9f,0.7f,1.0f);
	scene.light[MAXLIGHTS-2].specularColor = make_float4(1.0f,1.0f,1.0f,0.0f);
	
	// Create a directional light
	scene.light[MAXLIGHTS-1].vec = cu_normalize(make_float4(1.0f,-0.1f,-0.3f,0.0f));
	//cu_normalize(make_float4(sin(time),sin(time*0.5f),cos(time),0.0f));
	scene.light[MAXLIGHTS-1].diffusePower = 0.5f;
	scene.light[MAXLIGHTS-1].specularPower = 0.2f;
	scene.light[MAXLIGHTS-1].diffuseColor = make_float4(0.5f, 0.7f,0.7f,1.0f);
	scene.light[MAXLIGHTS-1].specularColor = make_float4(1.0f,1.0f,1.0f,0.0f);


	// TRANSFORM

	// 2. Declare var for final color storage
// 	float4 finalColor = make_float4((1.0f+ray.dir.x),
// 		(1.0f+ray.dir.z),
// 		(1.0f+ray.dir.y),
// 		0.0f)*0.05f;
	float4 finalColor = make_float4(0.1f+(1.0f+ray.dir.z),
		0.5f+(1.0f+ray.dir.x)*1.5f,
		0.6f+(1.0f+ray.dir.y)*2.5f,
		0.0f)*0.15f;
	float3 bcol1=colarr2[12];
	float3 bcol2=colarr2[18];
	float3 bcol3=colarr2[28];
	float3 bcombo=bcol1*(1.0f+ray.dir.x)*0.5f+bcol2*(1.0f+ray.dir.y)*0.5f+bcol3*(1.0f+ray.dir.z)*0.5f;
	finalColor=make_float4(0.5f*bcombo,1.0f)*spacefade;

	Intersection intersection;
	intersection.dist = MAX_INTERSECT_DIST;
	intersection.surface.diffuse = make_float4(0.0f,0.0f,0.0f,0.0f);
	intersection.surface.specular = make_float4(0.0f,0.0f,0.0f,0.0f);
	intersection.surface.reflection= 0.0f;


	// Raytrace:
	float reflectionfactor = 0.0f;
	int max_depth = 1;
	int depth = 0;

	float4 currentColor = make_float4(0.0f,0.0f,0.0f,0.0f);
	SurfaceLightingData dat;
	float4 lightColor = make_float4(0.0f,0.0f,0.0f,0.0f);	
	float3 debugColor = make_float3(0.0f,0.0f,0.0f);

	// =======================================================

	// Main raytrace loop

	do
	{
		currentColor = make_float4(0.0f,0.0f,0.0f,1.0f);
		bool result=false;

		result = IntersectAll(&scene,&ray,&intersection,false,result,&debugColor);			// Do the intersection tests
		
		// If defined, render starry sky
#ifdef RENDER_STARRY_SKY
		result = MarchAll(&ray,&intersection,false,false);
#endif

		
		if (intersection.dist >= 0.0f && intersection.dist<MAX_INTERSECT_DIST)
		{

			// finalColor=intersection.color*Lambert(&light,&intersection);
			dat.diffuseColor = make_float4(0.0f,0.0f,0.0f,0.0f);
			dat.specularColor = make_float4(0.0f,0.0f,0.0f,0.0f);
			float4 ambient=make_float4(0.0f, 0.0f, 0.0f,0.0f); // ambient
			// float4 ambient=(float4)(0.0f, 0.0f, 0.0f,0.0f); // ambient

			currentColor=ambient; // ambient base add (note: on do this on current colour for ambient on shadows)

			// add all lights
			for (int i=0;i<MAXLIGHTS;i++)
			{				

				lightColor = make_float4(0.0f,0.0f,0.0f,0.0f);		
				BlinnPhong(&dat,&(scene.light[i]),&viewFrameDir,&intersection);
				lightColor+=intersection.surface.diffuse*dat.diffuseColor;
				lightColor+=intersection.surface.specular*dat.specularColor;

				// second intersection test for shadows, return true on very first hit
				float lightHits=0;
				if (shadowMode!=RAYTRACESHADOWMODE_OFF && scene.light[i].vec.w<0.5f ) // shadow
					lightHits=ShadowCastAll(&(scene.light[i]),shadowMode,
											&shadowRay,&intersection,&scene,u,v,partLit);
				else              // no shadow
					lightHits=1.0f;

				currentColor += lightColor*lightHits;
			}


			// Add color of this pixel(modified by previous pixel's reflection value) to final
			float alpha = intersection.surface.diffuse.w;
			float4 alphablend=finalColor*(1.0f-alpha)+currentColor*alpha;
			if (depth==0)
				finalColor = alphablend;
			else
				finalColor += alphablend * reflectionfactor; 

			// special fallof, to avoid reflections in plane for this particular scene
			// otherwise, this is unwanted and should be 1
			float rflfallof=max(0.0f,(1.0f-0.001f*squaredLen(intersection.pos)));
			
			// do reflection
			reflectionfactor = intersection.surface.reflection * rflfallof;
			if (reflectionfactor>0.01f)
			{
				ray.origin = intersection.pos;
				cu_reflect(ray.dir,intersection.normal,ray.dir);
				cu_normalize(ray.dir);
 			}

			intersection.dist = MAX_INTERSECT_DIST;

			depth++;

		}
		else
			depth=max_depth;
		
	}  while (reflectionfactor>0.01f && depth<max_depth);

	//float4 tCol=tex2D(tex, u, v);
	//float3 back=make_float3(tCol.x,tCol.y,tCol.z);

	// Set the color
	float showgrid=0.0f;
	if (drawMode==1) 
		showgrid=1.0f;
	else if (drawMode==2)
	{
		finalColor.x=debugColor.x;
		finalColor.y=debugColor.y;
		finalColor.z=debugColor.z;
	}
	float dbgGridX=showgrid*((float)blockIdx.x/(float)gridDim.x);
	float dbgGridY=showgrid*((float)blockIdx.y/(float)gridDim.y);
	p_outPixel[R_CH] = finalColor.x + dbgGridX; // red
	p_outPixel[G_CH] = finalColor.y + dbgGridY; // green
	p_outPixel[B_CH] = finalColor.z; // blue
	//p_outPixel[R_CH] = /*+back.x*/(debugColor.x) + dbgGridX; // red
	//p_outPixel[G_CH] = /*+back.y*/(debugColor.y) + dbgGridY; // green
	//p_outPixel[B_CH] = /*+back.z*/(debugColor.z); // blue
	p_outPixel[A_CH] = finalColor.w; // alpha
}

#endif