#pragma once
#include "KernelRaytraceCommon.h"
#include "KernelRaytraceLight.h"



__device__ bool IntersectSphere(const Sphere* in_sphere, const Ray* in_ray, Intersection* inout_intersection, bool storeResult)
{
	// sphere intersection
	bool result = false;
	float4 delta = in_sphere->pos - in_ray->origin;
	float B = dot(in_ray->dir, delta);
	float D = B*B - dot(delta, delta) + in_sphere->rad * in_sphere->rad; 
	if (D >= 0.0f) 
	{
		float t0 = B - sqrt(D); 
		float t1 = B + sqrt(D);
		if ((t0 > 0.001f) && (t0 < inout_intersection->dist)) 
		{
			if (storeResult)
			{
				inout_intersection->dist = t0;
				inout_intersection->surface=in_sphere->mat;
				inout_intersection->pos=in_ray->origin+t0*in_ray->dir;
				inout_intersection->normal=(inout_intersection->pos - in_sphere->pos)/in_sphere->rad;
			}
			result = true;
		} 
		if ((t1 > 0.001f) && (t1 < inout_intersection->dist)) 
		{
			if (storeResult)
			{
				inout_intersection->dist = t1; 
				inout_intersection->surface=in_sphere->mat;
				inout_intersection->pos=in_ray->origin+t1*in_ray->dir;
				inout_intersection->normal=(inout_intersection->pos - in_sphere->pos)/in_sphere->rad;
			}
			result = true;
		}
	}
	return result;
}


// small difference, crazy results
__device__ bool IntersectSphereCRAZYDREAMVERSION(const Sphere* in_sphere, const Ray* in_ray, Intersection* inout_intersection, bool storeResult)
{
	// sphere intersection
	bool result = false;
	float4 delta = in_sphere->pos - in_ray->origin;
	float B = dot(in_ray->dir, delta);
	float D = B*B - dot(delta, delta) + in_sphere->rad * in_sphere->rad; 
	if (D >= 0.0f) 
	{
		float t0 = B*B - D; 
		float t1 = B*B + D;
		if ((t0 > 0.001f) && (t0 < inout_intersection->dist)) 
		{
			if (storeResult)
			{
				inout_intersection->dist = t0;
				inout_intersection->surface=in_sphere->mat;
				inout_intersection->pos=in_ray->origin+t0*in_ray->dir;
				inout_intersection->normal=(inout_intersection->pos - in_sphere->pos)/in_sphere->rad;
			}
			result = true;
		} 
		if ((t1 > 0.001f) && (t1 < inout_intersection->dist)) 
		{
			if (storeResult)
			{
				inout_intersection->dist = t1; 
				inout_intersection->surface=in_sphere->mat;
				inout_intersection->pos=in_ray->origin+t1*in_ray->dir;
				inout_intersection->normal=(inout_intersection->pos - in_sphere->pos)/in_sphere->rad;
			}
			result = true;
		}
	}
	return result;
}


// Distance estimator for a field of spheres
__device__ float SphereFieldDE(float4 v)
{
	float3 vv = (float3)(v.x,v.y,v.z);
	
	float vxS = v.x/fabs(v.x);
	float vzS = v.z/fabs(v.z);
	vv.xz = fmod(vv.xz, 1.0f) - (float2)(vxS*0.5f,vzS*0.5f); // instance on xz-plane

	return fast_length(vv)-0.3f;							 // sphere DE
}





// Distance estimator for a space of spheres
// also returns unique index of sphere
__device__ float SphereSpaceDE(float4 in_v, float3* out_idx)
{	
	float localVolume = 400.0f; // the local space containing the sphere
	float localVolumeH = localVolume*0.5f;

	// for offset, shift ray here -------------------------v
	float3 vv = (float3)(in_v.x,in_v.y,in_v.z)/* - (float3)(sin(in_v.x/localVolume)*30.0f,0,0)*/;
	*out_idx = vv/localVolumeH;
	(*out_idx).x = floor((*out_idx).x+0.5f);
	(*out_idx).y = floor((*out_idx).y+0.5f);
	(*out_idx).z = floor((*out_idx).z+0.5f);

	float vxS = in_v.x/fabs(in_v.x);
	float vyS = in_v.y/fabs(in_v.y);
	float vzS = in_v.z/fabs(in_v.z);

	vv = fmod(vv, localVolume) - (float3)(vxS*localVolumeH,vyS*localVolumeH,vzS*localVolumeH); // instance in xyz-volume
	
	/*if (sin((*out_idx).x)>0.0f && sin((*out_idx).x)<1.0f
		&& sin((*out_idx).y)>0.0f && sin((*out_idx).y)<1.0f
		&& sin((*out_idx).z)>0.0f && sin((*out_idx).z)<1.0f)   // Culling can be done here, an example would be sampling against a 3d noise texture for a "cloud effect"
		return fabs(fast_length(vv)+40.0f); // distance to next
	else*/
	   return fabs(fast_length(vv)-20.0f);		  // sphere DE
}

__device__ float RecursiveTetraDE1(float4 in_v,float3 in_pos)
{
	float3 z = (float3)(in_v.x,in_v.y,in_v.z)-in_pos;
	float Scale = 2.0f;
	float3 a1 = (float3)(1,1,1);
	float3 a2 = (float3)(-1,-1,1);
	float3 a3 = (float3)(1,-1,-1);
	float3 a4 = (float3)(-1,1,-1);
	float3 c;
	int n = 0;
	float dist, d;
	while (n < 8) {
		c = a1; dist = length(z-a1);
		d = length(z-a2); if (d < dist) { c = a2; dist=d; }
		d = length(z-a3); if (d < dist) { c = a3; dist=d; }
		d = length(z-a4); if (d < dist) { c = a4; dist=d; }
		z = Scale*z-c*(Scale-1.0);
		n++;
	}

	return fast_length(z) * pow(Scale, -(float)(n));
}

__device__ float RecursiveTetraDE2(float4 in_v,float3 in_pos)
{
	float3 z = (float3)(in_v.x,in_v.y,in_v.z)-in_pos;
    float r;
    int n = 0;
	float Scale = 2.0f;
	float Offset = 1.0f;
    while (n < 10) {
       if(z.x+z.y<0) z.xy = -z.yx; // fold 1
       if(z.x+z.z<0) z.xz = -z.zx; // fold 2
       if(z.y+z.z<0) z.zy = -z.yz; // fold 3	
       z = z*Scale - Offset*(Scale-1.0);
       n++;
    }
    return (fast_length(z) ) * pow(Scale, -(float)(n));
}

// ray marching
__device__ bool MarchSphere(const Sphere* in_sphere, const Ray* in_ray, Intersection* inout_intersection, bool storeResult)
{
	float totalDistance = 0.0;
	int steps;
	int maximumRaySteps = 50;
	float minimumDistance = 0.01f;
	float4 p;
	float3 sphereIdx = (float3)(0,0,0);
	for (steps=0; steps < maximumRaySteps; steps++) 
	{
		p = in_ray->origin + totalDistance * in_ray->dir;
		float distance = SphereSpaceDE(p,&sphereIdx);
		// float distance =  RecursiveTetraDE2(p,(float3)(400.0f,400.0f,400.0f));
		totalDistance += distance;
		if (distance < minimumDistance) break;
	}

	if (steps < maximumRaySteps && (totalDistance > 0.001f) && 
		(totalDistance < inout_intersection->dist))
	{
		if (storeResult)
		{
			inout_intersection->dist = totalDistance;
			inout_intersection->pos=p;
			inout_intersection->surface.diffuse = (float4)(1.0f+sin(sphereIdx.x)*0.5f,1.0f+sin(sphereIdx.y)*0.5f,1.0f+sin(sphereIdx.z)*0.5f,1.0f);
			// Run fractal
			// float3 offset = (float3)(p.x,p.y,p.z);
			float3 offset = (float3)(sphereIdx.x,sphereIdx.y,sphereIdx.z)*200.0f; // * localVolumeH
			
			totalDistance = 0.0;
			// float4 new_orig = p;
			for (steps=0; steps < maximumRaySteps; steps++) 
			{
				p = in_ray->origin + totalDistance * in_ray->dir;
				float distance = RecursiveTetraDE2(p,offset);
				totalDistance += distance;
				if (distance < minimumDistance) break;
			}
			
			if (steps < maximumRaySteps && (totalDistance > 0.001f) && 
				(totalDistance < inout_intersection->dist+400.0f))
			{
					maximumRaySteps = 1000;
					minimumDistance = totalDistance*0.5f;
					inout_intersection->dist = totalDistance; // distance

					inout_intersection->surface=in_sphere->mat; // material

					// for now, do super simple AO
					float c = (1.0f-(float)steps/(float)maximumRaySteps);
					inout_intersection->surface.diffuse = (float4)((1.0f+sin(sphereIdx.x))*0.5f,(1.0f+sin(sphereIdx.y))*0.5f,(1.0f+sin(sphereIdx.z))*0.5f,1.0f)-(float4)(c);
					// inout_intersection->surface.diffuse = (float4)(0.0f,1.0f,1.0f,0.0f);

					// store pos
					inout_intersection->pos=p;

					// Calculate the normal
					/*float4 xDir = (float4)(1.0f,0.0f,0.0f,0.0f)*0.05f;
					float4 yDir = (float4)(0.0f,1.0f,0.0f,0.0f)*0.05f;
					float4 zDir = (float4)(0.0f,0.0f,1.0f,0.0f)*0.05f;
					inout_intersection->normal=fast_normalize((float4)(SphereSpaceDE(p+xDir,&sphereIdx)-SphereSpaceDE(p-xDir,&sphereIdx),
																	   SphereSpaceDE(p+yDir,&sphereIdx)-SphereSpaceDE(p-yDir,&sphereIdx),
																	   SphereSpaceDE(p+zDir,&sphereIdx)-SphereSpaceDE(p-zDir,&sphereIdx),0.0f));*/
					float3 f = offset;
					float4 xDir = (float4)(1.0f,0.0f,0.0f,0.0f)*0.0001f;
					float4 yDir = (float4)(0.0f,1.0f,0.0f,0.0f)*0.0001f;
					float4 zDir = (float4)(0.0f,0.0f,1.0f,0.0f)*0.0001f;
					inout_intersection->normal=fast_normalize((float4)(RecursiveTetraDE2(p+xDir,f)-RecursiveTetraDE2(p-xDir,f),
																	   RecursiveTetraDE2(p+yDir,f)-RecursiveTetraDE2(p-yDir,f),
																	   RecursiveTetraDE2(p+zDir,f)-RecursiveTetraDE2(p-zDir,f),0.0f));
					// inout_intersection->normal = (float4)(0.0f,1.0f,0.0f,0.0f);
			}
		}

		return true;
	}
	

	return false;
}