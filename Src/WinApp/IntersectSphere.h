#ifndef INTERSECT_SPHERE_H
#define INTERSECT_SPHERE_H

#include <vector> 
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector_types.h>

#include "RaytraceDefines.h"
#include "KernelMathHelper.h"
#include "RaytraceLighting.h"
#include "Primitives.h"
#include "Ray.h"


using std::vector; 

__device__ bool IntersectSphere(const Sphere* in_sphere, 
								const Ray* in_ray, 
								Intersection* inout_intersection, 
								bool storeResult)
{
	// sphere intersection
	bool result = false;
	float4 delta = in_sphere->pos - in_ray->origin;
	float B = cu_dot(in_ray->dir, delta);
	float D = B*B - cu_dot(delta, delta) + in_sphere->rad * in_sphere->rad; 
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
/*
__device__ bool IntersectSphereCRAZYDREAMVERSION(const Sphere* in_sphere, 
												 const Ray* in_ray, 
												 Intersection* inout_intersection, 
												 bool storeResult)
{
	// sphere intersection
	bool result = false;
	float4 delta = in_sphere->pos - in_ray->origin;
	float B = cu_dot(in_ray->dir, delta);
	float D = B*B - cu_dot(delta, delta) + in_sphere->rad * in_sphere->rad; 
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
*/

// Distance estimator for a field of spheres
__device__ float SphereFieldDE(float4 v)
{
	//float3 vv = make_float3(v.x,v.y,v.z);
	float2 vv = make_float2(v.x,v.z);
	
	float vxS = v.x/fabs(v.x);
	float vzS = v.z/fabs(v.z);
	vv = cu_fmodf(vv, make_float2(1.0f,1.0f)) - make_float2(vxS*0.5f,vzS*0.5f); // instance on xz-plane

	return cu_length(vv)-0.3f;							 // sphere DE
}





// Distance estimator for a space of spheres
// also returns unique index of sphere
__device__ float SphereSpaceDE(float4 in_v, float3* out_idx)
{	
	float localVolume = SPACEDIST; // the local space containing the sphere
	float localVolumeH = localVolume*0.5f;

	// for offset, shift ray here -------------------------v
	float3 vv = make_float3(in_v.x,in_v.y,in_v.z)/* - (float3)(sin(in_v.x/localVolume)*30.0f,0,0)*/;
	*out_idx = (vv/localVolumeH);
	out_idx->z = floor(out_idx->z+0.5f);
	out_idx->x = floor(out_idx->x+0.5f);
	out_idx->y = floor(out_idx->y+0.5f);

	float vxS = in_v.x/fabs(in_v.x);
	float vyS = in_v.y/fabs(in_v.y);
	float vzS = in_v.z/fabs(in_v.z);

	vv = cu_fmodf(vv, make_float3(localVolume,localVolume,localVolume)) - make_float3(vxS*localVolumeH,vyS*localVolumeH,vzS*localVolumeH); // instance in xyz-volume
	
	/*if (sin((*out_idx).x)>0.0f && sin((*out_idx).x)<1.0f
		&& sin((*out_idx).y)>0.0f && sin((*out_idx).y)<1.0f
		&& sin((*out_idx).z)>0.0f && sin((*out_idx).z)<1.0f)   // Culling can be done here, an example would be sampling against a 3d noise texture for a "cloud effect"
		return fabs(fast_length(vv)+40.0f); // distance to next
	else*/
	   return fabs(cu_length(vv)-SPHERERADIUS);		  // sphere DE
}

__device__ float RecursiveTetraDE1(float4 in_v,float3 in_pos)
{
	float3 z = make_float3(in_v.x,in_v.y,in_v.z)-in_pos;
	float Scale = 2.0f;
	float3 a1 = make_float3(1,1,1);
	float3 a2 = make_float3(-1,-1,1);
	float3 a3 = make_float3(1,-1,-1);
	float3 a4 = make_float3(-1,1,-1);
	float3 c;
	int n = 0;
	float dist, d;
	while (n < 8) {
		c = a1; dist = cu_length(z-a1);
		d = cu_length(z-a2); if (d < dist) { c = a2; dist=d; }
		d = cu_length(z-a3); if (d < dist) { c = a3; dist=d; }
		d = cu_length(z-a4); if (d < dist) { c = a4; dist=d; }
		z = Scale*z-c*(Scale-1.0);
		n++;
	}

	return cu_length(z) * pow(Scale, -(float)(n));
}

__device__ float RecursiveTetraDE2(float4 in_v,float3 in_pos)
{
	float3 z = make_float3(in_v.x,in_v.y,in_v.z)-in_pos;
	float3 g = z;
	int n = 0;
	float Scale = 2.0f;
	float Offset = 1.0f;
	while (n < 10) {
		g = z;
		if(z.x+z.y<0) {z.x = -g.y; z.y = -g.x; g = z;} // fold 1
		if(z.x+z.z<0) {z.x = -g.z; z.z = -g.x; g = z;} // fold 2
		if(z.y+z.z<0) {z.z = -g.y; z.y = -g.z; } // fold 3	
		z = z*Scale - Offset*(Scale-1.0);
		n++;
	}
	return (cu_length(z) ) * pow(Scale, -(float)(n));
}

__device__ float RecursiveMBulbDE(float4 in_v,float3 in_pos,float3 modifiers, float4& inout_orbittrap)
{
	float3 pos=make_float3(in_v.x,in_v.y,in_v.z)-in_pos;
	float3 z = pos;
	float3 g = z;
	float Power=8.0f-(modifiers.z*0.5f);
	float dr = 1.0f;
	float r = 0.0f;
	for (int i = 0; i < 10 ; i++) {
		r = cu_length(z);
		if (r>4.0f) break; // bailout
		if (r<inout_orbittrap.w) inout_orbittrap.w=r;
		if (z.x<inout_orbittrap.x) inout_orbittrap.x=z.x;
		if (z.y<inout_orbittrap.y) inout_orbittrap.y=z.y;
		if (z.z<inout_orbittrap.z) inout_orbittrap.z=z.z;

		// convert to polar coordinates
		//float theta = acos(z.z/r);
		//float phi = atan2(z.y,z.x);
		float theta = asin( z.z/r );
		float phi = atan2( z.y,z.x );
		dr =  pow( r, Power-1.0f)*Power*dr + 1.0f;

		// scale and rotate the point
		float zr = pow( r,Power);
		theta = (modifiers.y*0.5f+0.5f)*theta*Power;
		phi = (modifiers.z*0.5f+0.5f)*phi*Power;

		// convert back to Cartesian coordinates
		//z = zr*make_float3(sin(theta)*cos(phi), sin(phi)*sin(theta), cos(theta));
		float costheta=cos(theta);
		z = zr*make_float3(costheta*cos(phi), costheta*sin(phi), sin(theta));		
		z+=(modifiers.x+0.5f)*pos;

	}
	float detaiLv=0.1f;
	float clampDistN=min(squaredLen(&pos),20.0f)/20.0f; // distance to center, capped at 20 and normalized
	detaiLv += (1.0f-clampDistN)*0.4f; // distance reversed and scaled
	return detaiLv*log(r)*r/dr; // distance value used as multiplier for lod
}

// ray marching
__device__ bool MarchSphere(const Sphere* in_sphere, const Ray* in_ray, Intersection* inout_intersection, bool storeResult)
{

	float totalDistance = 0.0;
	int steps;
	int bsteps;
	int maximumRaySteps = 50;
	float minimumDistance = 0.01f;
	float4 p,p2;
	float3 sphereIdx = make_float3(0,0,0);
	float3 sphereIdx2 = make_float3(0,0,0);
	for (steps=0; steps < maximumRaySteps; steps++) 
	{
		p = in_ray->origin + totalDistance * in_ray->dir;
		float distance = SphereSpaceDE(p,&sphereIdx);
		//float distance =  RecursiveTetraDE2(p,(float3)(400.0f,400.0f,400.0f));
		totalDistance += distance;
		if (distance < minimumDistance) break;
	}

	if (steps < maximumRaySteps && (totalDistance > 0.001f) && 
		(totalDistance < inout_intersection->dist))
	{
		if (storeResult)
		{
			//float falloff=(cu_clamp(((float)steps/(float)maximumRaySteps)*4.0f+2.0f,2.0f,4.0f)-2.0f)*0.5f;
			// fog, easy when inside, as we see outer bounds
			float4 fogthick=in_ray->origin-p;
			float4 sphereC=make_float4(sphereIdx*SPACEDIST*0.5f,1.0f);
			// float inside=1.0f; not used right now
			// if outside, a bit trickier, get thickness from intersection-hidden back of sphere
			if (squaredLen(in_ray->origin-sphereC)>SPHERERADIUS*SPHERERADIUS)
			{
				//float4 radius=sphereC-p; // sphereIdx is center in world space
				float4 toCenter=sphereC-in_ray->origin;

				float4 pcd=cu_project(toCenter,cu_normalize(in_ray->dir))+in_ray->origin; // vector to center of sphere projected on ray
				float4 diff=p-pcd; // diff between first intersection and projected point
				float4 i2=diff*2.0f; // the other side is the offset from p by the double of diff
				fogthick=i2;
				//inside=0.0f;
			}
			float falloff = cu_length(fogthick)/(SPHERERADIUS*2.0f);
			falloff =min(1.0f,falloff*falloff*3.0f);


			// create ids with varying frequencies
			float3 uidValLong=make_float3(1.0f+sin(sphereIdx.x*0.1f)*0.5f,1.0f+sin(sphereIdx.y*0.1f)*0.5f,1.0f+sin(sphereIdx.z*0.1f)*0.5f);
			float3 uidValMed=make_float3(1.0f+sin(sphereIdx.x*0.3f)*0.5f,1.0f+sin(sphereIdx.y*0.3f)*0.5f,1.0f+sin(sphereIdx.z*0.3f)*0.5f);
			float3 uidValShort=make_float3(1.0f+sin(sphereIdx.x)*0.5f,1.0f+sin(sphereIdx.y)*0.5f,1.0f+sin(sphereIdx.z)*0.5f);
			float3 uidValMixed=make_float3(uidValLong.x*uidValMed.z+(1.0f-uidValShort.x*uidValMed.z),
				uidValLong.y*uidValMed.x+(1.0f-uidValShort.y*uidValMed.x),uidValLong.z*uidValMed.y+(1.0f-uidValShort.z*uidValMed.y));
			uidValMixed=(uidValMixed*uidValShort);
			inout_intersection->dist = totalDistance;
			inout_intersection->pos=p;
			float4 skycolor= make_float4(uidValShort*falloff,falloff);
			inout_intersection->surface.diffuse = skycolor;
			inout_intersection->normal = make_float4(0.0f,1.0f,0.0f,0.0f);


			// add starry sky
			float backtotdist=totalDistance+SPACEDIST*0.5f;
			for (bsteps=steps; bsteps < maximumRaySteps; bsteps++) 
			{
				p2 = in_ray->origin + backtotdist * in_ray->dir;
				float distance = SphereSpaceDE(p2,&sphereIdx2);
				backtotdist += distance;
				if (distance < minimumDistance) break;
			}
			float maxsightinatmos=2000.0f;
			float distfade=1.0f-( min(1.0f,backtotdist/maxsightinatmos) );
			float brightness=(1.0f-falloff)*distfade;
				//1.0f-(min(1.0f,(backtotdist/maxsightinatmos))*(1.0f-falloff));
			float3 uidValShort2=make_float3(1.0f+sin(sphereIdx2.x)*0.5f,1.0f+sin(sphereIdx2.y)*0.5f,1.0f+sin(sphereIdx2.z)*0.5f);
			if (bsteps < maximumRaySteps && (backtotdist > 0.001f))
			{
				inout_intersection->surface.diffuse = inout_intersection->surface.diffuse*(1.0f-brightness)+((skycolor*0.8f+make_float4(uidValShort2,1.0f))*brightness);
			}


			// Run fractal
			

			// float3 offset = (float3)(p.x,p.y,p.z);
			float3 offset = make_float3(sphereIdx.x,sphereIdx.y,sphereIdx.z)*SPACEDIST*0.5f; // * localVolumeH
			
			minimumDistance = max(0.001f,totalDistance*0.0001f);
			maximumRaySteps = 1110-cu_clamp(totalDistance,400,10);		
			totalDistance = 0.0;
			// float4 new_orig = p;
			float4 smallestorbit=make_float4(99999.0f);
			float minorbit=0.6f;
			float maxorbit=1.0f;
			for (steps=0; steps < maximumRaySteps; steps++) 
			{
				p = in_ray->origin + totalDistance * in_ray->dir;
				float distance = RecursiveMBulbDE(p,offset,uidValMixed,smallestorbit);
					//RecursiveTetraDE2(p,offset);
				totalDistance += distance;				
				if (distance < minimumDistance) break;
				// minimumDistance*=1.04f;
			}
			smallestorbit.x = cu_clamp(smallestorbit.x,minorbit,maxorbit);
			smallestorbit.y = cu_clamp(smallestorbit.y,minorbit,maxorbit);
			smallestorbit.z = cu_clamp(smallestorbit.z,minorbit,maxorbit);
			smallestorbit.w = cu_clamp(smallestorbit.w,minorbit,maxorbit);
			//if (smallestorbit<minorbit) smallestorbit=minorbit;
			//if (smallestorbit>maxorbit) smallestorbit=maxorbit;
			smallestorbit.x=(smallestorbit.x-minorbit)/(maxorbit-minorbit);
			smallestorbit.y=(smallestorbit.y-minorbit)/(maxorbit-minorbit);
			smallestorbit.z=(smallestorbit.z-minorbit)/(maxorbit-minorbit);
			smallestorbit.w=(smallestorbit.w-minorbit)/(maxorbit-minorbit);
			
			
			if (steps < maximumRaySteps && (totalDistance > 0.001f) && 
				(totalDistance < inout_intersection->dist+SPACEDIST))
			{			
					minimumDistance = totalDistance*0.5f;
					inout_intersection->dist = totalDistance; // distance

					inout_intersection->surface=in_sphere->mat; // material

					// for now, do super simple AO
					float c = smallestorbit.w;
					inout_intersection->surface.diffuse =make_float4(uidValShort.y*c,uidValShort.z*c,uidValShort.x*c,1.0f);
					// inout_intersection->surface.diffuse = (float4)(0.0f,1.0f,1.0f,0.0f);

					// store pos
					inout_intersection->pos=p;

					// Calculate the normal
					/*
					float4 xDir = make_float4(1.0f,0.0f,0.0f,0.0f)*0.1f;
					float4 yDir = make_float4(0.0f,1.0f,0.0f,0.0f)*0.1f;
					float4 zDir = make_float4(0.0f,0.0f,1.0f,0.0f)*0.1f;
					inout_intersection->normal=cu_normalize(make_float4(SphereSpaceDE(p+xDir,&sphereIdx)-SphereSpaceDE(p-xDir,&sphereIdx),
																	   SphereSpaceDE(p+yDir,&sphereIdx)-SphereSpaceDE(p-yDir,&sphereIdx),
																	   SphereSpaceDE(p+zDir,&sphereIdx)-SphereSpaceDE(p-zDir,&sphereIdx),0.0f));
					*/
					/*
					float3 f = offset;
					float discard=0.0f;
					float4 xDir = make_float4(1.0f,0.0f,0.0f,0.0f)*0.0001f;
					float4 yDir = make_float4(0.0f,1.0f,0.0f,0.0f)*0.0001f;
					float4 zDir = make_float4(0.0f,0.0f,1.0f,0.0f)*0.0001f;
					inout_intersection->normal=cu_normalize(make_float4(RecursiveMBulbDE(p+xDir,f,uidValMixed,discard)-RecursiveMBulbDE(p-xDir,f,uidValMixed,discard),
																	   RecursiveMBulbDE(p+yDir,f, uidValMixed,discard)-RecursiveMBulbDE(p-yDir,f,uidValMixed,discard),
																	   RecursiveMBulbDE(p+zDir,f, uidValMixed,discard)-RecursiveMBulbDE(p-zDir,f,uidValMixed,discard),0.0f));
					*/


					inout_intersection->normal = make_float4(0.0f,1.0f,0.0f,0.0f);
			}
		}

		return true;
	}
	

	return false;
}


#endif