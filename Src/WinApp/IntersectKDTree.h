#ifndef INTERSECT_KDTREE_H
#define INTERSECT_KDTREE_H

#include <vector> 
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector_types.h>


#include "KernelMathHelper.h"
#include "RaytraceLighting.h"
#include "Primitives.h"
#include "Scene.h"
#include "Ray.h"
#include "DeviceKDStructures.h"
#include "IntersectTriangle.h"

__device__ int getAxisNumber(DKDAxisMark p_axis)
{
	return p_axis.b_1+p_axis.b_2*2;
}

__device__ bool KDTraverse( const Scene* in_scene, const Ray* in_ray, /*float4x4 p_view,*/float3& p_extends, float3& p_pos,
						 DKDNode* p_nodes, DKDLeaf* p_leaflist, unsigned int* p_nodeIndices, 
						 unsigned int p_numNodeIndices,
						 float3* p_verts,float3* p_uvs,float3* p_norms,
						 float3* p_outDbgCol,
						 Intersection* inout_intersection, 
						 bool storeResult)
{
	float3 colarr[18]={make_float3(1.0f,0.0f,0.0f),
					   make_float3(0.0f,1.0f,0.0f),
					   make_float3(0.0f,0.0f,1.0f),
					   make_float3(1.0f,0.0f,1.0f),
					   make_float3(0.0f,1.0f,1.0f),
					   make_float3(1.0f,1.0f,0.0f),
					   make_float3(1.0f,0.5f,0.0f),
					   make_float3(0.5f,1.0f,0.0f),
					   make_float3(1.0f,0.0f,0.5f),
					   make_float3(0.5f,0.5f,0.5f),
					   make_float3(1.0f,0.7f,0.7f),
					   make_float3(1.0f,1.0f,0.2f),
					   make_float3(0.33f,0.25f,0.4f),
					   make_float3(0.8f,1.0f,0.24f),
					   make_float3(0.0f,0.66f,0.5f),
					   make_float3(0.1f,0.231f,0.13f),
					   make_float3(0.87f,0.5f,1.0f),
					   make_float3(0.2f,0.2f,1.0f)};

//	p_verts=in_scene->meshVerts; p_norms=in_scene->meshNorms;

	// Intersection intersection;
	// intersection.dist = MAX_INTERSECT_DIST;
	// intersection.surface.diffuse = make_float4(0.0f,0.0f,0.0f,0.0f);
	// intersection.surface.specular = make_float4(0.0f,0.0f,0.0f,0.0f);
	// intersection.surface.reflection= 0.0f;

	DKDLeaf leaf;
	Material material;


	float3 breakCol=make_float3(0.0f,0.0f,0.0f);
	*p_outDbgCol=breakCol;

	float3 result=make_float3(0.0f,0.0f,0.0f);
	float3 hitViz=make_float3(0.25f,0.15f,0.15f);
	float3 overlayViz=make_float3(0.0f,0.0f,0.0f);
	float tnear = 0.0f, tfar = MAX_INTERSECT_DIST, t;
	int retval = 0;
	float3 treePos=p_pos;
	float3 treeExt=p_extends;
	float3 p1 = treePos - treeExt*0.5f;				// Get box min
	float3 p2 = treePos + treeExt*0.5f;			// Get box max (world space)
	//mat4mul(&p_view,&p1, &p1);
	float3 D = make_float3(in_ray->dir.x,in_ray->dir.y,in_ray->dir.z), 
		   O = make_float3(in_ray->origin.x,in_ray->origin.y,in_ray->origin.z);	// Get ray
	// store in small arrays for axis access
	float ap1[3]={p1.x,p1.y,p1.z};
	float ap2[3]={p2.x,p2.y,p2.z};
	float aD[3]  ={D.x, D.y, D.z};
	float aO[3]  ={O.x, O.y, O.z};
	int mod_list[5] = {0,1,2,0,1}; // modulo for axes
	DKDStack kdStack[64]; // traversal stack

	// Exclude rays which are pointing to the left (for axis) and with an origin (axis) less than box negative extents
	// or if right to axis and origin more than positive extents
	// For which there zero chance of hit
	///////////////////////////////////////////
	#pragma unroll 3
	for ( int i = 0; i < 3; i++ ) 
	{
		if (aD[i] < 0.0f) 
		{
			if (aO[i] < ap1[i]) return false; // Negative extents(note that author has anchor in corner here)
		}
		else if (aO[i] > ap2[i]) return false;// positive extents
	}
	///////////////////////////////////////////
	///////////////////////////////////////////
	// clip ray segment to box
	///////////////////////////////////////////
	bool isInside=IntersectAABBCage(treePos, treeExt, in_ray, tfar, tnear, tfar);
	if (!isInside) 
		return false;
	if (tnear>tfar)
	{
		float t=tfar;
		tfar=tnear;
		tnear=t;
	}

	///////////////////////////////////////////
	///////////////////////////////////////////
	// init stack of traversal
	// stack has 64 slots
	////////	struct kdstack
	////////	{
	////////		KdTreeNode* node;
	////////		real t;
	////////		vector3 pb;
	////////		int prev, dummy1, dummy2;
	////////	};
	////////struct DKDStack
	////////{
	////////	int m_nodeIdx;				// idx to node in array
	////////	float m_t;					// distance
	////////	float pb[3];				// point on box
	////////	int prev, dummy1, dummy2;
	////////};
	int entrypoint = 0, exitpoint = 1;
	// init traversal
	int farchildNodeIdx=1, currNodeIdx; // farchild seems to be sibling of current. Current is the currently active node while traversing
	currNodeIdx = 1;//m_Scene->GetKdTree()->GetRoot(); // start at root node
	kdStack[entrypoint].m_t = tnear; // add near value to "t"
	DKDNode currNode;

	// if near is more than zero
	// add start point on hit on box
	// ray origin + direction, scaled with near distance
	if (tnear > 0.0f) 
		kdStack[entrypoint].m_pb = O + D * tnear;
	else 
		kdStack[entrypoint].m_pb = O;

	// Add the furthest point (back of the voxel) to the exit point 
	// in the stack.
	kdStack[exitpoint].m_t = tfar;
	kdStack[exitpoint].m_pb = O + D * tfar;
	kdStack[exitpoint].m_nodeIdx = 0;

	// Now we have the entry- and exit points on the box!

	///////////////////////////////////////////
	///////////////////////////////////////////
	// traverse kd-tree
	///////////////////////////////////////////
	///////////////////////////////////////////
	int totalIndices=0;
	while (currNodeIdx>0/* && breaker>0*/) // While we have a current node
	{
		currNode=p_nodes[currNodeIdx]; // Copy current node to register

		///////////////////////////////////////////
		// While we have a current node that is not a leaf
		while (currNode.m_isLeaf<1/* && bbr>0.0f*/)
		{
			// get split dist and axis for node
			float splitpos = currNode.m_position;
			int axis = getAxisNumber(currNode.m_split); // get the current active axis 0=x,1=y,2=z (index used for addressing)
			float3 enpb = kdStack[entrypoint].m_pb;
			float3 expb = kdStack[exitpoint].m_pb;
			float entry_pb[3]  ={enpb.x,enpb.y,enpb.z};
			float exit_pb[3]  ={expb.x,expb.y,expb.z};
			bool nodeSet=false;
			//--------------------------------------------------
		    // if active axis of ENTRYpoint is less than split value
			if (entry_pb[axis] <= splitpos) 
			{
				
				if (exit_pb[axis] <= splitpos) // if active axis of EXITpoint is less than split dist
				{
					//hitViz=make_float3(0.0f,0.5f,0.0f);
					currNodeIdx = currNode.m_leftChildIdx+1; // iterate to the left child of current
					//if (currNodeIdx<=0) return hitViz;
					//currNode=p_nodes[currNodeIdx];
					nodeSet=true; // NEXT ITERATION!!!
				}
				if (!nodeSet && exit_pb[axis] == splitpos) // if active axis of EXITpoint is equal to split dist LOL
				{
					currNodeIdx = currNode.m_leftChildIdx; // iterate to the right child of current
					//if (currNodeIdx<=0) return hitViz;
					//currNode=p_nodes[currNodeIdx];
					nodeSet=true; // NEXT ITERATION!!!
				}
				// Default: iterate to the left child of current
				if (!nodeSet)
				{
					 //currNodeIdx = currNode.m_leftChildIdx; 
					 //farchildNodeIdx = currNodeIdx + 1; // GetRight(); // set farchild to sibling of current
					currNodeIdx = currNode.m_leftChildIdx+1;
					farchildNodeIdx = currNode.m_leftChildIdx;
					//if (currNodeIdx<=0) return hitViz;
					//currNode=p_nodes[currNodeIdx];
				}
			}
			// if active axis of ENTRYpoint is more than or equal to split value
			else
			{
				
				if (exit_pb[axis] > splitpos) // if active axis of EXITpoint is greater than split dist
				{
					//hitViz=make_float3(0.5f,0.0f,0.0f);
					currNodeIdx = currNode.m_leftChildIdx; // iterate to the right child of current
					//if (currNodeIdx<=0) return hitViz;
					//currNode=p_nodes[currNodeIdx];
					nodeSet=true;  // NEXT ITERATION!!!
				}					
				// Default: iterate to the right child of current
				if (!nodeSet)
				{
					 //farchildNodeIdx = currNodeIdx; // set sibling to left child of current
					 //currNodeIdx = farchildNodeIdx + 1; // GetRight(); // set current to right child of current
					currNodeIdx = currNode.m_leftChildIdx;
					farchildNodeIdx = currNode.m_leftChildIdx+1;
					//if (currNodeIdx<=0) return hitViz;
					//currNode=p_nodes[currNodeIdx];
				}			
			}
			if (!nodeSet)
			{
				//--------------------------------------------------
				// update distance width 
				t = (splitpos - aO[axis]) / aD[axis]; // set t-distance to (splitdist - (active axis of rayorig)) / (active axis of ray dir)
				// increase exit point, and store it
				int tmp = exitpoint++;
				// if the exitpoint==entrypoint, inrease exitpoint again
				if (exitpoint == entrypoint) exitpoint++; 
				// update pb
				expb = kdStack[exitpoint].m_pb;
				exit_pb[0] = expb.x; exit_pb[1]=expb.y; exit_pb[2]=expb.z;
				// Set values for exitpoint
				kdStack[exitpoint].m_prev = tmp; // previous is same if exitpoint wasnt entry, otherwise it is previous
				kdStack[exitpoint].m_t = t;
				kdStack[exitpoint].m_nodeIdx = farchildNodeIdx;

				exit_pb[axis]=splitpos;
				kdStack[exitpoint].m_pb.x = exit_pb[0];kdStack[exitpoint].m_pb.y = exit_pb[1];kdStack[exitpoint].m_pb.z = exit_pb[2];

				int nextaxis = mod_list[axis + 1];
				int prevaxis = mod_list[axis + 2];

				exit_pb[0]=kdStack[exitpoint].m_pb.x; exit_pb[1]=kdStack[exitpoint].m_pb.y; exit_pb[2]=kdStack[exitpoint].m_pb.z;
				exit_pb[nextaxis] = aO[nextaxis] + t * aD[nextaxis];
				exit_pb[prevaxis] = aO[prevaxis] + t * aD[prevaxis];
				kdStack[exitpoint].m_pb.x = exit_pb[0];kdStack[exitpoint].m_pb.y = exit_pb[1];kdStack[exitpoint].m_pb.z = exit_pb[2];
			}
			// Fetch new node
			if (currNodeIdx<=0)
			{
				*p_outDbgCol=hitViz*0.6f+overlayViz;
				return false;
			}
			currNode=p_nodes[currNodeIdx];

		}
		//if (hitViz<0.47f) hitViz=1.0f;
		//hitViz+=make_float3(0.0f,0.0f,0.01f);
		// End while not leaf
		///////////////////////////////////////////
		///////////////////////////////////////////

		// Now we have a leaf!

		///////////////////////////////////////////
		// Get list of current triangles for leaf
					
		if (currNodeIdx>0) hitViz=colarr[currNodeIdx%17];
		if (currNodeIdx>0) overlayViz+=colarr[currNodeIdx%17]*0.5f;
		int leafId=currNode.m_leftChildIdx;
		

		if (leafId>-1)
		{
			
			leaf=p_leaflist[leafId];
			int indexOffset=leaf.m_offset;
			int indexCount=leaf.m_count;

			material.diffuse = make_float4(hitViz.x,hitViz.y,hitViz.z,0.0f);
			material.specular = make_float4(0.0f, 0.0f, 0.0f,0.0f);
			material.reflection = 0.0f;

			bool hit=false;
			float maxdist=kdStack[exitpoint].m_t;
			float od=inout_intersection->dist;
			if (maxdist<inout_intersection->dist) inout_intersection->dist=maxdist;
			unsigned int* ind = p_nodeIndices;
			for (unsigned int i=0;i<indexCount;i+=3)
			{			
				hit|=IntersectTriangle(p_verts, p_uvs, p_norms, 
					ind[indexOffset+i], ind[indexOffset+i+1], ind[indexOffset+i+2], 
					&material, 
					in_ray, inout_intersection,storeResult);
			
			}	// for each face (three indices)
			if (hit)
			{
				*p_outDbgCol=hitViz*(0.6f+(inout_intersection->normal.x+inout_intersection->normal.y+inout_intersection->normal.z)/3.0f);
				return true;
					// make_float3(intersection.surface.diffuse.x,intersection.surface.diffuse.y,intersection.surface.diffuse.z);
			}
			else
			{
				inout_intersection->dist=od;
			}
		}

			
		//return hitViz;
		// If we got a hit, we return the result:
		// not checking hits here (we're using global memory) if (retval) return retval;

		// Otherwise, start checking the neighbour behind
		// By setting the new entry point to this voxel's exitpoint
		entrypoint = exitpoint;
		currNodeIdx = kdStack[exitpoint].m_nodeIdx;
		//if (currNodeIdx<=0) return hitViz;
		//currNode=p_nodes[currNodeIdx];
		exitpoint = kdStack[entrypoint].m_prev;
	} // endwhile we have node

	///////////////////////////////////////////
	///////////////////////////////////////////
	
	*p_outDbgCol=hitViz*0.6f+overlayViz;
	return false; 
		//hitViz*0.6f+overlayViz;
}

#endif