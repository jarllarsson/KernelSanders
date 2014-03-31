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

int getAxisNumber(DKDAxisMark p_axis)
{
	return p_axis.b_1+p_axis.b_2*2;
}

float KDTraverse( Scene* in_scene, const Ray* in_ray, float3 p_extends, float3 p_pos,
						 DKDNode* p_nodes, DKDLeaf* p_leaflist, unsigned int* p_nodeIndices,
						 float3* p_verts,float3* p_norms)
{
	float hitViz=0.0f;
	float tnear = 0.0f, tfar = MAX_INTERSECT_DIST, t;
	int retval = 0;
	float3 p1 = p_pos;				// Get box center
	float3 p2 = p1 + p_extends;			// Get box extents (world space)
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
		if (aD[i] < 0) 
		{
			if (aO[i] < ap1[i]) return 0; // Negative extents(note that author has anchor in corner here)
		}
		else if (aO[i] > ap2[i]) return 0;// positive extents
	}
	///////////////////////////////////////////
	///////////////////////////////////////////
	// clip ray segment to box
	///////////////////////////////////////////
	#pragma unroll 3
	for (int i = 0; i < 3; i++ )
	{
		float pos = aO[i] + tfar * aD[i];
		if (aD[i] < 0.0f)
		{
			// clip end point
			if (pos < ap1[i]) tfar = tnear + (tfar - tnear) * ((aO[i] - ap1[i]) / (aO[i] - pos));
			// clip start point
			if (aO[i] > ap2[i]) tnear += (tfar - tnear) * ((aO[i] - ap2[i]) / (tfar * aD[i]));
		}
		else
		{
			// clip end point
			if (pos > ap2[i]) tfar = tnear + (tfar - tnear) * ((ap2[i] - aO[i]) / (pos - aO[i]));
			// clip start point
			if (aO[i] < ap1[i]) tnear += (tfar - tnear) * ((ap1[i] - aO[i]) / (tfar * aD[i]));
		}
		if (tnear > tfar) return 0;
	}
	O.x=aO[0];O.y=aO[0];O.y=aO[0]; // copy back
	D.x=aD[0];D.y=aD[0];D.y=aD[0]; //
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
	int farchildNodeIdx, currNodeIdx; // farchild seems to be sibling of current. Current is the currently active node while traversing
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
	while (currNodeIdx>0) // While we have a current node
	{
		currNode=p_nodes[currNodeIdx]; // Copy current node to register
		hitViz+=0.01f;
		///////////////////////////////////////////
		// While we have a current node that is not a leaf
		while (currNode.m_isLeaf<1) 
		{
			// get split dist and axis for node
			float splitpos = currNode.m_position;
			int axis = getAxisNumber(currNode.m_split); // get the current active axis 0=x,1=y,2=z (index used for addressing)
			float3 enpb = kdStack[entrypoint].m_pb;
			float3 expb = kdStack[exitpoint].m_pb;
			float entry_pb[3]  ={enpb.x,enpb.y,enpb.z};
			float exit_pb[3]  ={expb.x,expb.y,expb.z};
			//--------------------------------------------------
		    // if active axis of ENTRYpoint is less than split value
			if (entry_pb[axis] <= splitpos) 
			{
				if (exit_pb[axis] <= splitpos) // if active axis of EXITpoint is less than split dist
				{
					currNodeIdx = currNode.m_leftChildIdx; // iterate to the left child of current
					continue; // NEXT ITERATION!!!
				}
				if (exit_pb[axis] == splitpos) // if active axis of EXITpoint is equal to split dist LOL
				{
					currNodeIdx = currNode.m_leftChildIdx+1; // iterate to the right child of current
					continue; // NEXT ITERATION!!!
				}
				// Default: iterate to the left child of current
				currNodeIdx = currNode.m_leftChildIdx; 
				farchildNodeIdx = currNodeIdx + 1; // GetRight(); // set farchild to sibling of current
			}
			// if active axis of ENTRYpoint is more than or equal to split value
			else
			{
				if (exit_pb[axis] > splitpos) // if active axis of EXITpoint is greater than split dist
				{
					currNodeIdx = currNode.m_leftChildIdx+1; // iterate to the right child of current
					continue;  // NEXT ITERATION!!!
				}
				// Default: iterate to the right child of current
				farchildNodeIdx = currNodeIdx; // set sibling to left child of current
				currNodeIdx = farchildNodeIdx + 1; // GetRight(); // set current to right child of current
			}
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
			// Fetch new node
			currNode=p_nodes[currNodeIdx];
		}
		// End while not leaf
		///////////////////////////////////////////
		///////////////////////////////////////////

		// Now we have a leaf!

		///////////////////////////////////////////
		// Get list of current triangles for leaf
		int leafId=currNode.m_leftChildIdx;
		DKDLeaf leaf=p_leaflist[leafId];
		int indexOffset=leaf.m_offset;
		int indexCount=leaf.m_count;
		indexCount -= cu_imaxi(0,(indexOffset+indexCount)-MAXMESHLOCAL_INDICESBIN);
		float dist = kdStack[exitpoint].m_t; // get the current max distance (voxel back)
		// Check all triangles that's in the node
		// Fetch all triangles in path
		int vertOffset=in_scene->numVerts;
		in_scene->numIndices+=indexCount;
		in_scene->numVerts+=indexCount;
		for (unsigned int i=0;i<indexCount;i++)
		{
			int index=p_nodeIndices[indexOffset+i]; // fetch index
			int newindex=vertOffset+i;
			in_scene->meshVerts[newindex]=p_verts[index]; // and get corresponding
			in_scene->meshNorms[newindex]=p_norms[index]; // vertex and normals data
			in_scene->meshIndices[newindex]=newindex; // store new index (note this method creates vertex copies)
		}

		//while (list) // can make this forall essentially
		//{
		//	Primitive* pr = list->GetPrimitive(); // tri here
		//	int result;
		//	m_Intersections++; // count intersections
		//	// If we hit:
		//	if (result = pr->Intersect( a_Ray, dist ))
		//	{
		//		// register result and distance
		//		retval = result;
		//		a_Dist = dist;
		//		a_Prim = pr;
		//	}
		//	// fetch next triangle to check
		//	list = list->GetNext();
		//}
		// If we got a hit, we return the result:
		// not checking hits here (we're using global memory) if (retval) return retval;

		// Otherwise, start checking the neighbour behind
		// By setting the new entry point to this voxel's exitpoint
		entrypoint = exitpoint;
		currNodeIdx = kdStack[exitpoint].m_nodeIdx;
		currNode=p_nodes[currNodeIdx];
		exitpoint = kdStack[entrypoint].m_prev;
	}
	///////////////////////////////////////////
	///////////////////////////////////////////
	return hitViz;
}

#endif