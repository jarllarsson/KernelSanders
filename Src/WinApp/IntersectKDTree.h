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
#include "Ray.h"


int Engine::FindNearest( Ray& a_Ray, real& a_Dist, Primitive*& a_Prim )
{
	real tnear = 0, tfar = a_Dist, t;
	int retval = 0;
	vector3 p1 = m_Scene->GetExtends().GetPos();				// Get box center
	vector3 p2 = p1 + m_Scene->GetExtends().GetSize();			// Get box extents (world space)
	vector3 D = a_Ray.GetDirection(), O = a_Ray.GetOrigin();	// Get ray
	// Exclude rays which are pointing to the left (for axis) and with an origin (axis) less than box negative extents
	// or if right to axis and origin more than positive extents
	// For which there zero chance of hit
	///////////////////////////////////////////
	for ( int i = 0; i < 3; i++ ) 
	{
		if (D.cell[i] < 0) 
		{
			if (O.cell[i] < p1.cell[i]) return 0; // Negative extents(note that author has anchor in corner here)
		}
		else if (O.cell[i] > p2.cell[i]) return 0;// positive extents
	}
	///////////////////////////////////////////
	///////////////////////////////////////////
	// clip ray segment to box
	///////////////////////////////////////////
	for ( i = 0; i < 3; i++ )
	{
		real pos = O.cell[i] + tfar * D.cell[i];
		if (D.cell[i] < 0)
		{
			// clip end point
			if (pos < p1.cell[i]) tfar = tnear + (tfar - tnear) * ((O.cell[i] - p1.cell[i]) / (O.cell[i] - pos));
			// clip start point
			if (O.cell[i] > p2.cell[i]) tnear += (tfar - tnear) * ((O.cell[i] - p2.cell[i]) / (tfar * D.cell[i]));
		}
		else
		{
			// clip end point
			if (pos > p2.cell[i]) tfar = tnear + (tfar - tnear) * ((p2.cell[i] - O.cell[i]) / (pos - O.cell[i]));
			// clip start point
			if (O.cell[i] < p1.cell[i]) tnear += (tfar - tnear) * ((p1.cell[i] - O.cell[i]) / (tfar * D.cell[i]));
		}
		if (tnear > tfar) return 0;
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
	int entrypoint = 0, exitpoint = 1;
	// init traversal
	KdTreeNode* farchild, *currnode; // farchild seems to be sibling of current. Current is the currently active node while traversing
	currnode = m_Scene->GetKdTree()->GetRoot(); // start at root node
	m_Stack[entrypoint].t = tnear; // add near value to "t"

	// if near is more than zero
	// add start point on hit on box
	// ray origin + direction, scaled with near distance
	if (tnear > 0.0f) 
		m_Stack[entrypoint].pb = O + D * tnear;
	else 
		m_Stack[entrypoint].pb = O;

	// Add the furthest point (back of the voxel) to the exit point 
	// in the stack.
	m_Stack[exitpoint].t = tfar;
	m_Stack[exitpoint].pb = O + D * tfar;
	m_Stack[exitpoint].node = 0;

	// Now we have the entry- and exit points on the box!

	///////////////////////////////////////////
	///////////////////////////////////////////
	// traverse kd-tree
	///////////////////////////////////////////
	///////////////////////////////////////////
	while (currnode) // While we have a current node
	{
		///////////////////////////////////////////
		// While we have a current node that is not a leaf
		while (!currnode->IsLeaf()) 
		{
			// get split dist and axis for node
			real splitpos = currnode->GetSplitPos();
			int axis = currnode->GetAxis(); // get the current active axis 0=x,1=y,2=z (index used for addressing)
			//--------------------------------------------------
		    // if active axis of ENTRYpoint is less than split value
			if (m_Stack[entrypoint].pb.cell[axis] <= splitpos) 
			{
				if (m_Stack[exitpoint].pb.cell[axis] <= splitpos) // if active axis of EXITpoint is less than split dist
				{
					currnode = currnode->GetLeft(); // iterate to the left child of current
					continue; // NEXT ITERATION!!!
				}
				if (m_Stack[exitpoint].pb.cell[axis] == splitpos) // if active axis of EXITpoint is equal to split dist LOL
				{
					currnode = currnode->GetRight(); // iterate to the right child of current
					continue; // NEXT ITERATION!!!
				}
				// Default: iterate to the left child of current
				currnode = currnode->GetLeft(); 
				farchild = currnode + 1; // GetRight(); // set farchild to sibling of current
			}
			// if active axis of ENTRYpoint is more than or equal to split value
			else
			{
				if (m_Stack[exitpoint].pb.cell[axis] > splitpos) // if active axis of EXITpoint is greater than split dist
				{
					currnode = currnode->GetRight(); // iterate to the right child of current
					continue;  // NEXT ITERATION!!!
				}
				// Default: iterate to the right child of current
				farchild = currnode->GetLeft(); // set sibling to left child of current
				currnode = farchild + 1; // GetRight(); // set current to right child of current
			}
			//--------------------------------------------------
			t = (splitpos - O.cell[axis]) / D.cell[axis];
			int tmp = exitpoint++;
			if (exitpoint == entrypoint) exitpoint++;
			m_Stack[exitpoint].prev = tmp;
			m_Stack[exitpoint].t = t;
			m_Stack[exitpoint].node = farchild;
			m_Stack[exitpoint].pb.cell[axis] = splitpos;
			int nextaxis = m_Mod[axis + 1];
			int prevaxis = m_Mod[axis + 2];
			m_Stack[exitpoint].pb.cell[nextaxis] = O.cell[nextaxis] + t * D.cell[nextaxis];
			m_Stack[exitpoint].pb.cell[prevaxis] = O.cell[prevaxis] + t * D.cell[prevaxis];
		}
		// End while not leaf
		///////////////////////////////////////////
		///////////////////////////////////////////

		// Now we have a leaf!

		///////////////////////////////////////////
		// Get list of current triangles for leaf
		ObjectList* list = currnode->GetList();
		real dist = m_Stack[exitpoint].t; // get the current max distance (voxel back)
		// Check all triangles that's in the node
		while (list) // can make this forall essentially
		{
			Primitive* pr = list->GetPrimitive(); // tri here
			int result;
			m_Intersections++; // count intersections
			// If we hit:
			if (result = pr->Intersect( a_Ray, dist ))
			{
				// register result and distance
				retval = result;
				a_Dist = dist;
				a_Prim = pr;
			}
			// fetch next triangle to check
			list = list->GetNext();
		}
		// If we got a hit, we return the result:
		if (retval) return retval;

		// Otherwise, start checking the neighbour behind
		// By setting the new entry point to this voxel's exitpoint
		entrypoint = exitpoint;
		currnode = m_Stack[exitpoint].node;
		exitpoint = m_Stack[entrypoint].prev;
	}
	///////////////////////////////////////////
	///////////////////////////////////////////
	return 0;
}

#endif