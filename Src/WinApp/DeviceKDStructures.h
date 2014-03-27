#ifndef RAYTRACE_KDNODESTRUCTURES_H
#define RAYTRACE_KDNODESTRUCTURES_H


// =======================================================================================
//                                    KD Node Structures
// =======================================================================================

#include <vector> 
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector_types.h>

struct DKDAxisMark
{
	int b_1; // use ints while testing
	int b_2;
};

struct DKDNode
{
	DKDAxisMark m_split;
	int m_isLeaf;
	float m_position;
	int m_leftChildIdx;
};

struct DKDLeaf
{
	int m_offset; // offset in KD-sorted index list
	int m_count; // number of indices for this node
};

struct DKDStack
{
	int m_nodeIdx;				// idx to node in array
	float m_t;					// distance
	float pb[3];				// point on box
	int prev, dummy1, dummy2;
};

#endif