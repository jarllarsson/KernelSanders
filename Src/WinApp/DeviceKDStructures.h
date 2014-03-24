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

struct DKDNode
{
	KDAxisMark m_split;
	bool m_isLeaf;
	float m_position;
	int m_leftChildIdx; // right child is always left child+1
	// and if leaf then this is the index to
	// to array of indices
	// short leafcount SEE IF NEEDED
};

struct DKDLeaf
{
	int m_offset; // offset in KD-sorted index list
	int m_count; // number of indices for this node
};