#pragma once

#include "KDTreeFactory.h"

KDTreeFactory::KDTreeFactory()
{

}

KDTreeFactory::~KDTreeFactory()
{
	for (int i=0;i<m_trees.size();i++)
	{
		delete m_trees[i];
	}
	m_trees.clear();
	for (int i=0;i<m_leafLists.size();i++)
	{
		delete m_leafLists[i];
	}
	m_leafLists.clear();
}

int KDTreeFactory::calculateKDTree( void* p_vec3ArrayXYZ, int p_vertCount, 
								    unsigned int* p_indexArray, int p_iCount )
{
	// Create root node

	// Build rest of tree using it
}



void KDTreeFactory::subdivide( KDNode& p_node, int p_dimsz, int p_dim, int p_idx, const glm::vec3& pos, const glm::vec3& parentSize )
{

}

bool KDTreeFactory::triIntersectNode( int p_triangleIdx, const glm::vec3& pos, const glm::vec3& parentSize )
{

}

float KDTreeFactory::findOptimalSplitPos( KDNode& p_node, const KDAxisMark& p_axis, const glm::vec3& p_currentSize, const glm::vec3& p_currentPos )
{

}

float KDTreeFactory::getLeftExtreme( int p_triangleIdx, const glm::vec3& p_axis )
{

}

float KDTreeFactory::getRightExtreme( int p_triangleIdx, const glm::vec3& p_axis )
{

}

float KDTreeFactory::calculatecost( const KDNode& p_node, float p_splitpos, const glm::vec3& p_axis, const glm::vec3& p_currentSize, const glm::vec3& p_currentPos )
{

}

void KDTreeFactory::calculatePrimitiveCount( const KDNode& p_node, const glm::vec3& p_leftBox,const glm::vec3& p_rightBox, const glm::vec3& p_leftBoxPos, const glm::vec3& p_rightBoxPos, int* p_outLeftCount, int* p_outRightCount )
{

}

float KDTreeFactory::calculateArea( glm::vec3& p_extents )
{
	return 2.0f * p_extents.x * p_extents.y * p_extents.z;
}

void KDTreeFactory::getChildVoxelsMeasurement( float p_inSplitpos, const glm::vec3& p_axis, 
											   const glm::vec3& p_inParentSize, glm::vec3* p_outLeftSz, glm::vec3* p_outRightSz )
{

}

glm::vec3 KDTreeFactory::entrywiseMul( const glm::vec3& p_a, const glm::vec3& p_b )
{
	return glm::vec3(p_a.x*p_b.x,p_a.y*p_b.y,p_a.z*p_b.z);
}

int KDTreeFactory::addTree( vector<KDNode>* p_tree, vector<KDLeaf>* p_leafList )
{
	m_trees.push_back(p_tree);
	m_leafLists.push_back(p_leafList);
	return m_trees.size()-1;
}