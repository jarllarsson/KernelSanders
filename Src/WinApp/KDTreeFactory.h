#pragma once

#include <vector>
#include "KDNode.h"

using namespace std;

// =======================================================================================
//                                      KDTreeFactory
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Generates an array describing a KD-tree based on a triangle mesh
///        
/// # KDTreeFactory
/// 
/// 11-3-2014 Jarl Larsson
///---------------------------------------------------------------------------------------


class KDTreeFactory
{
public:


	KDTreeFactory();
	virtual ~KDTreeFactory();

	// Builds a tree, stores it and returns the index to it
	int calculateKDTree(void* p_vec3ArrayXYZ, int p_vertCount,
						 unsigned int* p_indexArray, int p_iCount);

	vector<KDNode>* getTree(int p_idx);
	vector<KDLeaf>* getLeafList(int p_idx);
	
protected:
private:

	void subdivide(KDNode& p_node, int p_dimsz, int p_dim, int p_idx, const glm::vec3& pos, const glm::vec3& parentSize);

	bool triIntersectNode(int p_triangleIdx, const glm::vec3& pos, const glm::vec3& parentSize);

	float findOptimalSplitPos(KDNode& p_node, const KDAxisMark& p_axis, 
							  const glm::vec3&  p_currentSize,  const glm::vec3& p_currentPos);

	float getLeftExtreme(int p_triangleIdx, const glm::vec3& p_axis);

	float getRightExtreme(int p_triangleIdx, const glm::vec3& p_axis);

	float calculatecost(const KDNode& p_node, float p_splitpos,  const glm::vec3& p_axis, 
						const glm::vec3& p_currentSize, const glm::vec3& p_currentPos);

	void calculatePrimitiveCount(const KDNode& p_node, const glm::vec3& p_leftBox,const glm::vec3& p_rightBox,
								 const glm::vec3& p_leftBoxPos, const glm::vec3& p_rightBoxPos,
								 int* p_outLeftCount, int* p_outRightCount);

	float calculateArea( glm::vec3& p_extents);

	void getChildVoxelsMeasurement(float p_inSplitpos, const glm::vec3& p_axis, const glm::vec3& p_inParentSize,
								  glm::vec3* p_outLeftSz, glm::vec3* p_outRightSz);

	glm::vec3 entrywiseMul(const glm::vec3& p_a, const glm::vec3& p_b);

	int addTree(vector<KDNode>* p_tree, vector<KDLeaf>* p_leafList);

	// Storage
	vector<vector<KDNode>*> m_trees;
	vector<vector<KDLeaf>*> m_leafLists;
};