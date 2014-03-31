#pragma once

#include <stack>
#include <vector>
#include "KDNode.h"
#include "KDBounds.h"


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
private:
	static const int sc_treeListMaxSize=8024;
public:


	KDTreeFactory();
	virtual ~KDTreeFactory();

	// Builds a tree, stores it and returns the index to it
	int buildKDTree(void* p_vec3ArrayXYZ,void* p_normArrayXYZ, int p_vertCount, unsigned int* p_indexArray, int p_iCount, glm::vec3 p_boundsMin, glm::vec3 p_boundsMax);

	vector<KDNode>* getTree(int p_idx);
	vector<KDLeaf>* getLeafList(int p_idx);
	vector<int>* getLeafDataList(int p_idx);
	vector<KDBounds>* getTreeBounds(int p_idx);
	
protected:
private:
	enum EXTREME
	{
		RIGHT=-1,
		LEFT =1
	};

	struct Tri
	{
		int m_ids[3];
	};

	struct Triparam
	{
		int m_faceId;
		Tri m_tri;
	};

	void subdivide(unsigned int p_treeId, KDNode& p_node, vector<Tri>* p_tris, int p_dimsz, int p_dim, int p_idx, const glm::vec3& pos, const glm::vec3& parentSize);

	bool triIntersectNode(const Triparam& p_tri, const glm::vec3& pos, const glm::vec3& parentSize);

	float findOptimalSplitPos(KDNode& p_node, vector<Tri>* p_tris, const KDAxisMark& p_axis, const glm::vec3& p_currentSize, const glm::vec3& p_currentPos);

	void getTriangleExtents(const Tri& p_triRef, glm::vec3& p_outTriangleExtentsMax, glm::vec3& p_outTriangleExtentsMin);

	float getExtreme(const glm::vec3& p_triangleExtentsMax, const glm::vec3& p_triangleExtentsMin, const glm::vec3& p_axis, EXTREME p_side);

	float calculatecost(const KDNode& p_node, vector<Tri>* p_tris, float p_splitpos, const glm::vec3& p_axis, const glm::vec3& p_currentSize, const glm::vec3& p_currentPos);

	void calculatePrimitiveCount(const KDNode& p_node, vector<Tri>* p_tris,const glm::vec3& p_leftBox,const glm::vec3& p_rightBox, const glm::vec3& p_leftBoxPos, const glm::vec3& p_rightBoxPos, int& p_outLeftCount, int& p_outRightCount);

	float calculateArea( glm::vec3& p_extents);

	void getChildVoxelsMeasurement(float p_inSplitpos, const glm::vec3& p_axis, const glm::vec3& p_inParentSize, 
								   glm::vec3& p_outLeftSz, glm::vec3& p_outRightSz);

	glm::vec3 entrywiseMul(const glm::vec3& p_a, const glm::vec3& p_b);

	int addTree(vector<KDNode>* p_tree, vector<KDLeaf>* p_leafList, vector<int>* p_leafDataList,vector<KDBounds>* p_boundsList);

	//void clearTempStack();

	// set traversal/intersection cost vars
	float m_traversalCost;
	float m_intersectionCost;


	// Storage
	vector<vector<KDNode>*> m_trees;
	vector<vector<KDLeaf>*> m_leafLists;
	vector<vector<int>*> m_leafDataLists;
	vector<vector<KDBounds>*> m_treeBounds;
	// Temp
	//stack<vector<Tri>*>* m_tempTriListStack;
	glm::vec3* m_tempVertexList;
	glm::vec3* m_tempNormalsList;
};