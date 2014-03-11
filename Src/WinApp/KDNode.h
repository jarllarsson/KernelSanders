#pragma once

#include "KDAxisMark.h"

// =======================================================================================
//                                      KDNode
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Brief
///        
/// # Node
/// 
/// 11-3-2014 Jarl Larsson
///---------------------------------------------------------------------------------------



class KDNode
{
public:
	KDNode();
	KDNode(int p_splitDim);
	~KDNode();

	KDAxisMark getAxis() const;
	bool isLeaf() const;
	float getPos() const;
	// Get id to index list if leaf
	int getIndexListIdxFromLeaf() const;
	// Other child ids if ordinary
	int getLeftChild() const;
	int getRightChild() const;

	void setAxis(KDAxisMark p_split);
	void setLeftChild(int p_idx);
	void setToLeaf();
	void setLeafData(int p_leafDataIdx);

protected:
private:
	void init();

	KDAxisMark m_split;
	bool m_isLeaf;
	float m_position;
	int m_leftChildIdx; // right child is always left child+1
						// and if leaf then this is the index to
						// to array of indices
};

struct KDLeaf
{
	int m_indices[3*3];
};