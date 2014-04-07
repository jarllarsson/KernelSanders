#pragma once

#include "KDAxisMark.h"
#include <fstream>

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

#define KD_MIN_INDICES_IN_NODE 3*3 // Faces*3vert_ids
#define KD_EMPTY_LEAF -1

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

	void setAxis(const KDAxisMark& p_split);
	void setLeftChild(int p_idx);
	void setToLeaf();
	void setLeafData(int p_leafDataIdx);
	void setPos(float p_pos);

	friend std::ostream& operator<<( std::ostream& stream, const KDNode& v );
	friend std::istream& operator>>( std::istream& stream, KDNode& v );

protected:
private:
	void init();

	KDAxisMark m_split;
	int m_isLeaf;
	float m_position;
	int m_leftChildIdx; // right child is always left child+1
						// and if leaf then this is the index to
						// to array of indices
	// short leafcount SEE IF NEEDED
};

struct KDLeaf
{
	int m_offset; // offset in KD-sorted index list
	int m_count; // number of indices for this node

	friend std::ostream& operator<<( std::ostream& stream, const KDLeaf& v );
	friend std::istream& operator>>( std::istream& stream, KDLeaf& v );

};