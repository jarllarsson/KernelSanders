#include "KDNode.h"

KDNode::KDNode()
{
	init();
}

KDNode::KDNode( int p_splitDim )
{
	init();
	m_split.setVec(p_splitDim);
}

KDNode::~KDNode()
{

}

KDAxisMark KDNode::getAxis() const
{
	return m_split;
}

bool KDNode::isLeaf() const
{
	return m_isLeaf==1;
}

float KDNode::getPos() const
{
	return m_position;
}

int KDNode::getIndexListIdxFromLeaf() const
{
	return m_leftChildIdx;
}

void KDNode::init()
{
	m_isLeaf=0;
	m_position=0.0f;
	int m_leftChildIdx=-1;
}

int KDNode::getLeftChild() const
{
	return m_leftChildIdx;
}

int KDNode::getRightChild() const
{
	return m_leftChildIdx+1;
}

void KDNode::setAxis( const KDAxisMark& p_split )
{
	m_split=p_split;
}

void KDNode::setToLeaf()
{
	m_isLeaf=1;
}

void KDNode::setLeafData( int p_leafDataIdx )
{
	m_leftChildIdx=p_leafDataIdx;
}

void KDNode::setPos( float p_pos )
{
	m_position=p_pos;
}

void KDNode::setLeftChild( int p_idx )
{
	m_leftChildIdx=p_idx;
}
