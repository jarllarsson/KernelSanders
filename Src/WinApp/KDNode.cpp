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

KDAxisMark KDNode::getAxis()
{
	return m_split;
}

bool KDNode::isLeaf()
{
	return m_isLeaf;
}

float KDNode::getPos()
{
	return m_position;
}

int KDNode::getIndexListIdxFromLeaf()
{
	return m_leftChildIdx;
}

void KDNode::init()
{
	m_isLeaf=false;
	m_position=0.0f;
	int m_leftChildIdx=-1;
}

int KDNode::getLeftChild()
{
	return m_leftChildIdx;
}

int KDNode::getRightChild()
{
	return m_leftChildIdx+1;
}

void KDNode::setToLeaf()
{
	m_isLeaf=true;
}

void KDNode::setLeafData( int p_leafDataIdx )
{
	m_leftChildIdx=p_leafDataIdx;
}
