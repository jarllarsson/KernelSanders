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


// write to the stream
std::ostream& operator<<( std::ostream &stream,  const KDNode& v )
{
	//stream.write( (char*)&v.m_split.b_1, sizeof(v.m_split.b_1) );
	//stream.write( (char*)&v.m_split.b_2, sizeof(v.m_split.b_2) );
	//stream.write( (char*)&v.m_isLeaf, sizeof(v.m_isLeaf) );
	//stream.write( (char*)&v.m_position, sizeof(v.m_position) );
	//stream.write( (char*)&v.m_leftChildIdx, sizeof(v.m_leftChildIdx) );
	stream<<v.m_split.b_1 <<'\n';
	stream<<v.m_split.b_2 <<'\n';
	stream<<v.m_isLeaf <<'\n';
	stream<<v.m_position <<'\n';
	stream<<v.m_leftChildIdx <<'\n';
	return stream;
}

std::istream& operator>>( std::istream& stream, KDNode& v )
{
	//stream.read( (char*)&v.m_split.b_1, sizeof(v.m_split.b_1) );
	//stream.read( (char*)&v.m_split.b_2, sizeof(v.m_split.b_2) );
	//stream.read( (char*)&v.m_isLeaf, sizeof(v.m_isLeaf) );
	//stream.read( (char*)&v.m_position, sizeof(v.m_position) );
	//stream.read( (char*)&v.m_leftChildIdx, sizeof(v.m_leftChildIdx) );
	int i=0;
	stream>>i;
	v.m_split.b_1=i;
	stream>>v.m_split.b_2;
	stream>>v.m_isLeaf;
	stream>>v.m_position;
	stream>>v.m_leftChildIdx;
	return stream;
}


// write to the stream
std::ostream& operator<<( std::ostream &stream,  const KDLeaf& v )
{
	//stream.write( (char*)&v.m_offset, sizeof(v.m_offset) );
	//stream.write( (char*)&v.m_count, sizeof(v.m_count) );
	stream<<v.m_offset <<'\n';
	stream<<v.m_count <<'\n';
	return stream;
}

std::istream& operator>>( std::istream& stream, KDLeaf& v )
{
	//stream.read( (char*)&v.m_offset, sizeof(v.m_offset) );
	//stream.read( (char*)&v.m_count, sizeof(v.m_count) );
	stream>>v.m_offset;
	stream>>v.m_count;
	return stream;
}
