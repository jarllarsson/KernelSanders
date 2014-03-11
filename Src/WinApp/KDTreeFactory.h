#pragma once

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
	int calculateKDTree(void* p_vec3ArrayXYZ, int p_vertCount,
						 unsigned int* p_indexArray, int p_iCount);

	
protected:
private:
};