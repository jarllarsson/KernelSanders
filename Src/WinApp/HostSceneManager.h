#pragma once
#include "HostScene.h"
// =======================================================================================
//                                      HostSceneManager
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Brief
///        
/// # HostSceneManager
/// 
/// 29-1-2014 Jarl Larsson
///---------------------------------------------------------------------------------------

class HostSceneManager
{
public:
	HostSceneManager();
	virtual ~HostSceneManager();
	HScene* getScenePtr();

	void addMeshTris(void* p_vec3ArrayXYZ, int p_vertCount,char32_t* p_indexArray, int p_iCount, void* p_vec3normalArrayXYZ,void* p_vec3uvArray);
	void addKDTree(KDBounds p_bounds, KDNode* p_nodeArray, int p_nodeCount, KDLeaf* p_leafArray, int p_leafCount, int* p_nodeIndices, int p_iCount);
	void addTexture(RawTexture* p_texture);
protected:
private:
	HScene m_scene;
};