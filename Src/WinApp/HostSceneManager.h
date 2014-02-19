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

	void addMeshTris(void* p_vec3ArrayXYZ, int p_vertCount,int* p_indexArray, int p_iCount);
protected:
private:
	HScene m_scene;
};