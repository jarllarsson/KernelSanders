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

	void addMeshTris(void* p_vec3ArrayXYZ, int p_vertCount,unsigned int* p_indexArray, int p_iCount, void* p_vec3normalArrayXYZ);
protected:
private:
	HScene m_scene;
};