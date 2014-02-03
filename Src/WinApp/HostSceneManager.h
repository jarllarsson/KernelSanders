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
protected:
private:
	HScene m_scene;
};