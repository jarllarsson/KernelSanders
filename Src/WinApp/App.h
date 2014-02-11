#pragma once

#include <windows.h>

class Context;
class GraphicsDevice;
class KernelDevice;
class TempController;
class OISHelper;
class ModelImporter;
class HostSceneManager;

// =======================================================================================
//                                      App
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Brief
///        
/// # App
/// 
/// 18-4-2013 Jarl Larsson
///---------------------------------------------------------------------------------------

class App
{
public:
	App(HINSTANCE p_hInstance);
	virtual ~App();

	void run();
protected:
	Context* m_context;
	GraphicsDevice* m_graphicsDevice;
	KernelDevice* m_kernelDevice;
private:
	static const double DTCAP;
	float fpsUpdateTick;

	TempController* m_controller;
	HostSceneManager* m_sceneMgr; // the "world"
	ModelImporter*  m_modelImporter;
	OISHelper*		m_input;
};