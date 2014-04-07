#define WIN32_LEAN_AND_MEAN
#ifndef _WINDEF_
struct HINSTANCE__;
typedef HINSTANCE__* HINSTANCE;
#endif

#include <vector>
#include "App.h"
#include <DebugPrint.h>
#include <ToString.h>
#include <vld.h>
#include <MeasurementBin.h>

using namespace std;


// =======================================================================================
//     
//							     Codename "Kernel Sanders"
//
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Main entry point
///        
/// # main
/// 
/// 17-4-2013 Jarl Larsson
///---------------------------------------------------------------------------------------

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPWSTR pCmdLine, int nCmdShow)
{
#if defined(_WIN32) && (defined(DEBUG) || defined(_DEBUG))
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif
	SetThreadAffinityMask(GetCurrentThread(), 1);

	/*
	Number of threads per thread group. 
	Screen resolution. 
	Trace depth. 
	Number of light sources. 
	Number of triangles. 
	*/

	MeasurementBin measurer;

	App* mainApp = new App(hInstance,&measurer);
	mainApp->run();
	delete mainApp;

	measurer.saveResults("../Output/ALL");

	return 0;
}