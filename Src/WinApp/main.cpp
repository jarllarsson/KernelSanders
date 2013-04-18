#define WIN32_LEAN_AND_MEAN
#ifndef _WINDEF_
class HINSTANCE__;
typedef HINSTANCE__* HINSTANCE;
#endif

#include <vector>
#include "App.h"
#include <DebugPrint.h>
#include <ToString.h>

using namespace std;

extern "C" void RunCubeKernel(vector<float>& data, vector<float>& result);

// =======================================================================================
//     
//							       Unnamed CUDA project
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


	vector<float> data(96); 
	for (int i = 0; i < 96; ++i) 
	{ 
		data[i] = static_cast<float>(i); 
	} 

	// Compute cube on the device 
	vector<float> cube(96); 
	RunCubeKernel(data, cube); 

	// Print out results 
	DEBUGPRINT(("\n\nCube kernel results:\n")); 

	for (int i = 0; i < 96; ++i) 
	{ 
		DEBUGPRINT((toString(cube[i]).c_str())); 
	} 

	
	App myApp(hInstance);
	myApp.run();

	return 0;
}