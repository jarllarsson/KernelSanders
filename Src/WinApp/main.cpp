#define WIN32_LEAN_AND_MEAN
#ifndef _WINDEF_
class HINSTANCE__;
typedef HINSTANCE__* HINSTANCE;
#endif

#include "Context.h"
#include "ContextException.h"
#include "DebugPrint.h"
#include <vector>

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
	SetThreadAffinityMask(GetCurrentThread(), 1);

	Context* context;

	try
	{
		context = new Context(hInstance,"Test",600,400);
	}
	catch (ContextException& e)
	{
		DEBUGWARNING((e.what()));
	}

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

	delete context;

	return 0;
}