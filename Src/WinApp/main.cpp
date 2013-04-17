#define WIN32_LEAN_AND_MEAN
#ifndef _WINDEF_
class HINSTANCE__;
typedef HINSTANCE__* HINSTANCE;
#endif

#include "Context.h"
#include "ContextException.h"
#include "DebugPrint.h"

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

	delete context;

	return 0;
}