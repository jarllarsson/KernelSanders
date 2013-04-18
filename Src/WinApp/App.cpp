#include "App.h"
#include "Context.h"
#include "ContextException.h"
#include "DebugPrint.h"

App::App( HINSTANCE p_hInstance )
{
	try
	{
		m_context = new Context(p_hInstance,"Test",600,400);
	}
	catch (ContextException& e)
	{
		DEBUGWARNING((e.what()));
	}
}

App::~App()
{
	delete m_context;
}

void App::run()
{
	// Set up windows timer
	__int64 countsPerSec = 0;
	__int64 currTimeStamp = 0;
	QueryPerformanceFrequency((LARGE_INTEGER*)&countsPerSec);
	double secsPerCount = 1.0f / (float)countsPerSec;

	double dt = 0.0f;
	__int64 m_prevTimeStamp = 0;

	QueryPerformanceCounter((LARGE_INTEGER*)&m_prevTimeStamp);
	QueryPerformanceCounter((LARGE_INTEGER*)&currTimeStamp);

	MSG msg = {0};

	// secondary run variable
	// lets non-context systems quit the program
	bool run=true;

	while (!m_context->closeRequested() && run)
	{
		if( PeekMessage( &msg, NULL, 0, 0, PM_REMOVE) )
		{
			TranslateMessage( &msg );
			DispatchMessage( &msg );
		}
		else
		{
			QueryPerformanceCounter((LARGE_INTEGER*)&currTimeStamp);
			dt = (currTimeStamp - m_prevTimeStamp) * secsPerCount;

			m_prevTimeStamp = currTimeStamp;

			DEBUGPRINT((("\n"+toString(dt)).c_str())); 

			// dt = clamp(dt,0.0,DTCAP);

		}
	}
}
