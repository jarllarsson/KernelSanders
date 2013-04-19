#include "App.h"
#include "Context.h"
#include "ContextException.h"
#include "DebugPrint.h"
#include "GraphicsDevice.h"
#include "GraphicsException.h"

App::App( HINSTANCE p_hInstance )
{
	int width=600,
		height=400;
	bool windowMode=true;
	try
	{
		m_context = new Context(p_hInstance,"Test",width,height);
	}
	catch (ContextException& e)
	{
		DEBUGWARNING((e.what()));
	}	
	
	try
	{
		m_graphicsDevice = new GraphicsDevice(m_context->getWindowHandle(),width,height,windowMode);
	}
	catch (GraphicsException& e)
	{
		DEBUGWARNING((e.what()));
	}
}

App::~App()
{
	delete m_context;
	delete m_graphicsDevice;
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
			// apply resizing on graphics device if it has been triggered by the context
			if (m_context->isSizeDirty())
			{
				pair<int,int> sz=m_context->getSize();
				m_graphicsDevice->updateResolution(sz.first,sz.second);
			}

			// Get Delta time
			QueryPerformanceCounter((LARGE_INTEGER*)&currTimeStamp);
			dt = (currTimeStamp - m_prevTimeStamp) * secsPerCount;

			m_prevTimeStamp = currTimeStamp;

			DEBUGPRINT((("\n"+toString(dt)).c_str())); 

			// Run the graphics device
			m_graphicsDevice->clearRenderTargets();

			// m_graphicsDevice->mapGBufferSlot(GraphicsDevice::DIFFUSE);

			// dt = clamp(dt,0.0,DTCAP);
			m_graphicsDevice->flipBackBuffer();
		}
	}
}
