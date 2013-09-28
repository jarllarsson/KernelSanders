#include "App.h"
#include <DebugPrint.h>

#include <Context.h>
#include <ContextException.h>

#include <GraphicsDevice.h>
#include <GraphicsException.h>

#include "KernelDevice.h"
#include "KernelException.h""

#include <ValueClamp.h>
#include "TempController.h"


#include <SDL.h>


const double App::DTCAP=0.5;

App::App( HINSTANCE p_hInstance )
{
	int width=600,
		height=400;
	bool windowMode=true;
	// Context
	try
	{
		m_context = new Context(p_hInstance,"Kernel Sanders",width,height);
	}
	catch (ContextException& e)
	{
		DEBUGWARNING((e.what()));
	}	
	
	// Graphics
	try
	{
		m_graphicsDevice = new GraphicsDevice(m_context->getWindowHandle(),width,height,windowMode);
	}
	catch (GraphicsException& e)
	{
		DEBUGWARNING((e.what()));
	}

	// Kernels
	try
	{
		m_kernelDevice = new KernelDevice(m_graphicsDevice->getDevicePointer());
		m_kernelDevice->registerCanvas(m_graphicsDevice->getInteropCanvasHandle());
	}
	catch (KernelException& e)
	{
		DEBUGWARNING((e.what()));
	}

	// For now init SDL2 here, no video
	if (SDL_Init( SDL_INIT_EVENTS | SDL_INIT_JOYSTICK | SDL_INIT_GAMECONTROLLER )>0)
	{
		DEBUGWARNING((string("Unable to init SDL2!").c_str()));
	}
	//

	fpsUpdateTick=0.0f;
	m_controller = new TempController();
}

App::~App()
{	
	SDL_Quit();
	delete m_kernelDevice;
	delete m_graphicsDevice;
	delete m_context;

	delete m_controller;
}

void App::run()
{
	// Set up windows timer
	__int64 countsPerSec = 0;
	__int64 currTimeStamp = 0;
	QueryPerformanceFrequency((LARGE_INTEGER*)&countsPerSec);
	double secsPerCount = 1.0f / (float)countsPerSec;

	double dt = 0.0;
	double fps = 0.0f;
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
			//Event handler
			SDL_Event e;
			//Handle events on queue
			while( SDL_PollEvent( &e ) != 0 )
			{
				//User presses a key
				if( e.type == SDL_KEYDOWN )
				{
					//Select surfaces based on key press
					switch( e.key.keysym.sym )
					{
					case SDLK_ESCAPE:
						run=false;
						break;

// 					case SDLK_DOWN:
// 						gCurrentSurface = gKeyPressSurfaces[ KEY_PRESS_SURFACE_DOWN ];
// 						break;
// 
// 					case SDLK_LEFT:
// 						gCurrentSurface = gKeyPressSurfaces[ KEY_PRESS_SURFACE_LEFT ];
// 						break;
// 
// 					case SDLK_RIGHT:
// 						gCurrentSurface = gKeyPressSurfaces[ KEY_PRESS_SURFACE_RIGHT ];
// 						break;
// 
// 					default:
// 						gCurrentSurface = gKeyPressSurfaces[ KEY_PRESS_SURFACE_DEFAULT ];
// 						break;
					}
				}
			}


			// apply resizing on graphics device if it has been triggered by the context
			if (m_context->isSizeDirty())
			{
				pair<int,int> sz=m_context->getSize();
				m_graphicsDevice->updateResolution(sz.first,sz.second);
			}

			// Get Delta time
			QueryPerformanceCounter((LARGE_INTEGER*)&currTimeStamp);

			dt = (currTimeStamp - m_prevTimeStamp) * secsPerCount;
			fps = 1.0f/dt;
			
			dt = clamp(dt,0.0,DTCAP);
			m_prevTimeStamp = currTimeStamp;

			fpsUpdateTick-=(float)dt;
			if (fpsUpdateTick<=0.0f)
			{
				m_context->updateTitle((" | FPS: "+toString((int)fps)).c_str());
				//DEBUGPRINT((("\n"+toString(dt)).c_str())); 
				fpsUpdateTick=0.3f;
			}

			m_graphicsDevice->clearRenderTargets();									// Clear render targets

			// temp controller update code
			m_controller->setFovFromAngle(52.0f,m_graphicsDevice->getAspectRatio());
			m_controller->update(dt);

			// Run the devices
			// ---------------------------------------------------------------------------------------------
			m_kernelDevice->update((float)dt,m_controller);								// Update kernel data



			m_kernelDevice->executeKernelJob((float)dt,KernelDevice::J_RAYTRACEWORLD);		// Run kernels

			m_graphicsDevice->executeRenderPass(GraphicsDevice::P_COMPOSEPASS);		// Run passes
			m_graphicsDevice->flipBackBuffer();										// Flip!
			// ---------------------------------------------------------------------------------------------
		}
	}
}
