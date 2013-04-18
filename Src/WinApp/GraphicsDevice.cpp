#include "GraphicsDevice.h"
#include "GraphicsException.h"

GraphicsDevice::GraphicsDevice( HWND p_hWnd, int p_width, int p_height, bool p_windowMode )
{
	m_width=p_width;
	m_height=p_height;
	m_windowMode = p_windowMode;
}

void GraphicsDevice::clearRenderTargets()
{

}

void GraphicsDevice::flipBackBuffer()
{

}

void GraphicsDevice::updateResolution( int p_width, int p_height )
{
	m_width = p_width;
	m_height = p_height;
	m_deviceContext->OMSetRenderTargets(0, 0, 0);

	releaseBackBuffer();
	//m_deferredRenderer->releaseRenderTargetsAndDepthStencil();

	HRESULT hr;
	// Resize the swap chain
	hr = m_swapChain->ResizeBuffers(0, p_width, p_height, DXGI_FORMAT_R8G8B8A8_UNORM, 0);
	if(FAILED(hr))
		throw GraphicsException(hr,__FILE__,__FUNCTION__,__LINE__);

	initBackBuffer();
	fitViewport();

	// m_deferredRenderer->initRendertargetsAndDepthStencil( m_width, m_height );
}

void GraphicsDevice::setWindowMode( bool p_windowed )
{

}

void GraphicsDevice::initSwapChain( HWND p_hWnd )
{

}

void GraphicsDevice::initHardware()
{

}

void GraphicsDevice::initBackBuffer()
{

}

void GraphicsDevice::releaseBackBuffer()
{
	SAFE_RELEASE( m_backBuffer );
}

