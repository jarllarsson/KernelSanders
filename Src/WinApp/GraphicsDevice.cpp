#include "GraphicsDevice.h"
#include "GraphicsException.h"
#include "ViewFactory.h"


GraphicsDevice::GraphicsDevice( HWND p_hWnd, int p_width, int p_height, bool p_windowMode )
{
	m_width=p_width;
	m_height=p_height;
	m_windowMode = p_windowMode;

	// 1. init hardware
	initSwapChain(p_hWnd);
	initHardware();

	// 2.  init factories
	m_viewFactory = new ViewFactory(m_device,m_deviceContext);

	// 3. init views
	initBackBuffer();
	initGBufferAndDepthStencil();
}

GraphicsDevice::~GraphicsDevice()
{
	delete m_viewFactory;
	releaseGBufferAndDepthStencil();
	SAFE_RELEASE(m_device);
	SAFE_RELEASE(m_deviceContext);
	SAFE_RELEASE(m_swapChain);
	releaseBackBuffer();
}


void GraphicsDevice::clearRenderTargets()
{
	static float clearColor[4] = { 0.0f, 1.0f, 0.0f, 1.0f };

	// clear gbuffer
	unmapAllBuffers();
	unsigned int start = GBufferChannel::DIFFUSE;
	unsigned int end = GBufferChannel::COUNT;
	for( unsigned int i=start; i<end; i++ ) 
	{
		m_deviceContext->ClearRenderTargetView( m_gRtv[i], clearColor );
	}
	// clear backbuffer
	m_deviceContext->ClearRenderTargetView(m_backBuffer,clearColor);
	// clear depth stencil
	m_deviceContext->ClearDepthStencilView(m_depthStencilView, D3D11_CLEAR_DEPTH, 1.0f, 0);
}

void GraphicsDevice::flipBackBuffer()
{
	m_swapChain->Present( 0, 0);
}

void GraphicsDevice::updateResolution( int p_width, int p_height )
{
	m_width = p_width;
	m_height = p_height;
	m_deviceContext->OMSetRenderTargets(0, 0, 0);

	releaseBackBuffer();
	releaseGBufferAndDepthStencil();

	HRESULT hr;
	// Resize the swap chain
	hr = m_swapChain->ResizeBuffers(0, p_width, p_height, DXGI_FORMAT_R8G8B8A8_UNORM, 0);
	if(FAILED(hr))
		throw GraphicsException(hr,__FILE__,__FUNCTION__,__LINE__);

	initBackBuffer();
	fitViewport();

	initGBufferAndDepthStencil();
}

void GraphicsDevice::setWindowMode( bool p_windowed )
{
	m_windowMode=p_windowed;
}

void GraphicsDevice::fitViewport()
{
	D3D11_VIEWPORT vp;
	vp.TopLeftX = 0;
	vp.TopLeftY = 0;
	vp.Width	= static_cast<float>(m_width);
	vp.Height	= static_cast<float>(m_height);
	vp.MinDepth = 0.0f;
	vp.MaxDepth = 1.0f;
	m_deviceContext->RSSetViewports(1,&vp);
}


void GraphicsDevice::mapGBuffer()
{
	unsigned int start = GBufferChannel::DIFFUSE;
	unsigned int end = GBufferChannel::COUNT;
	for( unsigned int i=start; i<end; i++ ) 
	{
		m_deviceContext->PSSetShaderResources( i, 1, &m_gSrv[i] );
	}
}

void GraphicsDevice::mapGBufferSlot( GBufferChannel p_slot )
{
	unsigned int i = static_cast<unsigned int>(p_slot);
	m_deviceContext->PSSetShaderResources( i, 1, &m_gSrv[i] );
}

void GraphicsDevice::mapDepth()
{
	unsigned int i = static_cast<unsigned int>(GBufferChannel::DEPTH);
	m_deviceContext->PSSetShaderResources( i, 1, &m_depthSrv );
}


void GraphicsDevice::unmapGBuffer()
{
	ID3D11ShaderResourceView* nulz = NULL;
	unsigned int start = GBufferChannel::DIFFUSE;
	unsigned int end = GBufferChannel::COUNT;
	for( unsigned int i=start; i<end; i++ ) 
	{
		m_deviceContext->PSSetShaderResources( i, 1, &nulz );
	}
}


void GraphicsDevice::unmapGBufferSlot( GBufferChannel p_slot )
{
	ID3D11ShaderResourceView* nulz = NULL;
	m_deviceContext->PSSetShaderResources( static_cast<unsigned int>(p_slot), 1, &nulz );
}

void GraphicsDevice::unmapDepth()
{
	ID3D11ShaderResourceView* nulz = NULL;
	m_deviceContext->PSSetShaderResources( static_cast<unsigned int>(GBufferChannel::DEPTH), 
										   1, &nulz );
}

void GraphicsDevice::unmapAllBuffers()
{
	unmapGBuffer();
	unmapDepth();
}


void GraphicsDevice::initSwapChain( HWND p_hWnd )
{
	ZeroMemory( &m_swapChainDesc, sizeof(DXGI_SWAP_CHAIN_DESC) );
	m_swapChainDesc.BufferCount = 1;
	m_swapChainDesc.BufferDesc.Width = m_width;
	m_swapChainDesc.BufferDesc.Height = m_height;
	m_swapChainDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	m_swapChainDesc.BufferDesc.RefreshRate.Numerator = 60;
	m_swapChainDesc.BufferDesc.RefreshRate.Denominator = 1;
	m_swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	m_swapChainDesc.OutputWindow = p_hWnd;
	m_swapChainDesc.SampleDesc.Count = 1;
	m_swapChainDesc.SampleDesc.Quality = 0;
	m_swapChainDesc.Windowed = m_windowMode;
}

void GraphicsDevice::initHardware()
{
	HRESULT hr = S_OK;
	UINT createDeviceFlags = 0;
#ifdef _DEBUG
	createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif
	D3D_DRIVER_TYPE driverTypes[] = 
	{
		D3D_DRIVER_TYPE_HARDWARE,
		D3D_DRIVER_TYPE_REFERENCE,
	};
	UINT numDriverTypes = sizeof(driverTypes) / sizeof(driverTypes[0]);

	D3D_FEATURE_LEVEL featureLevelsToTry[] = {
		D3D_FEATURE_LEVEL_11_0,
		D3D_FEATURE_LEVEL_10_1,
		D3D_FEATURE_LEVEL_10_0
	};
	D3D_FEATURE_LEVEL initiatedFeatureLevel;

	int selectedDriverType = -1;

	for( UINT driverTypeIndex = 0; driverTypeIndex < numDriverTypes; driverTypeIndex++ )
	{
		D3D_DRIVER_TYPE driverType;
		driverType = driverTypes[driverTypeIndex];
		hr = D3D11CreateDeviceAndSwapChain(
			NULL,
			driverType,
			NULL,
			createDeviceFlags,
			featureLevelsToTry,
			ARRAYSIZE(featureLevelsToTry),
			D3D11_SDK_VERSION,
			&m_swapChainDesc,
			&m_swapChain,
			&m_device,
			&initiatedFeatureLevel,
			&m_deviceContext);

		if (hr == S_OK)
		{
			selectedDriverType = driverTypeIndex;
			break;
		}
	}
	if ( selectedDriverType > 0 )
		throw GraphicsException("Couldn't create a D3D Hardware-device, software render enabled."
		,__FILE__, __FUNCTION__, __LINE__);
}

void GraphicsDevice::initBackBuffer()
{
	m_viewFactory->constructBackbuffer(&m_backBuffer,m_swapChain);
}

void GraphicsDevice::initDepthStencil()
{
	m_viewFactory->constructDepthStencilViewAndShaderResourceView(&m_depthStencilView,
																  &m_depthSrv,
																  m_width,m_height);
}

void GraphicsDevice::initGBuffer()
{
	// Init all slots in gbuffer
	unsigned int start = GBufferChannel::DIFFUSE;
	unsigned int end = GBufferChannel::COUNT;
	for( unsigned int i=start; i<end; i++ ) 
	{
		m_viewFactory->constructRenderTargetViewAndShaderResourceView( &m_gRtv[i], 
																	   &m_gSrv[i], 
																	   m_width, m_height,
																	   DXGI_FORMAT_R8G8B8A8_UNORM );
	}
}


void GraphicsDevice::initGBufferAndDepthStencil()
{
	initDepthStencil();
	initGBuffer();
}

void GraphicsDevice::releaseBackBuffer()
{
	SAFE_RELEASE( m_backBuffer );
}

void GraphicsDevice::releaseGBufferAndDepthStencil()
{
	SAFE_RELEASE(m_depthStencilView);

	for (int i = 0; i < GBufferChannel::COUNT; i++)
	{
		SAFE_RELEASE(m_gRtv[i]);
		SAFE_RELEASE(m_gSrv[i]);
	}
}
