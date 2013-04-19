#pragma once
#include <windows.h>
#include <d3d11.h>

class ViewFactory;

// =======================================================================================
//                                     GraphicsDevice
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Class for all things graphics, allocation and rendering
///        
/// # GraphicsDevice
/// 
/// 18-4-2013 Jarl Larsson
///---------------------------------------------------------------------------------------

class GraphicsDevice
{
public:
	const static int RT0 = 0;
	const static int RT1 = 1;
	const static int DEPTH_IDX = 10;
	enum GBufferChannel {
		INVALID	= -1,
		DIFFUSE		= RT0,				// R, G, B, (Something)
		NORMAL		= RT1,				// X, Y, Z, (Something)		
		COUNT,
		DEPTH		= DEPTH_IDX,		// Depth
	};

	GraphicsDevice(HWND p_hWnd, int p_width, int p_height, bool p_windowMode);
	virtual ~GraphicsDevice();

	// States
	// Clear render targets in a color
	void clearRenderTargets();								///< Clear all rendertargets
	void flipBackBuffer();									///< Fliparoo!

	void updateResolution( int p_width, int p_height );		///< Update resolution
	void setWindowMode(bool p_windowed);					///< Set window mode on/off
	void fitViewport();										///< Fit viewport to width and height

	// Mapping/Unmapping
	void mapGBuffer();
	void mapGBufferSlot(GBufferChannel p_slot);
	void mapDepth();

	void unmapGBuffer();
	void unmapGBufferSlot(GBufferChannel p_slot);
	void unmapDepth();
	void unmapAllBuffers();

protected:
private:
	// Initialisations
	void initSwapChain(HWND p_hWnd);
	void initHardware();	
	void initBackBuffer();
	void initDepthStencil();
	void initGBuffer();
	void initGBufferAndDepthStencil();
	// Releases
	void releaseBackBuffer();
	void releaseGBufferAndDepthStencil();

	// Members
	int m_height;
	int m_width;
	bool m_windowMode;

	// Factories
	ViewFactory* m_viewFactory;

	// D3D specific
	// device
	ID3D11Device*			m_device;
	ID3D11DeviceContext*	m_deviceContext;
	// swap chain
	DXGI_SWAP_CHAIN_DESC	m_swapChainDesc;
	IDXGISwapChain*			m_swapChain;
	D3D_FEATURE_LEVEL		m_featureLevel;
	// views
	ID3D11RenderTargetView*		m_backBuffer;
	ID3D11ShaderResourceView*	m_depthSrv;
	ID3D11DepthStencilView*		m_depthStencilView;
	ID3D11RenderTargetView*		m_gRtv[GBufferChannel::COUNT];
	ID3D11ShaderResourceView*	m_gSrv[GBufferChannel::COUNT];


};