#pragma once
#include <windows.h>
#include <d3d11.h>

#define SAFE_RELEASE(x) if( x ) { (x)->Release(); (x) = NULL; }
#define SAFE_DELETE(x) if( x ) { delete (x); (x) = NULL; }

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


	GraphicsDevice(HWND p_hWnd, int p_width, int p_height, bool p_windowMode);
	virtual ~GraphicsDevice();

	// States
	// Clear render targets in a color
	void clearRenderTargets();								///< Clear all rendertargets
	void flipBackBuffer();									///< Fliparoo!

	void updateResolution( int p_width, int p_height );		///< Update resolution
	void setWindowMode(bool p_windowed);					///< Set window mode on/off
	void fitViewport();										///< Fit viewport to width and height

protected:
private:
	// Initialisations
	void initSwapChain(HWND p_hWnd);
	void initHardware();	
	void initBackBuffer();
	// Releases
	void releaseBackBuffer();

	// Members
	int m_height;
	int m_width;
	bool m_windowMode;

	// D3D specific
	// device
	ID3D11Device*			m_device;
	ID3D11DeviceContext*	m_deviceContext;
	// swap chain
	DXGI_SWAP_CHAIN_DESC	m_swapChainDesc;
	IDXGISwapChain*			m_swapChain;
	D3D_FEATURE_LEVEL		m_featureLevel;
	// render targets
	ID3D11RenderTargetView* m_backBuffer;
};