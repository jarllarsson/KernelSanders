#pragma once

#include <d3d11.h>

// =======================================================================================
//                                   GraphicsDeviceFactory
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Base class for factories that needs access to the device and devicecontext
///        
/// # GraphicsDeviceFactory
/// 
/// 18-4-2013 Jarl Larsson
///---------------------------------------------------------------------------------------

class GraphicsDeviceFactory
{
public:
	GraphicsDeviceFactory(ID3D11Device* p_device,ID3D11DeviceContext* p_deviceContext)
	{
		m_device=p_device; m_deviceContext=p_deviceContext;
	}
	virtual ~GraphicsDeviceFactory() {}
protected:
	ID3D11Device* m_device;
	ID3D11DeviceContext* m_deviceContext;
private:
};