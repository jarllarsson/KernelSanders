#pragma once

#include "GraphicsDeviceFactory.h"
#include <string>
using namespace std;

// =======================================================================================
//                                      ViewFactory
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Factory for constructing various views
///        
/// # ViewFactory
/// 
/// 18-4-2013 Jarl Larsson
///---------------------------------------------------------------------------------------

class ViewFactory : public GraphicsDeviceFactory
{
public:
	ViewFactory(ID3D11Device* p_device,ID3D11DeviceContext* p_deviceContext) : GraphicsDeviceFactory(p_device,p_deviceContext){}
	void constructDepthStencilViewAndShaderResourceView( ID3D11DepthStencilView** p_outDsv, ID3D11ShaderResourceView** p_outSrv, int p_width,int p_height);
	void constructRenderTargetViewAndShaderResourceView( ID3D11RenderTargetView** p_outRtv, ID3D11ShaderResourceView** p_outSrv, int p_width,int p_height,DXGI_FORMAT p_format);
	void constructBackbuffer(ID3D11RenderTargetView** p_outRtv,
							 IDXGISwapChain* p_inSwapChain);
protected:
	void checkHRESULT(HRESULT p_res,const string& p_file,
		const string& p_function, int p_line);
private:
};