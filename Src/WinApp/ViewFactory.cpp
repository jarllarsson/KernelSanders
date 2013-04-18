#include "ViewFactory.h"
#include "GraphicsException.h"

void ViewFactory::constructDepthStencilViewAndShaderResourceView(ID3D11DepthStencilView* p_outDsv, 
																 ID3D11ShaderResourceView* p_outSrv, 
																 int p_width,int p_height)
{
	HRESULT hr = S_OK;

	ID3D11Texture2D* depthStencilTexture;
	D3D11_TEXTURE2D_DESC depthStencilDesc;
	depthStencilDesc.Width = p_width;
	depthStencilDesc.Height = p_height;
	depthStencilDesc.MipLevels = 1;
	depthStencilDesc.ArraySize = 1;
	depthStencilDesc.Format = DXGI_FORMAT_R32_TYPELESS;
	depthStencilDesc.Usage = D3D11_USAGE_DEFAULT;
	depthStencilDesc.BindFlags = D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE;
	depthStencilDesc.SampleDesc.Count = 1;
	depthStencilDesc.SampleDesc.Quality = 0;
	depthStencilDesc.CPUAccessFlags = 0;
	depthStencilDesc.MiscFlags = 0;

	HRESULT createTexHr = m_device->CreateTexture2D( &depthStencilDesc,NULL,&depthStencilTexture);
	checkHRESULT( createTexHr, __FILE__, __FUNCTION__, __LINE__ );


	D3D11_DEPTH_STENCIL_VIEW_DESC depthStencilViewDesc;
	ZeroMemory(&depthStencilViewDesc, sizeof(depthStencilViewDesc));
	depthStencilViewDesc.Format = DXGI_FORMAT_D32_FLOAT;
	depthStencilViewDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
	depthStencilViewDesc.Texture2D.MipSlice = 0;

	HRESULT createDepthStencilViewHr = m_device->CreateDepthStencilView(
		depthStencilTexture, &depthStencilViewDesc, &p_outDsv );
	checkHRESULT( createDepthStencilViewHr, __FILE__, __FUNCTION__, __LINE__ );

	D3D11_SHADER_RESOURCE_VIEW_DESC shaderResourceDesc;
	ZeroMemory(&shaderResourceDesc,sizeof(shaderResourceDesc));
	shaderResourceDesc.Format = DXGI_FORMAT_R32_FLOAT;
	shaderResourceDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
	shaderResourceDesc.Texture2D.MostDetailedMip = 0;
	shaderResourceDesc.Texture2D.MipLevels = 1;

	HRESULT createDepthShaderResourceView = m_device->CreateShaderResourceView(
		depthStencilTexture, &shaderResourceDesc, &p_outSrv );
	checkHRESULT( createDepthShaderResourceView, __FILE__, __FUNCTION__, __LINE__ );


	depthStencilTexture->Release();
}

void ViewFactory::checkHRESULT( HRESULT p_res,const string& p_file,
							   const string& p_function, int p_line )
{
	if ( p_res != S_OK ) {
		throw GraphicsException( p_res, p_file, p_function, p_line );
	}
}

