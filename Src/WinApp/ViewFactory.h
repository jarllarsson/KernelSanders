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
	static void constructDepthStencilViewAndShaderResourceView(
									ID3D11DepthStencilView* p_outDsv, 
									ID3D11ShaderResourceView* p_outSrv, 
									int p_width,int p_height);
protected:
	void checkHRESULT(HRESULT p_res,const string& p_file,
		const string& p_function, int p_line);
private:
};