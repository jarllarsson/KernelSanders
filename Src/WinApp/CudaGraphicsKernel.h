#pragma once

#include "IKernelHandler.h"

// =======================================================================================
//                                   CudaGraphicsKernel
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Kernel handler for cuda kernel with graphics resources
///        
/// # CudaGraphicsKernel
/// 
/// 21-4-2013 Jarl Larsson
///---------------------------------------------------------------------------------------

class CudaGraphicsKernel : IKernelHandler
{
public:
	CudaGraphicsKernel();
	virtual ~CudaGraphicsKernel();


	virtual void SetPerKernelArgs();

	virtual bool SetPerFrameArgs();

	virtual void Execute(float dt);
protected:
private:


};