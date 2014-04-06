#include "IKernelHandler.h"

IKernelHandler::IKernelHandler(MeasurementBin* p_measurer)
{
	m_doMeasurements=false;
	m_measurments=p_measurer;
}

IKernelHandler::~IKernelHandler()
{

}
