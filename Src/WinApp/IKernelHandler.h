#pragma once

#include "KernelHelper.h"
#include <MeasurementBin.h>

struct KernelData
{

};

// =======================================================================================
//                                    IKernelHandler
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Base class for general kernel handling
///        
/// # IKernelHandler
/// 
/// 18-4-2013 Jarl Larsson
///---------------------------------------------------------------------------------------

class IKernelHandler
{
public:
	IKernelHandler(MeasurementBin* p_measurer);
	virtual ~IKernelHandler();
	/* For OpenCL or directCompute, loading and binding would be necessary
	void LoadProgram(const string& path);
	void BindToProgram();
	*/

	///-----------------------------------------------------------------------------------
	/// Settings to be done on data on on a per kernel basis
	/// \return void
	///-----------------------------------------------------------------------------------
	virtual void SetPerKernelArgs()=0;


	///-----------------------------------------------------------------------------------
	/// Execute kernel program
	/// \param dt
	/// \return void
	///-----------------------------------------------------------------------------------
	virtual void Execute(KernelData* p_data, float p_dt)=0;


	///-----------------------------------------------------------------------------------
	/// Get the latest timing results
	/// \return double
	///-----------------------------------------------------------------------------------
	//MeasurementBin* GetExecTimes() {return &m_measurments;}

	void ActivateMeasurements() {m_doMeasurements=true;}
	void DeactivateMeasurements() {m_doMeasurements=false;}

protected:

	// Debugging
	//double m_profilingExecTime;
	MeasurementBin* m_measurments;

	bool m_doMeasurements;
private:
};