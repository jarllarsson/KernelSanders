#pragma once

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
	IKernelHandler();
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
	/// Settings to be done on data on a per kernel/per frame basis
	/// \param dt
	/// \return bool
	///-----------------------------------------------------------------------------------
	virtual bool SetPerFrameArgs(float dt) {return true;}

	///-----------------------------------------------------------------------------------
	/// Execute kernel program
	/// \param dt
	/// \return void
	///-----------------------------------------------------------------------------------
	virtual void Execute(float dt)=0;


	///-----------------------------------------------------------------------------------
	/// Get the latest timing results
	/// \return double
	///-----------------------------------------------------------------------------------
	double GetLastExecTimeNS() const {return m_profilingExecTime;}
protected:

	// Debugging
	double m_profilingExecTime;

private:
};