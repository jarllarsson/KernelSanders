#pragma once
#include <vector>
#include <string>

using namespace std;

// =======================================================================================
//                                  Measurement Bin
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Brief
///        
/// # MeasurementBin
/// 
/// 6-4-2014 Jarl Larsson
///---------------------------------------------------------------------------------------

class MeasurementBin
{
public:
	float calculateMean();
	float calculateSTD();
	void finishRound();
	bool saveResults(string fileName);

	vector<float> m_measurements;
	double m_mean;
	double m_std;
	vector<double> m_allMeans;
	vector<double> m_allSTDs;
};