#include "MeasurementBin.h"
#include <fstream>

float MeasurementBin::calculateMean()
{
	float accumulate=0.0f;
	unsigned int count=m_measurements.size();
	for (int i=0;i<count;i++)
	{
		accumulate+=m_measurements[i];
	}
	m_mean = accumulate/(float)count;
	return m_mean;
}

float MeasurementBin::calculateSTD()
{
	float mean=calculateMean();
	unsigned int count=m_measurements.size();
	float squaredDistsToMean;
	for (int i=0;i<count;i++)
	{
		float dist=m_measurements[i]-mean;
		squaredDistsToMean+=dist*dist;
	}
	float standardDeviation=sqrt((double)squaredDistsToMean/(double)count);
	m_std=standardDeviation;
	return standardDeviation;
}

bool MeasurementBin::saveResults( string fileName )
{
	ofstream outFile;
	outFile.open( fileName+".csv" );

	if( !outFile.good() ) 
	{
		return false;
	} 
	else 
	{
		// Gfx settings
		outFile << "Mean time,Standard deviation"<<"\n";
		for (int i=0;i<m_allMeans.size();i++)
		{
			outFile << m_allMeans[i]<<","<<m_allSTDs[i]<< "\n";
		}

	}
	outFile.close();
}

void MeasurementBin::finishRound()
{
	calculateMean();
	calculateSTD();
	m_allMeans.push_back(m_mean);
	m_allSTDs.push_back(m_mean);
}
