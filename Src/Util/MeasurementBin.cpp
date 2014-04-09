#include "MeasurementBin.h"
#include <fstream>

MeasurementBin::MeasurementBin()
{
	m_active=false;
	m_mean=0.0;
	m_std=0.0f;
}

float MeasurementBin::calculateMean()
{
	double accumulate=0.0;
	unsigned int count=m_measurements.size();
	for (int i=0;i<count;i++)
	{
		accumulate+=(double)m_measurements[i];
	}
	m_mean = accumulate/(double)count;
	return m_mean;
}

float MeasurementBin::calculateSTD()
{
	float mean=calculateMean();
	unsigned int count=m_measurements.size();
	double squaredDistsToMean=0.0;
	for (int i=0;i<count;i++)
	{
		double dist=(double)m_measurements[i]-(double)mean;
		squaredDistsToMean+=dist*dist;
	}
	double standardDeviation=sqrt(squaredDistsToMean/(double)count);
	m_std=standardDeviation;
	return standardDeviation;
}

bool MeasurementBin::saveResults( string fileName )
{
	if (m_active)
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
			outFile<<"\nRaw measurements\n";
			for (int i=0;i<m_measurements.size();i++)
			{
				outFile << m_measurements[i] << "\n";
			}

		}
		outFile.close();
	}

}

void MeasurementBin::finishRound()
{
	if (m_active)
	{
		calculateMean();
		calculateSTD();
		m_allMeans.push_back(m_mean);
		m_allSTDs.push_back(m_std);
	}
}

void MeasurementBin::activate()
{
	m_active=true;
}

void MeasurementBin::saveMeasurement( float p_measurement )
{
	if (m_active)
		m_measurements.push_back(p_measurement);
}

bool MeasurementBin::isActive()
{
	return m_active;
}
