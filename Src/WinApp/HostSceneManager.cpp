#include "HostSceneManager.h"
#include "RaytraceDefines.h"
#include <DebugPrint.h>
#include <ToString.h>


HostSceneManager::HostSceneManager()
{
	for (int i=0;i<3;i++)
	{
		HTriPart tri;
		for (int x=0;x<3;x++)
		{
			tri.vertices[x]=glm::vec3((float)i+x*0.5f, 
			sin(0.5f+(float)i+x*0.01f) + ((i%2)*2-1)*(float)(x%2)*0.5f, 
			sin((float)(x+i)*2.5f)*-3.0f);
		}
		m_scene.tri.push_back(tri);
	}
}

HostSceneManager::~HostSceneManager()
{

}

HScene* HostSceneManager::getScenePtr()
{
	return &m_scene;
}

void HostSceneManager::addTris( void* p_vec3ArrayXYZ, int p_count )
{
	int start=m_scene.tri.size();
	int maxcount=min(start+p_count,MAXTRIS);
	glm::vec3* arr=reinterpret_cast<glm::vec3*>(p_vec3ArrayXYZ);
	for (int i=start;i<maxcount;i+=3)
	{
		HTriPart tri;
		for (int x=0;x<3;x++)
		{
			tri.vertices[x]=arr[i+x];
		}
		m_scene.tri.push_back(tri);
	}
}

