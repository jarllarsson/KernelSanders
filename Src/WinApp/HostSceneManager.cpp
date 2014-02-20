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

void HostSceneManager::addMeshTris( void* p_vec3ArrayXYZ, int p_vertCount,
								    unsigned int* p_indexArray, int p_iCount, 
								    void* p_vec3normalArrayXYZ )
{
	//int start=m_scene.tri.size();
	//int maxcount=start+p_vertCount;
		//min(start+p_vertCount,MAXTRIS);
	glm::vec3* arrV=reinterpret_cast<glm::vec3*>(p_vec3ArrayXYZ);
	glm::vec3* arrN=reinterpret_cast<glm::vec3*>(p_vec3normalArrayXYZ);
	m_scene.meshVerts.insert(m_scene.meshVerts.end(),arrV,arrV+p_vertCount);
	m_scene.meshIndices.insert(m_scene.meshIndices.end(),p_indexArray,p_indexArray+p_iCount);
	m_scene.meshNorms.insert(m_scene.meshNorms.end(),arrN,arrN+p_vertCount);
// 	for (int i=start;i<maxcount;i+=3)
// 	{
// 		HTriPart tri;
// 		for (int x=0;x<3;x++)
// 		{
// 			tri.vertices[x]=arr[i+x];
// 		}
// 		m_scene.tri.push_back(tri);
// 	}
}

