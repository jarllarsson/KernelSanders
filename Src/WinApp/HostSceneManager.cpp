#include "HostSceneManager.h"


HostSceneManager::HostSceneManager()
{
	for (int i=0;i<10;i++)
	{
		HTri tri;
		for (int x=0;x<3;x++)
		{
			tri.vertices[x]=glm::vec4((float)i+x*0.5f, 
			sin(0.5f+(float)i+x*0.01f) + ((i%2)*2-1)*(float)(x%2)*0.5f, 
			sin((float)(x+i)*0.5f)*-3.0f,0.0f);
		}
		m_scene.tri.push_back(tri);
	}
}

HScene* HostSceneManager::getScenePtr()
{
	return &m_scene;
}

