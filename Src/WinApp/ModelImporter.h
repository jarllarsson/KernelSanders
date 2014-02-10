#pragma once

#include <assimp/cimport.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <vector>

using namespace std;

// =======================================================================================
//                                      ModelImporter
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Brief
///        
/// # ModelImporter
/// 
/// 10-2-2014 Jarl Larsson
///---------------------------------------------------------------------------------------

class ModelImporter
{
public:
	struct ModelData
	{
		const aiScene* m_model;
		aiVector3D m_sceneMin, m_sceneMax, m_sceneCenter;
	};

	ModelImporter();
	virtual ~ModelImporter();

	int loadFile(const char* p_path);

	ModelData* getStoredModel(int p_id);

protected:
private:
	void getBoundingBoxForNode(const aiScene* p_scene, aiVector3D* p_min, aiVector3D* p_max, 
		aiMatrix4x4* p_trafo,
		const aiNode* p_nd=NULL);
	void getBoundingBox(const aiScene* p_scene, aiVector3D* p_min,aiVector3D* p_max);
	vector<ModelData*> m_models;
};