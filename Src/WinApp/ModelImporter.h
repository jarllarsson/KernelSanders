#pragma once

#include <assimp/cimport.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <vector>
#include "KDTreeFactory.h"
#include <RawTexture.h>
#include <TextureParser.h>

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
		vector<unsigned int> m_trisIndices; // currently keeps a copy of tris-only index list
		glm::vec3 m_sceneMin, m_sceneMax, m_sceneCenter;
		int m_treeId, m_textureId;
	};

	ModelImporter();
	virtual ~ModelImporter();

	int loadFile(const char* p_path);

	ModelData* getStoredModel(int p_id);
	vector<KDNode>* getKDTree(int p_idx);
	vector<KDLeaf>* getKDLeafList(int p_idx);
	vector<int>* getKDLeafDataList(int p_idx);
	KDBounds getTreeBounds(int p_idx);
	vector<KDBounds>* getDebugNodeBounds(int p_idx);
	RawTexture* getModelTexture(int p_idx);

protected:
private:
	void getBoundingBoxForNode(const aiScene* p_scene, aiVector3D* p_min, aiVector3D* p_max, 
		aiMatrix4x4* p_trafo,
		const aiNode* p_nd=NULL);
	void getBoundingBox(const aiScene* p_scene, aiVector3D* p_min,aiVector3D* p_max);
	vector<ModelData*> m_models;
	vector<RawTexture*> m_textures;
	KDTreeFactory m_treeFactory;
	TextureParser m_textureParser;
};