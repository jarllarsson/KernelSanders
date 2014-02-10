#include "ModelImporter.h"

ModelImporter::ModelImporter()
{

}

ModelImporter::~ModelImporter()
{
	for (int i=0;i<m_models.size();i++)
	{
		aiReleaseImport(m_models[i]->m_model);
		delete m_models[i];
	}
	aiDetachAllLogStreams();
}

int ModelImporter::loadFile( const char* p_path )
{
	// we are taking one of the postprocessing presets to avoid
	// spelling out 20+ single postprocessing flags here.
	const aiScene* tscene = aiImportFile(p_path,aiProcessPreset_TargetRealtime_MaxQuality);

	if (tscene) 
	{
		ModelData* model=new ModelData;
		model->m_model=tscene;
		getBoundingBox(tscene, &model->m_sceneMin,&model->m_sceneMax);
		model->m_sceneCenter.x = (model->m_sceneMin.x + model->m_sceneMax.x) / 2.0f;
		model->m_sceneCenter.y = (model->m_sceneMin.y + model->m_sceneMax.y) / 2.0f;
		model->m_sceneCenter.z = (model->m_sceneMin.z + model->m_sceneMax.z) / 2.0f;
		m_models.push_back(model);
		int uid = static_cast<int>(m_models.size()-1);
		return uid;
	}
	return -1;
}

void ModelImporter::getBoundingBoxForNode( const aiScene* p_scene,
										   aiVector3D* p_min, aiVector3D* p_max, 
										   aiMatrix4x4* p_trafo,
										   const aiNode* p_nd)
{
	aiMatrix4x4 prev;
	unsigned int n = 0, t;
	const aiNode* nd = p_scene->mRootNode;
	if (p_nd!=NULL)
		nd=p_nd;

	prev = *p_trafo;
	aiMultiplyMatrix4(p_trafo,&nd->mTransformation);

	for (; n < nd->mNumMeshes; ++n) {
		const aiMesh* mesh = p_scene->mMeshes[nd->mMeshes[n]];
		for (t = 0; t < mesh->mNumVertices; ++t) {

			aiVector3D tmp = mesh->mVertices[t];
			aiTransformVecByMatrix4(&tmp,p_trafo);

			p_min->x = min(p_min->x,tmp.x);
			p_min->y = min(p_min->y,tmp.y);
			p_min->z = min(p_min->z,tmp.z);

			p_max->x = max(p_max->x,tmp.x);
			p_max->y = max(p_max->y,tmp.y);
			p_max->z = max(p_max->z,tmp.z);
		}
	}

	for (n = 0; n < nd->mNumChildren; ++n) {
		getBoundingBoxForNode(p_scene,p_min,p_max,p_trafo,nd->mChildren[n]);
	}
	*p_trafo = prev;
}

void ModelImporter::getBoundingBox(const aiScene* p_scene, aiVector3D* p_min,aiVector3D* p_max )
{
	aiMatrix4x4 trafo;
	aiIdentityMatrix4(&trafo);

	p_min->x = p_min->y = p_min->z =  1e10f;
	p_max->x = p_max->y = p_max->z = -1e10f;
	getBoundingBoxForNode(p_scene,p_min,p_max,&trafo);
}
