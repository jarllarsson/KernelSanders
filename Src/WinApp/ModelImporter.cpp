#include "ModelImporter.h"
#include <DebugPrint.h>

ModelImporter::ModelImporter()
{
	m_models=vector<ModelData*>();
	m_textures=vector<RawTexture*>();
}

ModelImporter::~ModelImporter()
{
	for (int i=0;i<m_models.size();i++)
	{
		aiReleaseImport(m_models[i]->m_model);
		delete m_models[i];
	}
	for (int i=0;i<m_textures.size();i++)
	{
		delete [] m_textures[i]->m_data;
		delete m_textures[i];
	}
	aiDetachAllLogStreams();
}

int ModelImporter::loadFile( const char* p_path )
{
	int result=-1;
	// we are taking one of the postprocessing presets to avoid
	// spelling out 20+ single postprocessing flags here.
	const aiScene* tscene = aiImportFile(p_path,
		aiProcessPreset_TargetRealtime_MaxQuality | aiProcess_GenUVCoords | aiProcess_TransformUVCoords);

	if (tscene!=NULL) 
	{
		ModelData* model=new ModelData;
		model->m_model=tscene;
		getBoundingBox(tscene, (aiVector3D*)&model->m_sceneMin,(aiVector3D*)&model->m_sceneMax);
		model->m_sceneCenter.x = (model->m_sceneMin.x + model->m_sceneMax.x) / 2.0f;
		model->m_sceneCenter.y = (model->m_sceneMin.y + model->m_sceneMax.y) / 2.0f;
		model->m_sceneCenter.z = (model->m_sceneMin.z + model->m_sceneMax.z) / 2.0f;
		// Create index list that is guaranteed to only contain triangle faces (list size multiple of 3)
		aiMesh* mmesh=tscene->mMeshes[0];
		int indexCount=0;
		for (unsigned int i=0;i<mmesh->mNumFaces;i++)
		{
			aiFace* f = &mmesh->mFaces[i];
			if (f->mNumIndices<3)
				DEBUGWARNING(("Found a poly with less than 3 vertices, ignoring..."));
			else if (f->mNumIndices>3)
				DEBUGWARNING(("Found a poly with more than 3 vertices, ignoring..."));
			else
			{
				model->m_trisIndices.insert(model->m_trisIndices.end(),f->mIndices,f->mIndices+3); // NOTE! Only support for triangles!
				indexCount+=3;
			}
		}		
		int texId=m_textures.size();
		// string has="has uvs";
		// if (!mmesh->HasTextureCoords(0))
		// {
		// 	has="no uvs";
		// }
		// DEBUGWARNING(( has.c_str() ));
		m_textures.push_back(m_textureParser.loadTexture("../Assets/bmo.png"));
		// build kd-tree representation of the index list for the mesh
// 		glm::vec3 extents(max(abs(model->m_sceneMax.x),abs(model->m_sceneMin.x)),
// 			max(abs(model->m_sceneMax.y),abs(model->m_sceneMin.y)),
// 			max(abs(model->m_sceneMax.z),abs(model->m_sceneMin.z)));
// 		extents -= model->m_sceneCenter;
		int treeId = -1;
		int treeLevel=4;

		treeId=m_treeFactory.loadKDTree(treeLevel,p_path);
		if (treeId==-1)
		{
			treeId=m_treeFactory.buildKDTree(treeLevel,(void*)mmesh->mVertices,(void*)mmesh->mNormals,mmesh->mNumVertices,
								  &model->m_trisIndices[0],model->m_trisIndices.size(),model->m_sceneMin,model->m_sceneMax);
		
			m_treeFactory.saveKDTree(treeLevel,p_path,treeId);
		}

		model->m_treeId=treeId;
		model->m_textureId=texId;
		m_models.push_back(model);
		result = static_cast<int>(m_models.size()-1);
	}
	else
	{
		DEBUGWARNING((aiGetErrorString()));
	}

	return result;
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
	//aiMultiplyMatrix4(p_trafo,&nd->mTransformation);

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

ModelImporter::ModelData* ModelImporter::getStoredModel( int p_id )
{
	if (p_id<m_models.size() && p_id>=0)
		return m_models[p_id];
	return NULL;
}

vector<KDNode>* ModelImporter::getKDTree( int p_idx )
{
	return m_treeFactory.getTree(p_idx);
}

vector<KDLeaf>* ModelImporter::getKDLeafList( int p_idx )
{
	return m_treeFactory.getLeafList(p_idx);
}

vector<int>* ModelImporter::getKDLeafDataList( int p_idx )
{
	return m_treeFactory.getLeafDataList(p_idx);
}

KDBounds ModelImporter::getTreeBounds( int p_idx )
{
	return m_treeFactory.getTreeBounds(p_idx);
}

vector<KDBounds>* ModelImporter::getDebugNodeBounds( int p_idx )
{
	return m_treeFactory.getDebugNodeBounds(p_idx);
}

RawTexture* ModelImporter::getModelTexture( int p_idx )
{
	if (p_idx<m_textures.size() && p_idx>=0)
		return m_textures[p_idx];
	return NULL;
}
