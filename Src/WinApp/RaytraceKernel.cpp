#include "RaytraceKernel.h"
#include <DebugPrint.h>
#include <ToString.h>


extern "C"
{
	void RunRaytraceKernel(void* p_cb,void* p_colorArray,
		int width, int height, int pitch,
		void* p_verts,void* p_norms,unsigned int p_numVerts,
		void* p_indices,unsigned int p_numIndices,
		void* p_kdExtents, void* p_kdPos,
		void* p_tris, unsigned int p_numTris,
		void* p_nodes, void* p_leaflist, void* p_nodeIndices,
		unsigned int p_numNodes,unsigned int p_numLeaves,unsigned int p_numNodeIndices);
}

RaytraceKernel::RaytraceKernel() : IKernelHandler()
{

}

RaytraceKernel::~RaytraceKernel()
{

}

void RaytraceKernel::SetPerKernelArgs()
{

}

void RaytraceKernel::Execute( KernelData* p_data, float p_dt )
{

	RaytraceKernelData* blob = static_cast<RaytraceKernelData*>(p_data);
	RaytraceConstantBuffer* constantBuffer = blob->m_cb;
	int width = blob->m_width;
	int height = blob->m_height;
	size_t pitch = *blob->m_pitch;	
	// scene
	HScene* scene = blob->m_hostScene;
	unsigned int numverts=scene->meshVerts.size();
	unsigned int numindices=scene->meshIndices.size();
	unsigned int numtris=scene->tri.size();
	//
	unsigned int numKDnodes  =scene->KDnode.size();
	unsigned int numKDleaves =scene->KDleaves.size();
	unsigned int numKDindices=scene->KDindices.size();
	glm::vec3 kdBoundsMin(-100.0f,-100.0f,-100.0f);//=scene->KDboundsMin;
	glm::vec3 kdBoundsMax(100.0f,100.0f,100.0f);//=scene->KDboundsMax;

	// Map render textures
	cudaStream_t stream = 0;
	const int resourceCount = 1; // only use color for now
		//(int)blob->m_textureResource->size();
	cudaError_t res = cudaGraphicsMapResources(resourceCount, &blob->m_textureResource/*[0]*/, stream);
	KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);
	// Get pointers
	res = cudaGraphicsSubResourceGetMappedArray(&blob->m_textureView, blob->m_textureResource/*[0]*/, 0, 0);
	KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);

	// Get scene objects
	void* verts = NULL;
	void* norms = NULL;
	void* indices = NULL;
	void* tris = NULL;
	void** devVerts=blob->m_vertsLinearMemDeviceRef;
	void** devNorms=blob->m_normsLinearMemDeviceRef;
	void** devIndices=blob->m_indicesLinearMemDeviceRef;
	void** devTris=blob->m_trisLinearMemDeviceRef;

	void* KDnodes = NULL;
	void* KDleaves = NULL;
	void* KDindices = NULL;
	void** devKDNodes = blob->m_nodesLinearMemDeviceRef;
	void** devKDLeaves = blob->m_nodeLeavesLinearMemDeviceRef;
	void** devKDIndices = blob->m_nodeIndicesLinearMemDeviceRef;

	// Vertices and indices
	if (scene->isDirty(HScene::MESH)) // Mesh is updated
	{
		verts=reinterpret_cast<void*>(&scene->meshVerts[0]); // get verts as void* array
		norms=reinterpret_cast<void*>(&scene->meshNorms[0]);
		indices=reinterpret_cast<void*>(&scene->meshIndices[0]);
		//
		if (devVerts!=NULL && *devVerts!=NULL &&
			devIndices!=NULL && *devIndices!=NULL) 
		{
			res = cudaFree(*devVerts);
			KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);
			res = cudaFree(*devIndices);
			KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);
		}		
		if (devNorms!=NULL && *devNorms!=NULL) 
		{
			res = cudaFree(*devNorms);
			KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);
		}

		if (scene->KDnode.size()>0) KDnodes  =reinterpret_cast<void*>(&scene->KDnode[0]);
		if (scene->KDleaves.size()>0) KDleaves =reinterpret_cast<void*>(&scene->KDleaves[0]);
		if (scene->KDindices.size()>0) KDindices=reinterpret_cast<void*>(&scene->KDindices[0]);
		if (devKDNodes!=NULL && *devKDNodes!=NULL &&
			devKDLeaves!=NULL && *devKDLeaves!=NULL &&
			devKDIndices!=NULL && *devKDIndices!=NULL) 
		{
			res = cudaFree(*devKDNodes);
			KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);
			res = cudaFree(*devKDLeaves);
			KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);
			res = cudaFree(*devKDIndices);
			KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);
		}
		// Copy new data to device array
		res = cudaMalloc((void**)devVerts, sizeof(glm::vec3) * numverts);
		KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);
		res = cudaMemcpy((void*)*devVerts, verts, sizeof(glm::vec3) * numverts, cudaMemcpyHostToDevice);
		KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);
		//
		res = cudaMalloc((void**)devNorms, sizeof(glm::vec3) * numverts);
		KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);
		res = cudaMemcpy((void*)*devNorms, norms, sizeof(glm::vec3) * numverts, cudaMemcpyHostToDevice);
		KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);
		//
		res = cudaMalloc((void**)devIndices, sizeof(unsigned int) * numindices);
		KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);
		res = cudaMemcpy((void*)*devIndices, indices, sizeof(unsigned int)*numindices, cudaMemcpyHostToDevice);
		KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);
		//
		// KD data ---------------------------------------------------------------------------------------------
		res = cudaMalloc((void**)devKDNodes, sizeof(glm::vec3) * numKDnodes);
		KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);
		res = cudaMemcpy((void*)*devKDNodes, KDnodes, sizeof(glm::vec3) * numKDnodes, cudaMemcpyHostToDevice);
		KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);
		//
		res = cudaMalloc((void**)devKDLeaves, sizeof(glm::vec3) * numKDleaves);
		KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);
		res = cudaMemcpy((void*)*devKDLeaves, KDleaves, sizeof(glm::vec3) * numKDleaves, cudaMemcpyHostToDevice);
		KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);
		//
		res = cudaMalloc((void**)devKDIndices, sizeof(unsigned int) * numKDindices);
		KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);
		res = cudaMemcpy((void*)*devKDIndices, KDindices, sizeof(unsigned int)*numKDindices, cudaMemcpyHostToDevice);
		KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);

		scene->setDirty(HScene::MESH,false);
	}

	// Tris
	if (scene->isDirty(HScene::TRI)) // Tris are updated
	{
		tris=reinterpret_cast<void*>(&scene->tri[0]); // get triangles as void* array
		if (devTris!=NULL && *devTris!=NULL) 
		{
			res = cudaFree(*devTris);
			KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);
		}
		// Copy new data to device array
		// for constant mem res = cudaMemcpyToSymbol(geomTriangles, p_tris, p_numTris*sizeof(TriPart));
		res = cudaMalloc((void**)devTris, sizeof(HTriPart) * numtris);
		KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);
		res = cudaMemcpy((void*)*devTris, tris, sizeof(HTriPart) * numtris, cudaMemcpyHostToDevice);
		KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);
		
		scene->setDirty(HScene::TRI,false);
	}

	// Run the kernel
	RunRaytraceKernel(reinterpret_cast<void*>(constantBuffer),
					  blob->m_textureLinearMemDevice,
					  width,height,(int)pitch,					  
					  *devVerts,*devNorms,numverts,
					  *devIndices,numindices,
					  (void*)&kdBoundsMin,(void*)&kdBoundsMax,
					  *devTris,numtris,
					  *devKDNodes,*devKDLeaves,*devKDIndices,
					  numKDnodes,numKDleaves,numKDindices); 
	// ---

	// copy color array to texture (device->device)
	res = cudaMemcpy2DToArray(
		blob->m_textureView, // dst array
		0, 0,    // offset
		blob->m_textureLinearMemDevice, pitch,       // src
		width*4*sizeof(float), height, // extent
		cudaMemcpyDeviceToDevice); // kind
	KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);

	// unmap textures
	res = cudaGraphicsUnmapResources(resourceCount, &blob->m_textureResource/*[0]*/, stream);
	KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);

}

