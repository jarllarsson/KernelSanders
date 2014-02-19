#include "RaytraceKernel.h"
#include <DebugPrint.h>
#include <ToString.h>


extern "C"
{
	void RunRaytraceKernel(void* p_cb,void* colorArray,
		int width, int height, int pitch,
		void* p_verts,			unsigned int p_numVerts,
		unsigned int* p_indices,unsigned int p_numIndices,
		void* p_tris,			unsigned int p_numTris);
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
	unsigned int* indices = NULL;
	void* tris = NULL;
	void** devVerts=blob->m_vertsLinearMemDeviceRef;
	unsigned int** devIndices=blob->m_indicesLinearMemDeviceRef;
	void** devTris=blob->m_trisLinearMemDeviceRef;

	// Vertices and indices
	if (scene->isDirty(HScene::MESH)) // Mesh is updated
	{
		verts=reinterpret_cast<void*>(&scene->meshVerts[0]); // get verts as void* array
		indices=&scene->meshIndices[0];
		if (devVerts!=NULL && *devVerts!=NULL &&
			devIndices!=NULL && *devIndices!=NULL) 
		{
			res = cudaFree(*devVerts);
			KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);
			res = cudaFree(*devIndices);
			KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);
		}
		// Copy new data to device array
		res = cudaMalloc((void**)devVerts, sizeof(glm::vec3) * numverts);
		KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);
		res = cudaMemcpy((void*)*devVerts, verts, sizeof(glm::vec3) * numverts, cudaMemcpyHostToDevice);
		KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);
		//
		res = cudaMalloc((void**)devIndices, sizeof(int) * numindices);
		KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);
		res = cudaMemcpy((void*)*devIndices, verts, sizeof(int)*numindices, cudaMemcpyHostToDevice);
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
					  *devVerts,numverts,
					  *devIndices,numindices,
					  *devTris,numtris); 
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

