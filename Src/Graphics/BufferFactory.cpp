#include "BufferFactory.h"
#include "PVertex.h"
#include "D3DUtil.h"

BufferFactory::BufferFactory(ID3D11Device* p_device, ID3D11DeviceContext* p_deviceContext)
{
	m_device = p_device;
	m_deviceContext = p_deviceContext;
	m_elementSize = sizeof(float)*4;
}

BufferFactory::~BufferFactory()
{

}


Buffer<PVertex>* BufferFactory::createFullScreenQuadBuffer()
{
	PVertex mesh[]= {
		{ 1,	-1,	0},
		{ -1,	-1,	0},
		{ 1,	1,	0},

		{ -1, -1,	0},
		{ 1,	1,	0},
		{ -1,	1,	0}
	};
	Buffer<PVertex>* quadBuffer;

	// Create description for buffer
	BufferConfig::BUFFER_INIT_DESC bufferDesc;
	bufferDesc.ElementSize = sizeof(PVertex);
	bufferDesc.Usage = BufferConfig::BUFFER_DEFAULT;
	bufferDesc.NumElements = 6;
	bufferDesc.Type = BufferConfig::VERTEX_BUFFER;
	bufferDesc.Slot = BufferConfig::SLOT0;

	// Create buffer from config and data
	quadBuffer = new Buffer<PVertex>(m_device,m_deviceContext,&mesh[0],bufferDesc);
	SETDEBUGNAME((quadBuffer->getBufferPointer()),("fullscreenQuad_buffer"));

	return quadBuffer;
}


Buffer<unsigned int>* BufferFactory::createIndexBuffer( unsigned int* p_indices, 
												 unsigned int p_numberOfElements )
{	
	Buffer<unsigned int>* indexBuffer;

	// Create description for buffer
	BufferConfig::BUFFER_INIT_DESC indexBufferDesc;
	indexBufferDesc.ElementSize = sizeof(unsigned int);
	indexBufferDesc.Usage = BufferConfig::BUFFER_DEFAULT;
	indexBufferDesc.NumElements = p_numberOfElements;
	indexBufferDesc.Type = BufferConfig::INDEX_BUFFER;
	indexBufferDesc.Slot = BufferConfig::SLOT0;

	indexBuffer = new Buffer<unsigned int>(m_device,m_deviceContext, p_indices,
									 indexBufferDesc);

	return indexBuffer;
}
