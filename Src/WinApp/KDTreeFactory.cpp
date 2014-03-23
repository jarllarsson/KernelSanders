#pragma once

#include "KDTreeFactory.h"

// Defines for Box-Triangle test
// By Akenine-Möller
// http://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/code/tribox2.txt
// from
// http://www.flipcode.com/archives/Raytracing_Topics_Techniques-Part_7_Kd-Trees_and_More_Speed.shtml
#define X 0
#define Y 1
#define Z 2


#define DOT(v1,v2) (v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2])

#define FINDMINMAX(x0,x1,x2,min,max) \
	min = max = x0;   \
	if(x1<min) min=x1;\
	if(x1>max) max=x1;\
	if(x2<min) min=x2;\
	if(x2>max) max=x2;

int planeBoxOverlap(glm::vec3& normal,float d, float maxbox[3])
{
	int q;
	glm::vec3 vmin,vmax;
	for(q=X;q<=Z;q++)
	{
		if(normal[q]>0.0f)
		{
			vmin[q]=-maxbox[q];
			vmax[q]=maxbox[q];
		}
		else
		{
			vmin[q]=maxbox[q];
			vmax[q]=-maxbox[q];
		}
	}
	if(glm::dot(normal,vmin)+d>0.0f) return 0;
	if(glm::dot(normal,vmax)+d>=0.0f) return 1;

	return 0;
}

/*======================== X-tests ========================*/
#define AXISTEST_X01(a, b, fa, fb)             \
	p0 = a*v0[Y] - b*v0[Z];                    \
	p2 = a*v2[Y] - b*v2[Z];                    \
	if(p0<p2) {min=p0; max=p2;} else {min=p2; max=p0;} \
	rad = fa * boxhalfsize[Y] + fb * boxhalfsize[Z];   \
	if(min>rad || max<-rad) return 0;

#define AXISTEST_X2(a, b, fa, fb)              \
	p0 = a*v0[Y] - b*v0[Z];                    \
	p1 = a*v1[Y] - b*v1[Z];                    \
	if(p0<p1) {min=p0; max=p1;} else {min=p1; max=p0;} \
	rad = fa * boxhalfsize[Y] + fb * boxhalfsize[Z];   \
	if(min>rad || max<-rad) return 0;

/*======================== Y-tests ========================*/
#define AXISTEST_Y02(a, b, fa, fb)             \
	p0 = -a*v0[X] + b*v0[Z];                   \
	p2 = -a*v2[X] + b*v2[Z];                       \
	if(p0<p2) {min=p0; max=p2;} else {min=p2; max=p0;} \
	rad = fa * boxhalfsize[X] + fb * boxhalfsize[Z];   \
	if(min>rad || max<-rad) return 0;

#define AXISTEST_Y1(a, b, fa, fb)              \
	p0 = -a*v0[X] + b*v0[Z];                   \
	p1 = -a*v1[X] + b*v1[Z];                       \
	if(p0<p1) {min=p0; max=p1;} else {min=p1; max=p0;} \
	rad = fa * boxhalfsize[X] + fb * boxhalfsize[Z];   \
	if(min>rad || max<-rad) return 0;

/*======================== Z-tests ========================*/

#define AXISTEST_Z12(a, b, fa, fb)             \
	p1 = a*v1[X] - b*v1[Y];                    \
	p2 = a*v2[X] - b*v2[Y];                    \
	if(p2<p1) {min=p2; max=p1;} else {min=p1; max=p2;} \
	rad = fa * boxhalfsize[X] + fb * boxhalfsize[Y];   \
	if(min>rad || max<-rad) return 0;

#define AXISTEST_Z0(a, b, fa, fb)              \
	p0 = a*v0[X] - b*v0[Y];                \
	p1 = a*v1[X] - b*v1[Y];                    \
	if(p0<p1) {min=p0; max=p1;} else {min=p1; max=p0;} \
	rad = fa * boxhalfsize[X] + fb * boxhalfsize[Y];   \
	if(min>rad || max<-rad) return 0;


KDTreeFactory::KDTreeFactory()
{
	m_tempObjectsStack=new stack<vector<int>*>;
	m_traversalCost=0.3f;
	m_intersectionCost=1.0f;
}

KDTreeFactory::~KDTreeFactory()
{
	clearTempStack();
	delete m_tempObjectsStack;
	for (int i=0;i<m_trees.size();i++)
	{
		delete m_trees[i];
	}
	m_trees.clear();
	for (int i=0;i<m_leafLists.size();i++)
	{
		delete m_leafLists[i];
	}
	m_leafLists.clear();
}

int KDTreeFactory::calculateKDTree( void* p_vec3ArrayXYZ,void* p_normArrayXYZ, int p_vertCount, unsigned int* p_indexArray, int p_iCount )
{
	m_tempVertexList=(glm::vec3*)p_vec3ArrayXYZ;
	m_tempNormalsList=(glm::vec3*)p_normArrayXYZ;
	// Create root node

	// Build rest of tree using it
}



void KDTreeFactory::subdivide( KDNode& p_node, int p_dimsz, int p_dim, int p_idx, const glm::vec3& pos, const glm::vec3& parentSize )
{

}

bool KDTreeFactory::triIntersectNode( const Triparam& p_tri, const glm::vec3& pos, const glm::vec3& parentSize )
{
	glm::vec3* pv0=m_tempVertexList+(p_tri.m_ids[0]);
	glm::vec3* pv1=m_tempVertexList+(p_tri.m_ids[1]);
	glm::vec3* pv2=m_tempVertexList+(p_tri.m_ids[2]);
	glm::vec3* normal=m_tempNormalsList+(p_tri.m_faceId);
	float boxhalfsize[3]={parentSize.x*0.5f,parentSize.y*0.5f,parentSize.z*0.5f};
	// Use triangle-box overlap from Realtime rendering III pp.760-762
	// http://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/code/tribox2.txt
	// Also based on 
	// http://www.flipcode.com/archives/Raytracing_Topics_Techniques-Part_7_Kd-Trees_and_More_Speed.shtml
	glm::vec3 v0, v1, v2, e0, e1, e2;
	float min, max, p0, p1, p2, d, rad, fex, fey, fez;
	// Subtract box center from vertex position
	// to get triangle in aabb space
	v0 = *pv0 - pos;
	v1 = *pv1 - pos;
	v2 = *pv2 - pos;
	// Calc edges
	e0 = v1 - v0, e1 = v2 - v1, e2 = v0 - v2;
	/*  This is test 3. Is faster according to Möller */
	fex = fabsf( e0.x );
	fey = fabsf( e0.y );
	fez = fabsf( e0.z );
	AXISTEST_X01( e0.z, e0.y, fez, fey );
	AXISTEST_Y02( e0.z, e0.x, fez, fex );
	AXISTEST_Z12( e0.y, e0.x, fey, fex );
	fex = fabsf( e1.x );
	fey = fabsf( e1.y );
	fez = fabsf( e1.z );
	AXISTEST_X01( e1.z, e1.y, fez, fey );
	AXISTEST_Y02( e1.z, e1.x, fez, fex );
	AXISTEST_Z0 ( e1.y, e1.x, fey, fex );
	fex = fabsf( e2.x );
	fey = fabsf( e2.y );
	fez = fabsf( e2.z );
	AXISTEST_X2 ( e2.z, e2.y, fez, fey );
	AXISTEST_Y1 ( e2.z, e2.x, fez, fex );
	AXISTEST_Z12( e2.y, e2.x, fey, fex );
	// And here comes test 1
	/*  first test overlap in the {x,y,z}-directions */
	/*  find min, max of the triangle each direction, and test for overlap in */
	/*  that direction -- this is equivalent to testing a minimal AABB around */
	/*  the triangle against the AABB */
	FINDMINMAX( v0.x, v1.x, v2.x, min, max );
	if (min > boxhalfsize[0] || max < -boxhalfsize[0]) return false;
	FINDMINMAX( v0.x, v1.y, v2.y, min, max );
	if (min > boxhalfsize[1] || max < -boxhalfsize[1]) return false;
	FINDMINMAX( v0.z, v1.z, v2.z, min, max );
	if (min > boxhalfsize[2] || max < -boxhalfsize[2]) return false;
	// We already have all the normals
	d=glm::dot(*normal,v0);/* plane eq: normal.x+d=0 */
	if(!planeBoxOverlap(*normal,d,boxhalfsize)) return false;
	return true;
}

float KDTreeFactory::findOptimalSplitPos( KDNode& p_node, vector<int>* p_tris, const KDAxisMark& p_axis, const glm::vec3& p_currentSize, const glm::vec3& p_currentPos )
{
	float bestpos = 0.0f;
	float bestcost = FLT_MAX;
	glm::vec3 axis = p_axis.getVec();
	glm::vec3 aabbMax, aabbMin;
	unsigned int count=p_tris->size();
	int triangle[3];
	for (unsigned int i=0;i<count;i+=3)
	{
		for (unsigned int n=0;n<3;n++) triangle[n] = (*p_tris)[i+n]; // get face
		// Get aabb for triangle
		getTriangleExtents( triangle, aabbMax, aabbMin );
		float left_extreme = getExtreme(aabbMax, aabbMin, axis, EXTREME::LEFT);
		float right_extreme = getExtreme(aabbMax, aabbMin, axis, EXTREME::RIGHT);
		float cost = calculatecost(p_node, p_tris, left_extreme, axis, p_currentSize,p_currentPos);
		if (cost < bestcost)
		{
			bestcost = cost; bestpos = left_extreme;
		}
		//if (cost >= 1000000) Debug.Log("L!!! " + cost);
		cost = calculatecost(p_node, p_tris, right_extreme, axis, p_currentSize, p_currentPos);
		if (cost < bestcost)
		{
			bestcost = cost; bestpos = right_extreme;
		}
		//if (cost >= 1000000) Debug.Log("R!!! " + cost);
	}
	return bestpos;
}

void KDTreeFactory::getTriangleExtents( const int p_vertexIndices3[], glm::vec3& p_outTriangleExtentsMax, glm::vec3& p_outTriangleExtentsMin )
{
	glm::vec3 vert1=m_tempVertexList[p_vertexIndices3[0]];
	glm::vec3 vert2=m_tempVertexList[p_vertexIndices3[1]];
	glm::vec3 vert3=m_tempVertexList[p_vertexIndices3[2]];
	glm::vec3 extMax(max(vert3.x,max(vert1.x,vert2.x)),max(vert3.y,max(vert1.y,vert2.y)),max(vert3.z,max(vert1.z,vert2.z)));
	glm::vec3 extMin(min(vert3.x,min(vert1.x,vert2.x)),min(vert3.y,min(vert1.y,vert2.y)),min(vert3.z,min(vert1.z,vert2.z)));
	p_outTriangleExtentsMax=extMax;
	p_outTriangleExtentsMin=extMin;
}

float KDTreeFactory::getExtreme( const glm::vec3& p_triangleExtentsMax, const glm::vec3& p_triangleExtentsMin, const glm::vec3& p_axis, EXTREME p_side )
{
	// find the the point furthest away in one direction (axis*side)
	// Do this by masking extents values with axes.
	// Compare abs of masked min and max extents, and return the one of largest absolute.
	glm::vec3 pos = (float)p_side*entrywiseMul(p_triangleExtentsMax, p_axis); // mask
	float val1=pos.x + pos.y + pos.z;
	pos = (float)p_side*entrywiseMul(p_triangleExtentsMin, p_axis); // mask
	float val2=pos.x + pos.y + pos.z;
	float reval=val1;
	if (abs(val2)>abs(val1)) reval=val2;
	return reval;
}


float KDTreeFactory::calculatecost( const KDNode& p_node, vector<int>* p_tris, float p_splitpos, const glm::vec3& p_axis, const glm::vec3& p_currentSize, const glm::vec3& p_currentPos )
{
	glm::vec3 lsize;
	glm::vec3 rsize;
	getChildVoxelsMeasurement(p_splitpos, p_axis, p_currentSize,lsize, rsize);
	float leftarea = calculateArea(lsize);
	float rightarea = calculateArea(rsize);
	int leftcount, rightcount;
	glm::vec3 leftBoxPos = p_currentPos + 0.5f * entrywiseMul(lsize, p_axis);
	glm::vec3 rightBoxPos = p_currentPos - 0.5f * entrywiseMul(rsize, p_axis);
	calculatePrimitiveCount(p_node, p_tris, lsize,rsize,
		leftBoxPos, rightBoxPos,
		leftcount,rightcount);

	return m_traversalCost + m_intersectionCost * (leftarea * (float)leftcount + rightarea * (float)rightcount);
}

void KDTreeFactory::calculatePrimitiveCount( const KDNode& p_node, vector<int>* p_tris,
											 const glm::vec3& p_leftBox,const glm::vec3& p_rightBox, 
											 const glm::vec3& p_leftBoxPos, const glm::vec3& p_rightBoxPos, 
											 int& p_outLeftCount, int& p_outRightCount )
{
	p_outLeftCount=0;
	p_outRightCount=0;
	for (int i=0;i<p_tris->size();i+=3)
	{
		Triparam param = {i/3,(*p_tris)[i],(*p_tris)[i+1],(*p_tris)[i+2]};
		if (triIntersectNode(param, p_leftBoxPos, p_leftBox))
		{
			p_outLeftCount++;
		}
		if (triIntersectNode(param, p_rightBoxPos, p_rightBox))
		{
			p_outRightCount++;
		}
	}
}

float KDTreeFactory::calculateArea( glm::vec3& p_extents )
{
	return 2.0f * p_extents.x * p_extents.y * p_extents.z;
}

void KDTreeFactory::getChildVoxelsMeasurement( float p_inSplitpos, const glm::vec3& p_axis, const glm::vec3& p_inParentSize, 
											   glm::vec3& p_outLeftSz, glm::vec3& p_outRightSz )
{
	glm::vec3 offset = p_axis * p_inSplitpos;

	glm::vec3 splitH = 0.5f * p_axis;
	glm::vec3 lsize = (p_inParentSize - offset * 2.0f);
	lsize = lsize - entrywiseMul(lsize, splitH);
	glm::vec3 rsize = (p_inParentSize - offset * 2.0f);
	rsize = rsize - entrywiseMul(rsize, splitH);

	p_outLeftSz = lsize;
	p_outRightSz = rsize;
}

glm::vec3 KDTreeFactory::entrywiseMul( const glm::vec3& p_a, const glm::vec3& p_b )
{
	return glm::vec3(p_a.x*p_b.x,p_a.y*p_b.y,p_a.z*p_b.z);
}

int KDTreeFactory::addTree( vector<KDNode>* p_tree, vector<KDLeaf>* p_leafList )
{
	m_trees.push_back(p_tree);
	m_leafLists.push_back(p_leafList);
	return m_trees.size()-1;
}

void KDTreeFactory::clearTempStack()
{
	if (m_tempObjectsStack->size()>0)
	{
		for (int i=0;i<m_tempObjectsStack->size();i++)
		{
			vector<int>* elem = m_tempObjectsStack->top();
			delete elem;
			m_tempObjectsStack->pop();
		}
	}
}
