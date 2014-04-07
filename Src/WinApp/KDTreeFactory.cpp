#pragma once

#include <GraphicsDevice.h>
#include <ToString.h>
#include <DebugPrint.h>
#include "KDTreeFactory.h"
#include <Util.h>
#include <fstream>



// Defines for Box-Triangle test
// By Akenine-Möller
// http://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/code/tribox2.txt
// from
// http://www.flipcode.com/archives/Raytracing_Topics_Techniques-Part_7_Kd-Trees_and_More_Speed.shtml
#define X 0
#define Y 1
#define Z 2


#define DOT(v1,v2) (v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2])

#define FINDMINMAX(x0,x1,x2,trimin,trimax) \
	trimin = trimax = x0;   \
	if(x1<trimin) trimin=x1;\
	if(x1>trimax) trimax=x1;\
	if(x2<trimin) trimin=x2;\
	if(x2>trimax) trimax=x2;

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

glm::vec3 calcNormalOfFace( glm::vec3* verts[3], glm::vec3* normals[3] )
{
	glm::vec3 p0 = (*verts)[1] - (*normals)[0];
	glm::vec3 p1 = (*verts)[2] - (*normals)[0];
	glm::vec3 faceNormal = glm::cross( p0, p1 );

	glm::vec3 vertexNormal = (((*normals)[0]*(*normals)[1]+(*normals)[2])/3.0f);
	float dot = glm::dot( faceNormal, vertexNormal );

	return ( dot < 0.0f ) ? -faceNormal : faceNormal;
}

/*======================== X-tests ========================*/
#define AXISTEST_X01(a, b, fa, fb)             \
	p0 = a*v0[Y] - b*v0[Z];                    \
	p2 = a*v2[Y] - b*v2[Z];                    \
	if(p0<p2) {trimin=p0; trimax=p2;} else {trimin=p2; trimax=p0;} \
	rad = fa * boxhalfsize[Y] + fb * boxhalfsize[Z];   \
	if(trimin>rad || trimax<-rad) return 0;

#define AXISTEST_X2(a, b, fa, fb)              \
	p0 = a*v0[Y] - b*v0[Z];                    \
	p1 = a*v1[Y] - b*v1[Z];                    \
	if(p0<p1) {trimin=p0; trimax=p1;} else {trimin=p1; trimax=p0;} \
	rad = fa * boxhalfsize[Y] + fb * boxhalfsize[Z];   \
	if(trimin>rad || trimax<-rad) return 0;

/*======================== Y-tests ========================*/
#define AXISTEST_Y02(a, b, fa, fb)             \
	p0 = -a*v0[X] + b*v0[Z];                   \
	p2 = -a*v2[X] + b*v2[Z];                       \
	if(p0<p2) {trimin=p0; trimax=p2;} else {trimin=p2; trimax=p0;} \
	rad = fa * boxhalfsize[X] + fb * boxhalfsize[Z];   \
	if(trimin>rad || trimax<-rad) return 0;

#define AXISTEST_Y1(a, b, fa, fb)              \
	p0 = -a*v0[X] + b*v0[Z];                   \
	p1 = -a*v1[X] + b*v1[Z];                       \
	if(p0<p1) {trimin=p0; trimax=p1;} else {trimin=p1; trimax=p0;} \
	rad = fa * boxhalfsize[X] + fb * boxhalfsize[Z];   \
	if(trimin>rad || trimax<-rad) return 0;

/*======================== Z-tests ========================*/

#define AXISTEST_Z12(a, b, fa, fb)             \
	p1 = a*v1[X] - b*v1[Y];                    \
	p2 = a*v2[X] - b*v2[Y];                    \
	if(p2<p1) {trimin=p2; trimax=p1;} else {trimin=p1; trimax=p2;} \
	rad = fa * boxhalfsize[X] + fb * boxhalfsize[Y];   \
	if(trimin>rad || trimax<-rad) return 0;

#define AXISTEST_Z0(a, b, fa, fb)              \
	p0 = a*v0[X] - b*v0[Y];                \
	p1 = a*v1[X] - b*v1[Y];                    \
	if(p0<p1) {trimin=p0; trimax=p1;} else {trimin=p1; trimax=p0;} \
	rad = fa * boxhalfsize[X] + fb * boxhalfsize[Y];   \
	if(trimin>rad || trimax<-rad) return 0;


KDTreeFactory::KDTreeFactory()
{
	//m_tempTriListStack=new stack<vector<Tri>*>;
	m_traversalCost=10.0f;
		//0.3f;
	m_intersectionCost=1000.0f;
}

KDTreeFactory::~KDTreeFactory()
{
	//clearTempStack();
	//delete m_tempTriListStack;
	for (unsigned int i=0;i<m_trees.size();i++)
	{
		delete m_trees[i];
	}
	m_trees.clear();
	for (unsigned int i=0;i<m_leafLists.size();i++)
	{
		delete m_leafLists[i];
	}
	m_leafLists.clear();
	for (unsigned int i=0;i<m_leafDataLists.size();i++)
	{
		delete m_leafDataLists[i];
	}
	m_leafDataLists.clear();	
	m_treeBounds.clear();
	for (unsigned int i=0;i<m_debugTreeNodeBounds.size();i++)
	{
		delete m_debugTreeNodeBounds[i];
	}
	m_debugTreeNodeBounds.clear();
}

int KDTreeFactory::buildKDTree( int p_levels, void* p_vec3ArrayXYZ,void* p_normArrayXYZ, int p_vertCount, unsigned int* p_indexArray, int p_iCount, const glm::vec3& p_boundsMin, const glm::vec3& p_boundsMax )
{
	m_buildLevel=p_levels;
	m_tempVertexList=(glm::vec3*)p_vec3ArrayXYZ;
	m_tempNormalsList=(glm::vec3*)p_normArrayXYZ;
	vector<Tri> triList((Tri*)p_indexArray,(Tri*)p_indexArray+p_iCount/3);
	m_tempFaceNormalsList.clear();
	m_tempFaceNormalsList.assign(p_iCount/3,NULL);
	// Create root node
	KDNode root;
	//m_tempTriListStack->push(&triList);
	vector<KDNode>* tree=new vector<KDNode>/*(sc_treeListMaxSize)*/;
	vector<KDLeaf>*	leafList=new vector<KDLeaf>;
	vector<int>*	leafDataList=new vector<int>;
	vector<KDBounds>* debugNodeBoundsList=new vector<KDBounds>; // only for debug draw
	int treeId=addTree(tree,leafList,leafDataList,debugNodeBoundsList);
	//glm::vec3 offset(0.0f,0.0f,0.0f);
	//offset=p_boundsMin;

	glm::vec3 boxCenter=(p_boundsMin+p_boundsMax)*0.5f;
	glm::vec3 boxExt=(p_boundsMax-p_boundsMin);
	KDBounds boundsData={boxCenter,boxExt};
	m_tempRootPos=boxCenter;
	m_treeBounds.push_back(boundsData);
		
	__int64 countsPerSec = 0;
	QueryPerformanceFrequency((LARGE_INTEGER*)&countsPerSec);
	double secsPerCount = 1.0f / (float)countsPerSec;

	__int64 prevTimeStamp = 0;
	__int64 currTimeStamp = 0;
	QueryPerformanceCounter((LARGE_INTEGER*)&prevTimeStamp);

	/////////////////////////////////////////////////////////////////////////////////
	tree->push_back(KDNode());
	tree->push_back(root);
	subdivide(treeId, &triList, 0, 0, 1, boxCenter,boxExt,boxCenter,FLT_MAX); // start at 1	
	/////////////////////////////////////////////////////////////////////////////////

	QueryPerformanceCounter((LARGE_INTEGER*)&currTimeStamp);

	double timing=((currTimeStamp - prevTimeStamp) * secsPerCount);
	DEBUGWARNING(( (string("KD Tree build time for ")+toString(p_vertCount)+string(" vertices :")+toString(timing)+string(" seconds.")).c_str() ));

	for (int i=0;i<m_tempFaceNormalsList.size();i++)
	{
		SAFE_DELETE(m_tempFaceNormalsList[i]);
	}
	m_tempFaceNormalsList.clear();

	//m_tempObjectsStack.pop();
	//Debug.Log("fin stack: " + m_tempObjectsStack.Count);
	// Build rest of tree using it
	return treeId;
}



void KDTreeFactory::subdivide( unsigned int p_treeId, vector<Tri>* p_tris, int p_dimsz, int p_dim, int p_idx, const glm::vec3& pos, const glm::vec3& parentSize,const glm::vec3& p_splitOffset, float p_cost )
{
	//p_node.pos = pos;
	//p_node.size = parentSize;
	vector<KDNode>* tree = m_trees[p_treeId];
	KDNode p_node = (*tree)[p_idx];
	vector<KDLeaf>* leaflist = m_leafLists[p_treeId];
	vector<int>* leafdatalist = m_leafDataLists[p_treeId];
	vector<KDBounds>* debugboundslist = m_debugTreeNodeBounds[p_treeId];
	// Add debug information (for drawing wireframe boxes of tree)
	KDBounds nodeBounds={pos,parentSize};
	debugboundslist->push_back(nodeBounds);
	// End condition
	if (p_dimsz > m_buildLevel || p_tris->size() < KD_MIN_INDICES_IN_NODE/3/* || p_idx/ *<<1* />sc_treeListMaxSize/ *-2* /*/) 
	{
		int rem=(int)p_tris->size();
		//
		p_node.setToLeaf();
		if (rem>0)
		{// Generate leaf
			generateLeaf(p_treeId,&p_node,p_tris,leafdatalist,rem);
		}
		else
		{
			p_node.setLeafData(KD_EMPTY_LEAF);
		}
		// all changes made to node, add it to list
		(*tree)[p_idx]=p_node;
		return;
	}

	if (p_dim > 2) p_dim = 0;

	KDAxisMark splitPlane;
	splitPlane.setVec(p_dim);
	float costLeft = p_cost, costRight = p_cost, cost=FLT_MAX;
	float currentArea=calculateArea(parentSize);
	float splitpos = findOptimalSplitPos(p_tris, splitPlane,parentSize,currentArea,pos,costLeft,costRight,cost);
	float nosplitcost=m_intersectionCost*p_tris->size();
	// extra break condition to leaf, if too expensive to split
	if (/*splitpos!=0.0f && */cost>nosplitcost)
    {
		p_node.setToLeaf();
		generateLeaf(p_treeId,&p_node,p_tris,leafdatalist,(int)p_tris->size());
		(*tree)[p_idx]=p_node;
        return;
    }

	glm::vec3 split = splitPlane.getVec();
	glm::vec3 offset = split * splitpos;
	glm::vec3 currentOrigo = pos + offset;

	glm::vec3 lsize;
	glm::vec3 rsize;
	getChildVoxelsMeasurement(splitpos, split, parentSize,lsize, rsize);

	glm::vec3 leftBoxPos = currentOrigo + 0.5f * entrywiseMul(lsize, split);
	glm::vec3 rightBoxPos = currentOrigo - 0.5f * entrywiseMul(rsize, split);
	glm::vec3 leftBox = lsize;
	glm::vec3 rightBox = rsize;



	// Create children
	KDNode leftnode;
	KDNode rightnode; 
	tree->push_back(leftnode);
	int lnode=tree->size()-1;
	tree->push_back(rightnode);

	p_node.setLeftChild(lnode/*p_idx << 1*/);
	glm::vec3 tsplit(split.x,split.y,split.z);
	KDAxisMark transposedsplit=KDAxisMark(tsplit.x,tsplit.y,tsplit.z);
	p_node.setAxis(transposedsplit);
	glm::vec3 splitOffsetMask=entrywiseMul(p_splitOffset ,split);
	float splitposoffset=splitOffsetMask.x+splitOffsetMask.y+splitOffsetMask.z;

	float splitposW=splitposoffset+splitpos;
	p_node.setPos(splitposW);

	glm::vec3 nextLeftOffset = leftBoxPos/* + 1.0f * entrywiseMul(rsize, split)*/;
	glm::vec3 nextRightOffset= rightBoxPos/* + 2.0f * entrywiseMul(lsize, split)*/;

	// all changes made to node, add it to list
	(*tree)[p_idx]=p_node; 

	//
	vector<Tri> leftTris;
	vector<Tri> rightTris;
	//m_tempTriListStack->push(&rightTris);
	//m_tempTriListStack->push(&leftTris);
	//
	unsigned int count=p_tris->size();
	for (unsigned int i=0;i<count;i++)
	{
		Triparam param = {i,(*p_tris)[i]};
		if (triIntersectNode(param,leftBoxPos, leftBox*1.05f)) 
		{
			//p_node.m_objects.Remove(obj);
			leftTris.push_back(param.m_tri);
		}
		if (triIntersectNode(param, rightBoxPos, rightBox*1.05f))
		{
			//p_node.m_objects.Remove(obj);
			rightTris.push_back(param.m_tri);
		}
	}
	//Debug.Log("stack: "+m_tempTriListStack.Count);
	subdivide(p_treeId, &leftTris, p_dimsz + 1, p_dim + 1, p_node.getLeftChild(), leftBoxPos, leftBox, nextLeftOffset, costLeft); // power of two structure
	//m_tempTriListStack->pop();
	subdivide(p_treeId, &rightTris,p_dimsz + 1, p_dim + 1, p_node.getRightChild(), rightBoxPos, rightBox, nextRightOffset, costRight);
	//m_tempTriListStack->pop();
}

bool KDTreeFactory::triIntersectNode( const Triparam& p_tri, const glm::vec3& pos, const glm::vec3& parentSize )
{
	glm::vec3 pv0=(*(m_tempVertexList+(p_tri.m_tri.m_ids[0])));
	glm::vec3 pv1=(*(m_tempVertexList+(p_tri.m_tri.m_ids[1])));
	glm::vec3 pv2=(*(m_tempVertexList+(p_tri.m_tri.m_ids[2])));
	// If face normal doesn't exist, create it and cache it!
	//int facenormalIdx=(p_tri.m_faceId);
	//if (m_tempFaceNormalsList[facenormalIdx]==NULL)
	//{
	//	glm::vec3* nv0=m_tempNormalsList+(p_tri.m_tri.m_ids[0]);
	//	glm::vec3* nv1=m_tempNormalsList+(p_tri.m_tri.m_ids[1]);
	//	glm::vec3* nv2=m_tempNormalsList+(p_tri.m_tri.m_ids[2]);
	//	glm::vec3* v[]={pv0,pv1,pv2};
	//	glm::vec3* n[]={nv0,nv1,nv2};
	//	m_tempFaceNormalsList[facenormalIdx]=new glm::vec3(calcNormalOfFace(v,n));
	//}
	glm::vec3* nv0=m_tempNormalsList+(p_tri.m_tri.m_ids[0]);
	glm::vec3* nv1=m_tempNormalsList+(p_tri.m_tri.m_ids[1]);
	glm::vec3* nv2=m_tempNormalsList+(p_tri.m_tri.m_ids[2]);
	glm::vec3 normal( ((*nv0)+(*nv1)+(*nv2))/3.0f );
	//normal *= FLT_EPSILON;
		//m_tempFaceNormalsList[facenormalIdx];

	float boxhalfsize[3]={parentSize.x*0.5f,parentSize.y*0.5f,parentSize.z*0.5f};
	// Use triangle-box overlap from Realtime rendering III pp.760-762
	// http://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/code/tribox2.txt
	// Also based on 
	// http://www.flipcode.com/archives/Raytracing_Topics_Techniques-Part_7_Kd-Trees_and_More_Speed.shtml
	glm::vec3 v0, v1, v2, e0, e1, e2;
	float trimin, trimax, p0, p1, p2, d, rad, fex, fey, fez;
	// Subtract box center from vertex position
	// to get triangle in aabb space
	v0 = pv0 - pos;
	v1 = pv1 - pos;
	v2 = pv2 - pos;
	// pad extents of halsize
	boxhalfsize[0]+=max(max(abs(v0.x),abs(v1.x)),abs(v2.x))*0.5555f+FLT_EPSILON;
	boxhalfsize[1]+=max(max(abs(v0.y),abs(v1.y)),abs(v2.y))*0.5555f+FLT_EPSILON;
	boxhalfsize[2]+=max(max(abs(v0.z),abs(v1.z)),abs(v2.z))*0.5555f+FLT_EPSILON;
	// Calc edges
	e0 = v1 - v0, e1 = v2 - v1, e2 = v0 - v2;
	/*  This is test 3. Running it here is faster according to Möller */
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
	FINDMINMAX( v0.x, v1.x, v2.x, trimin, trimax );
	if (trimin > boxhalfsize[0] || trimax < -boxhalfsize[0]) return false;
	FINDMINMAX( v0.x, v1.y, v2.y, trimin, trimax );
	if (trimin > boxhalfsize[1] || trimax < -boxhalfsize[1]) return false;
	FINDMINMAX( v0.z, v1.z, v2.z, trimin, trimax );
	if (trimin > boxhalfsize[2] || trimax < -boxhalfsize[2]) return false;
	// We already have all the normals
	//CROSS(normal,e0,e1);
	normal=glm::cross(e0,e1);
	d=-glm::dot(normal,v0);/* plane eq: normal.x+d=0 */
	if(!planeBoxOverlap(normal,d,boxhalfsize)/* && !planeBoxOverlap(normal,-d,boxhalfsize)*/) return false;
	return true;
}

float KDTreeFactory::findOptimalSplitPos( vector<Tri>* p_tris, const KDAxisMark& p_axis, const glm::vec3& p_currentSize, float p_currentArea, const glm::vec3& p_currentPos,float& p_outCostLeft,float& p_outCostRight, float& p_outCost )
{
	float bestpos = 0.0f;
	float bestcost = FLT_MAX;
	float leftC = 0.0f, rightC = 0.0f;
	glm::vec3 axis = p_axis.getVec();
	glm::vec3 aabbMax, aabbMin;
	unsigned int count=p_tris->size();
	float left_extreme=0.0f;
	float right_extreme=0.0f;
	for (unsigned int i=0;i<count;i++)
	{
		// Get aabb for triangle
		getTriangleExtents( (*p_tris)[i], aabbMax, aabbMin );
		left_extreme = getExtreme(aabbMax, aabbMin, axis, p_currentPos, EXTREME::LEFT);
		right_extreme = getExtreme(aabbMax, aabbMin, axis, p_currentPos, EXTREME::RIGHT);
		//if (left_extreme<entrywiseMul(p_currentSize*0.5f,axis) )

		float cost = calculatecost( p_tris, left_extreme, axis, p_currentSize,p_currentArea, p_currentPos,leftC,rightC);
		if (cost < bestcost)
		{
			bestcost = cost; bestpos = left_extreme;
			p_outCostLeft=leftC;
			p_outCostRight=rightC;
		}

		//if (cost >= 1000000) Debug.Log("L!!! " + cost);
		cost = calculatecost(p_tris, right_extreme, axis, p_currentSize, p_currentArea, p_currentPos,leftC,rightC);
		if (cost < bestcost)
		{
			bestcost = cost; bestpos = right_extreme;
			p_outCostLeft=leftC;
			p_outCostRight=rightC;
		}
		//if (cost >= 1000000) Debug.Log("R!!! " + cost);
	}
	p_outCost=bestcost;
	return bestpos;
}

void KDTreeFactory::getTriangleExtents( const Tri& p_triRef, glm::vec3& p_outTriangleExtentsMax, glm::vec3& p_outTriangleExtentsMin )
{
	glm::vec3 vert1=m_tempVertexList[p_triRef.m_ids[0]];
	glm::vec3 vert2=m_tempVertexList[p_triRef.m_ids[1]];
	glm::vec3 vert3=m_tempVertexList[p_triRef.m_ids[2]];
	glm::vec3 extMax(max(vert3.x,max(vert1.x,vert2.x)),max(vert3.y,max(vert1.y,vert2.y)),max(vert3.z,max(vert1.z,vert2.z)));
	glm::vec3 extMin(min(vert3.x,min(vert1.x,vert2.x)),min(vert3.y,min(vert1.y,vert2.y)),min(vert3.z,min(vert1.z,vert2.z)));
	p_outTriangleExtentsMax=extMax;
	p_outTriangleExtentsMin=extMin;
}

// NOT WORKING!!!!
float KDTreeFactory::getExtreme( const glm::vec3& p_triangleExtentsMax, const glm::vec3& p_triangleExtentsMin, const glm::vec3& p_axis, const glm::vec3& p_parentPos, EXTREME p_side )
{
	// find the the point furthest away in one direction (axis*side)
	// Do this by masking extents values with axes.
	// Compare abs of masked min and max extents, and return the one of largest absolute.
	
	glm::vec3 ePos=/*p_parentPos+*/(p_triangleExtentsMin+p_triangleExtentsMax)*0.5f;
	glm::vec3 eExt=p_triangleExtentsMax-p_triangleExtentsMin;
	
	
	//glm::vec3 pos = (float)p_side*entrywiseMul(p_triangleExtentsMax, p_axis); // mask
	//float val1=pos.x + pos.y + pos.z;
	//pos = (float)p_side*entrywiseMul(p_triangleExtentsMin, p_axis); // mask
	//float val2=pos.x + pos.y + pos.z;
	//float reval=val1;
	//if (val2>val1) reval=val2;
	
	glm::vec3 pos = entrywiseMul(eExt*(float)p_side*0.5f+ePos/*+m_tempRootPos*/-p_parentPos, p_axis); // mask
	float reval=pos.x + pos.y + pos.z;



	return reval;
}


float KDTreeFactory::calculatecost( vector<Tri>* p_tris, float p_splitpos, const glm::vec3& p_axis, const glm::vec3& p_currentSize, float p_currentArea, const glm::vec3& p_currentPos,float& p_outCostLeft,float& p_outCostRight )
{
	glm::vec3 lsize;
	glm::vec3 rsize;
	getChildVoxelsMeasurement(p_splitpos, p_axis, p_currentSize,lsize, rsize);
	float leftarea = calculateArea(lsize);
	float rightarea = calculateArea(rsize);
	int leftcount=0, rightcount=0;
	glm::vec3 leftBoxPos = p_currentPos + 0.5f * entrywiseMul(lsize, p_axis);
	glm::vec3 rightBoxPos = p_currentPos - 0.5f * entrywiseMul(rsize, p_axis);
	calculatePrimitiveCount(p_tris, lsize,rsize,
		leftBoxPos, rightBoxPos,
		leftcount,rightcount);

	//return m_traversalCost + m_intersectionCost * (leftarea * (float)leftcount + rightarea * (float)rightcount);
	p_outCostLeft=m_traversalCost*0.5f + m_intersectionCost * (leftarea/p_currentArea * (float)leftcount);
	p_outCostRight=m_traversalCost*0.5f + m_intersectionCost * (rightarea/p_currentArea * (float)rightcount);
	return m_traversalCost + m_intersectionCost * (leftarea/p_currentArea * (float)leftcount + rightarea/p_currentArea * (float)rightcount);
}

void KDTreeFactory::calculatePrimitiveCount( vector<Tri>* p_tris,const glm::vec3& p_leftBox,const glm::vec3& p_rightBox, const glm::vec3& p_leftBoxPos, const glm::vec3& p_rightBoxPos, int& p_outLeftCount, int& p_outRightCount )
{
	//p_outLeftCount=0;
	//p_outRightCount=0;
	for (int i=0;i<p_tris->size();i++)
	{
		Triparam param = {i,(*p_tris)[i]};
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

float KDTreeFactory::calculateArea( const glm::vec3& p_extents )
{
	return 2.0f * p_extents.x * p_extents.y * p_extents.z;
}

void KDTreeFactory::getChildVoxelsMeasurement( float p_inSplitpos, const glm::vec3& p_axis, const glm::vec3& p_inParentSize, 
											   glm::vec3& p_outLeftSz, glm::vec3& p_outRightSz )
{
	glm::vec3 offset = p_axis * p_inSplitpos;


	glm::vec3 splitH = 0.5f * p_axis; // half-divider

// 	glm::vec3 lsize = (p_inParentSize - offset * 2.0f);
// 	lsize = lsize - entrywiseMul(lsize, splitH);
// 	glm::vec3 rsize = (p_inParentSize - offset * 2.0f);
// 	rsize = rsize - entrywiseMul(rsize, splitH);

	// subtract offset from amound of parent-voxel in set direction 
	glm::vec3 lsize = (p_inParentSize - offset*2.0f);
	lsize = lsize - entrywiseMul(lsize, splitH);

	// Now we thus have the sizes, but we only want the size for the relevant axis:
	// lsize = lsize - entrywiseMul(lsize, p_axis); // Masking 
	// The other is thus the remainder, along active axes
	// The other axes are the same. So use masking:
	glm::vec3 rsize = (p_inParentSize - entrywiseMul(lsize, p_axis));

	p_outLeftSz = lsize;
	p_outRightSz = rsize;
}

glm::vec3 KDTreeFactory::entrywiseMul( const glm::vec3& p_a, const glm::vec3& p_b )
{
	return glm::vec3(p_a.x*p_b.x, p_a.y*p_b.y, p_a.z*p_b.z);
}

int KDTreeFactory::addTree( vector<KDNode>* p_tree, vector<KDLeaf>* p_leafList, vector<int>* p_leafDataList, vector<KDBounds>* p_debugnodeboundsList )
{
	m_trees.push_back(p_tree);
	m_leafLists.push_back(p_leafList);
	m_leafDataLists.push_back(p_leafDataList);
	m_debugTreeNodeBounds.push_back(p_debugnodeboundsList);
	return m_trees.size()-1;
}

vector<KDNode>* KDTreeFactory::getTree( int p_idx )
{
	return m_trees[p_idx];
}

vector<KDLeaf>* KDTreeFactory::getLeafList( int p_idx )
{
	return m_leafLists[p_idx];
}

vector<int>* KDTreeFactory::getLeafDataList( int p_idx )
{
	return m_leafDataLists[p_idx];
}

void KDTreeFactory::generateLeaf( int p_treeId, KDNode* p_node, vector<Tri>* p_tris, vector<int>* p_leafDataList, int p_numTris )
{
	vector<KDLeaf>* leaflist = m_leafLists[p_treeId];
	KDLeaf leaf={0,0};
	int offset=p_leafDataList->size();
	for (int i=0;i<p_numTris;i++)
	for (int j=0;j<3;j++)
	{
		p_leafDataList->push_back((*p_tris)[i].m_ids[j]);
		leaf.m_count++;
	}		
	leaf.m_offset=offset;
	p_node->setLeafData((int)leaflist->size());
	// append leaf as well to separate list
	leaflist->push_back(leaf);
}

KDBounds KDTreeFactory::getTreeBounds( int p_idx )
{
	return m_treeBounds[p_idx];
}

vector<KDBounds>* KDTreeFactory::getDebugNodeBounds( int p_idx )
{
	return m_debugTreeNodeBounds[p_idx];
}

int KDTreeFactory::loadKDTree(int p_levels, const char* p_path)
{
	m_buildLevel=p_levels;
	vector<KDNode>* tree=new vector<KDNode>/*(sc_treeListMaxSize)*/;
	vector<KDLeaf>*	leafList=new vector<KDLeaf>;
	vector<int>*	leafDataList=new vector<int>;
	vector<KDBounds>* debugNodeBoundsList=new vector<KDBounds>; // only for debug draw
	KDBounds treeBounds;

	ifstream inFile;
	inFile.open(string(p_path)+"."+toString(p_levels)+".kdcache");
	if( !inFile.good() || !inFile.is_open() ) 
	{		
		delete tree;
		delete leafList;
		delete leafDataList;
		delete debugNodeBoundsList;
		return -1;
	}
	else
	{
		//string dump;
		int numNodes=0, numLeaves=0, numData=0, numDbg=0;
		// write all settings
		inFile >> numNodes;
		inFile >> numLeaves;
		inFile >> numData;
		inFile >> numDbg;
		inFile >> treeBounds;
		// data
		//inFile >> dump;
		for (int i=0;i<numNodes;i++)
		{
			KDNode node;
			inFile >> node;
			tree->push_back(node);
		}
		//inFile >> dump;
		for (int i=0;i<numLeaves;i++)
		{
			KDLeaf leaf;
			inFile >> leaf;
			leafList->push_back(leaf);
		}
		//inFile >> dump;
		for (int i=0;i<numData;i++)
		{
			int data;
			inFile >> data;
			leafDataList->push_back(data);
		}
		//inFile >> dump;
		for (int i=0;i<numDbg;i++)
		{			
			KDBounds dbg;
			inFile >> dbg;
			debugNodeBoundsList->push_back(dbg);
		}
		//write_file_binary("test.bin", 
		//	reinterpret_cast<char const *>(buffer), 
		//	sizeof(double)*100);
	}
	inFile.close();
	int treeId=addTree(tree,leafList,leafDataList,debugNodeBoundsList);
	m_treeBounds.push_back(treeBounds);
	return treeId;
}

bool KDTreeFactory::saveKDTree(int p_levels, const char* p_path, int p_idx)
{
	m_buildLevel=p_levels;
	ofstream outFile;
	outFile.open( string(p_path)+"."+toString(p_levels)+".kdcache" );

	if( !outFile.good() || !outFile.is_open() ) 
	{
		return false;
	} 
	else 
	{
		// write all settings
		outFile << m_trees[p_idx]->size() <<'\n';
		outFile << m_leafLists[p_idx]->size() <<'\n';
		outFile << m_leafDataLists[p_idx]->size() <<'\n';
		outFile << m_debugTreeNodeBounds[p_idx]->size() <<'\n';
		outFile << m_treeBounds[p_idx] << '\n';
		// data
		//outFile << "NODES";
		for (int i=0;i<m_trees[p_idx]->size();i++)
		{
			outFile << (*m_trees[p_idx])[i];
		}
		//outFile << "LEAVES";
		for (int i=0;i<m_leafLists[p_idx]->size();i++)
		{
			outFile << (*m_leafLists[p_idx])[i];
		}
		//outFile << "LEAF DATA";
		for (int i=0;i<m_leafDataLists[p_idx]->size();i++)
		{
			outFile << (*m_leafDataLists[p_idx])[i] <<'\n';
		}
		//outFile << "DEBUG DATA";
		for (int i=0;i<m_debugTreeNodeBounds[p_idx]->size();i++)
		{
			outFile << (*m_debugTreeNodeBounds[p_idx])[i];
		}
		//write_file_binary("test.bin", 
		//	reinterpret_cast<char const *>(buffer), 
		//	sizeof(double)*100);
	}
	outFile.close();

	//vector<KDNode>* tree=new vector<KDNode>/*(sc_treeListMaxSize)*/;
	//vector<KDLeaf>*	leafList=new vector<KDLeaf>;
	//vector<int>*	leafDataList=new vector<int>;
	//vector<KDBounds>* debugNodeBoundsList=new vector<KDBounds>; // only for debug draw
	//int treeId=addTree(tree,leafList,leafDataList,debugNodeBoundsList);
}

//void KDTreeFactory::clearTempStack()
//{
//	if (m_tempTriListStack->size()>0)
//	{
//		for (int i=0;i<m_tempTriListStack->size();i++)
//		{
//			vector<Tri>* elem = m_tempTriListStack->top();
//			delete elem;
//			m_tempTriListStack->pop();
//		}
//	}
//}
