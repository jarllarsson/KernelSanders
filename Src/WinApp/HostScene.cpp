#include "HostScene.h"


HScene::HScene()
{
	setDirty(HScene::BOX);
	setDirty(HScene::LIGHT);
	setDirty(HScene::PLANE);
	setDirty(HScene::SPHERE);
	setDirty(HScene::TRI);
}

void HScene::setDirty( OBJTYPE p_objType, bool p_status/*=true*/ )
{
	switch (p_objType)
	{
	case OBJTYPE::LIGHT:
		lightDirty=p_status;
		break;
	case OBJTYPE::SPHERE:
		sphereDirty=p_status;
		break;
	case OBJTYPE::PLANE:
		planeDirty=p_status;
		break;
	case OBJTYPE::TRI:
		triDirty=p_status;
		break;
	case OBJTYPE::BOX:
		boxDirty=p_status;
		break;
	default:
		break;		
	}
}



bool HScene::isDirty( OBJTYPE p_objType )
{
	bool res=false;
	switch (p_objType)
	{
	case OBJTYPE::LIGHT:
		res=lightDirty;
		break;
	case OBJTYPE::SPHERE:
		res=sphereDirty;
		break;
	case OBJTYPE::PLANE:
		res=planeDirty;
		break;
	case OBJTYPE::TRI:
		res=triDirty;
		break;
	case OBJTYPE::BOX:
		res=boxDirty;
		break;
	default:
		res=false;
		break;
	}
	return res;
}


