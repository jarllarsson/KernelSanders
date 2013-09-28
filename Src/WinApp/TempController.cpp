#include "TempController.h"
#include <glm\gtc\type_ptr.hpp>
#include <ToString.h>
#include <DebugPrint.h>

TempController::TempController()
{
	m_position = glm::vec4(0.0f,0.0f,0.0f,1.0f);
	m_fovDirtyBit = false;
}

TempController::~TempController()
{

}

void TempController::setFovFromAngle( float angle, float aspectRatio )
{
	setFovFromRad( angle*(float)TORAD, aspectRatio );
}

void TempController::setFovFromRad( float rad, float aspectRatio )
{
	float fovxRad = rad*0.5f;
	float fovyRad = fovxRad;
	m_fovTan.x=aspectRatio*tan(fovxRad); 
	m_fovTan.y=tan(fovyRad);
	m_fovDirtyBit=true;
}


bool TempController::isNewFovAvailable()
{
	return m_fovDirtyBit;
}

glm::vec2& TempController::getFovXY()
{
	 m_fovDirtyBit=false; 
	 return m_fovTan;
}

glm::mat4 TempController::calcRotationMatrix()
{
	glm::mat4 mat=glm::toMat4(m_rotation);
	return mat;
}

glm::vec4 TempController::getPos()
{
	return m_position;
}

void TempController::update( float p_dt )
{
	m_position.z += p_dt;
}

