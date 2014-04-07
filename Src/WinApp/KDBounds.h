#pragma once
#include <glm\gtc\type_ptr.hpp>
#include <fstream>
// =======================================================================================
//                                      KDBounds
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Brief
///        
/// # KDBounds
/// 
/// 31-3-2014 Jarl Larsson
///---------------------------------------------------------------------------------------

struct KDBounds
{
	glm::vec3 m_pos;
	glm::vec3 m_extents;
	friend std::ostream& operator<<( std::ostream& stream, const KDBounds& v );
	friend std::ifstream& operator>>( std::ifstream& stream, KDBounds& v );
};
