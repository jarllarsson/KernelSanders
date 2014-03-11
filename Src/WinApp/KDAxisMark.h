#pragma once

#include <glm/glm.hpp>

// =======================================================================================
//                                      KDAxisMark
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Brief
///        
/// # KDAxisMark
/// 
/// 11-3-2014 Jarl Larsson
///---------------------------------------------------------------------------------------

class KDAxisMark
{
public:
	enum AXIS
	{
		X,Y,Z
	};

	KDAxisMark()
	{
		b_1=false;
		b_2=false;
	}
	~KDAxisMark() {}
	
	void setVec(AXIS p_axis)
	{
		setVec((int)p_axis);
	}

	void setVec(int p_xyz)
	{
		if (p_xyz == 0)
			setVec(true, false, false);
		else if (p_xyz == 1)
			setVec(false, true, false);
		else
			setVec(false, false, true);
	}

	void setVec(bool p_x, bool p_y, bool p_z)
	{
		if (p_x)
		{
			b_1 = false;
			b_2 = false;
		}
		else if (p_y)
		{
			b_1 = true;
			b_2 = false;
		}
		else
		{
			b_1 = false;
			b_2 = true;
		}
	}

	glm::vec3 getVec()
	{
		glm::vec3 reval(0.0f,0.0f,0.0f);
		switch (b_1)
		{
		case false:
			{
				switch (b_2)
				{
				case false:
					{
						reval.x = 1.0f; // right
						break;
					}
				case true:
					{
						reval.z = 1.0f; // forward
						break;
					}
				}
				break;
			}
		case true:
			{
				reval.y = 1.0f; // up
				break;
			}
		}
		return reval;
	}
protected:
private:
	// b_2 b_1
	// 00 - x
	// 01 - y
	// 10 - z
	bool b_1;
	bool b_2;
};