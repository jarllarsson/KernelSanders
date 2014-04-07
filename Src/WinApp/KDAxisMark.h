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
		b_1=0;
		b_2=0;
	}
	KDAxisMark(const KDAxisMark& p_copy)
	{
		b_1=p_copy.b_1;
		b_2=p_copy.b_2;
	}
	KDAxisMark(int p_x, int p_y, int p_z)
	{
		setVec(p_x,p_y,p_z);
	}
	KDAxisMark(float p_x, float p_y, float p_z)
	{
		setVec((int)(p_x+0.5f),(int)(p_y+0.5f),(int)(p_z+0.5f));
	}

	~KDAxisMark() {}
	
	void setVec(AXIS p_axis)
	{
		setVec((int)p_axis);
	}

	void setVec(int p_xyz)
	{
		if (p_xyz == 0)
			setVec(1, 0, 0);
		else if (p_xyz == 1)
			setVec(0, 1, 0);
		else
			setVec(0, 0, 1);
	}

	void setVec(int p_x, int p_y, int p_z)
	{
		if (p_x>0)
		{
			b_1 = 0;
			b_2 = 0;
		}
		else if (p_y>0)
		{
			b_1 = 1;
			b_2 = 0;
		}
		else
		{
			b_1 = 0;
			b_2 = 1;
		}
	}

	KDAxisMark& operator =(const KDAxisMark& p_other)
	{
		b_1=p_other.b_1;
		b_2=p_other.b_2;
		return *this;
	}

	glm::vec3 getVec() const
	{
		glm::vec3 reval(0.0f,0.0f,0.0f);
		switch (b_1)
		{
		case 0:
			{
				switch (b_2)
				{
				case 0:
					{
						reval.x = 1.0f; // right
						break;
					}
				case 1:
					{
						reval.z = 1.0f; // forward
						break;
					}
				}
				break;
			}
		case 1:
			{
				reval.y = 1.0f; // up
				break;
			}
		}
		return reval;
	}
//protected:	
	// b_2 b_1
	// 00 - x
	// 01 - y
	// 10 - z
	//bool b_1;
	//bool b_2;
	int b_1; // use ints while testing
	int b_2;
private:

};