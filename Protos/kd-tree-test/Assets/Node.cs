using UnityEngine;
using System.Collections;
using System.Collections.Generic;

[System.Serializable]
public struct AxisMark
{
    public enum AXIS
    {
        X,Y,Z
    }
    // b_2 b_1
    // 00 - x
    // 01 - y
    // 10 - z
    public bool b_1;
    public bool b_2;

    public void setVec(AXIS p_axis)
    {
        setVec((int)p_axis);
    }

    public void setVec(int p_xyz)
    {
        if (p_xyz == 0)
            setVec(true, false, false);
        else if (p_xyz == 1)
            setVec(false, true, false);
        else
            setVec(false, false, true);
    }

    public void setVec(bool p_x, bool p_y, bool p_z)
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

    public Vector3 getVec()
    {
        Vector3 reval = Vector3.zero;
        switch (b_1)
        {
            case false:
                {
                    switch (b_2)
                    {
                        case false:
                            {
                                reval = Vector3.right;
                                break;
                            }
                        case true:
                            {
                                reval = Vector3.forward;
                                break;
                            }
                    }
                    break;
                }
            case true:
                {
                    reval = Vector3.up;
                    break;
                }
        }
        return reval;
    }
}
[System.Serializable]
public class Node
{
    public AxisMark m_split;
    public bool m_isLeaf = false;
    public float m_position = 0.0f;
    public float m_fposition = 0.0f;
    public int m_leftChildIdx = 0; // right child is always left child+1
    public List<object> m_objects = new List<object>();
    // Debug
    public Vector3 pos; 
    public Vector3 size;
}
