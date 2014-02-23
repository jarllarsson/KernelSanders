using UnityEngine;
using System.Collections;
using System.Collections.Generic;

public class Node
{
    public Vector3 m_split = new Vector3();
    public bool m_isLeaf = false;
    public float m_position = 0.0f;
    public int m_leftChildIdx = 0;
    public int m_rightChildIdx = 0;
    public List<object> m_objects = new List<object>();
}
