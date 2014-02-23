using UnityEngine;
using System.Collections;
using System.Collections.Generic;

public class kdTree : MonoBehaviour 
{
    public List<Node> m_tree = new List<Node>();
    public Vector3 m_box;
    public float m_firstSplitDist = 0.0f;
	// Use this for initialization
	void Start () 
    {
        Node tree = new Node();
        tree.m_split = Vector3.forward;
        tree.m_isLeaf = false;
        tree.m_position = 1.0f;
        tree.m_leftChildIdx = 1; // upcoming
        tree.m_rightChildIdx = 2;// upcoming
        m_tree.Add(tree);

        tree = new Node();
        tree.m_split = Vector3.up;
        tree.m_isLeaf = false;
        tree.m_position = 0.2f;
        tree.m_leftChildIdx = 3;
        tree.m_rightChildIdx = 4;
        m_tree.Add(tree);

        tree = new Node();
        tree.m_split = Vector3.up;
        tree.m_isLeaf = true;
        tree.m_position = -0.2f;
        tree.m_leftChildIdx = 0;
        tree.m_rightChildIdx = 0;
        m_tree.Add(tree);

        tree = new Node();
        tree.m_split = Vector3.right;
        tree.m_isLeaf = false;
        tree.m_position = 0.2f;
        tree.m_leftChildIdx = 5;
        tree.m_rightChildIdx = 6;
        m_tree.Add(tree);

        tree = new Node();
        tree.m_split = Vector3.right;
        tree.m_isLeaf = true;
        tree.m_position = -0.2f;
        tree.m_leftChildIdx = 0;
        tree.m_rightChildIdx = 0;
        m_tree.Add(tree);

        tree = new Node();
        tree.m_split = Vector3.forward;
        tree.m_isLeaf = true;
        tree.m_position = 0.1f;
        tree.m_leftChildIdx = 0;
        tree.m_rightChildIdx = 0;
        m_tree.Add(tree);

        tree = new Node();
        tree.m_split = Vector3.forward;
        tree.m_isLeaf = true;
        tree.m_position = -0.3f;
        tree.m_leftChildIdx = 0;
        tree.m_rightChildIdx = 0;
        m_tree.Add(tree);
	}
	
	// Update is called once per frame
	void Update () 
    {
        m_tree[0].m_position = m_firstSplitDist;
	}

    void OnDrawGizmos()
    {
        Gizmos.DrawWireCube(transform.position, m_box);

        Vector3 parentOrigo = transform.position;

        if (m_tree.Count>0)
            drawNode(parentOrigo, m_box,0);
    }

    void drawNode(Vector3 pos, Vector3 parentSize,int idx)
    {

        // offset to get split plane
        Vector3 split=m_tree[idx].m_split;
        Vector3 offset=split * m_tree[idx].m_position;
        Vector3 currentOrigo = pos + offset;
        Gizmos.color = new Color(split.x,split.y,(float)idx/(float)m_tree.Count);

        // draw split plane
        Vector3 drawSize=parentSize-new Vector3(parentSize.x*split.x,parentSize.y*split.y,parentSize.z*split.z);
        Gizmos.DrawWireCube(currentOrigo,drawSize);

        if (!m_tree[idx].m_isLeaf)
        {
            //Vector3 dir = new Vector3(0.0f, vertSplit, horzSplit);
            Vector3 splitH = 0.5f * split;
            Vector3 lsize = (parentSize-offset*2.0f);
            lsize=new Vector3(lsize.x-lsize.x*splitH.x, lsize.y-lsize.y*splitH.y, lsize.z-lsize.z*splitH.z);
            Vector3 rsize = (-parentSize-offset*2.0f);
            rsize=new Vector3(rsize.x-rsize.x*splitH.x, rsize.y-rsize.y*splitH.y, rsize.z-rsize.z*splitH.z);
            drawNode(currentOrigo+0.5f*new Vector3(lsize.x*split.x,lsize.y*split.y,lsize.z*split.z), lsize, m_tree[idx].m_leftChildIdx);
            drawNode(currentOrigo+0.5f*new Vector3(rsize.x*split.x,rsize.y*split.y,rsize.z*split.z), rsize, m_tree[idx].m_rightChildIdx);
        }
        return;
    }
}
