using UnityEngine;
using System.Collections;
using System.Collections.Generic;

public class kdTree : MonoBehaviour 
{
    public List<GameObject> m_objects;
    public Node[] m_tree;
    public Vector3 m_box;
    public float m_firstSplitDist = 0.0f;
    public float m_intersectionCost = 0.01f;
    public float m_traversalCost = 100000.3f;

    private Stack<List<object>> m_tempObjectsStack=new Stack<List<object>>();

	// Use this for initialization
	void Start () 
    {
        Random.seed=(int)(Time.time*1000.0f);
        m_tree = new Node[8024]; // pre-allocate

        Node root = new Node();

        List<object> rootobjects = new List<object>();
        m_tempObjectsStack.Push(rootobjects);

        m_objects.AddRange(GameObject.FindGameObjectsWithTag("intree"));
        foreach (GameObject obj in m_objects)
        {
            for (int i=0;i<3;i++)
            {
                float absa=2.0f*Mathf.Abs(obj.transform.position[i]-transform.position[i]);
                //Debug.Log(absa);
                if (absa>m_box[i])
                   m_box[i]=absa;
            }
            rootobjects.Add(obj as object);
        }
        m_tree[1] = root;
        buildTree(root, rootobjects, 0, 0, 1, transform.position, m_box); // start at 1
        m_tempObjectsStack.Pop();
        Debug.Log("fin stack: " + m_tempObjectsStack.Count);
	}
	
	// Update is called once per frame
	void Update () 
    {
        //m_tree[1].m_position = m_firstSplitDist;
	}

    void buildTree(Node p_node, List<object> p_objects, int p_dimsz, int p_dim, int p_idx, Vector3 pos, Vector3 parentSize)
    {
        p_node.pos = pos;
        p_node.size = parentSize;
        if (p_dimsz > 24 || p_objects.Count < 3 || p_idx<<1>m_tree.Length-2) 
        {
            p_node.m_objects = p_objects; // in c++ do deep copy here
            p_node.m_isLeaf = true;
            return;
        }

        if (p_dim > 2) p_dim = 0;
        
        AxisMark splitPlane=new AxisMark();
        splitPlane.setVec(p_dim);
	    float splitpos = findOptimalSplitPos(p_node, p_objects, splitPlane,parentSize,pos);

        Vector3 split = splitPlane.getVec();
        Vector3 offset = split * splitpos;
        Vector3 currentOrigo = pos + offset;

        Vector3 lsize;
        Vector3 rsize;
        getChildVoxelsMeasurement(splitpos, split, parentSize,
                                  out lsize, out rsize);


        Vector3 leftBoxPos = currentOrigo + 0.5f * entrywiseMul(lsize, split);
        Vector3 rightBoxPos = currentOrigo - 0.5f * entrywiseMul(rsize, split);
        Vector3 leftBox = lsize;
        Vector3 rightBox = rsize;

	    Node leftnode = new Node();
        Node rightnode = new Node(); 
        p_node.m_leftChildIdx = p_idx << 1;
        p_node.m_split = splitPlane;
        p_node.m_position = splitpos;
        //
        m_tree[p_node.m_leftChildIdx] = leftnode;
        m_tree[p_node.m_leftChildIdx+1] = rightnode;        
        //
        List<object> leftObjects = new List<object>();
        List<object> rightObjects = new List<object>();
        m_tempObjectsStack.Push(rightObjects);
        m_tempObjectsStack.Push(leftObjects);
        //
	    foreach (object obj in p_objects)
	    {
            if (objIntersectNode(/*true, splitpos, splitPlane,  */obj,leftBoxPos, leftBox)) 
            {
                //p_node.m_objects.Remove(obj);
                leftObjects.Add(obj);
            }
            if (objIntersectNode(/*false, splitpos, splitPlane, */obj, rightBoxPos, rightBox))
            {
                //p_node.m_objects.Remove(obj);
                rightObjects.Add(obj);
            }
	    }
        //Debug.Log("stack: "+m_tempObjectsStack.Count);
        buildTree(leftnode, leftObjects, p_dimsz + 1, p_dim + 1, p_node.m_leftChildIdx, leftBoxPos, leftBox); // power of two structure
        m_tempObjectsStack.Pop();
        buildTree(rightnode, rightObjects,p_dimsz + 1, p_dim + 1, p_node.m_leftChildIdx + 1, rightBoxPos, rightBox);
        m_tempObjectsStack.Pop();
    }

    bool objIntersectNode(/*bool p_isLeft, float p_splitpos, AxisMark p_axis, */object p_obj, Vector3 pos, Vector3 parentSize)
    {
        GameObject gobj = p_obj as GameObject;
        if (gobj.transform.position.x > pos.x + parentSize.x * 0.5f) return false;
        if (gobj.transform.position.x < pos.x - parentSize.x * 0.5f) return false;
        if (gobj.transform.position.y > pos.y + parentSize.y * 0.5f) return false;
        if (gobj.transform.position.y < pos.y - parentSize.y * 0.5f) return false;
        if (gobj.transform.position.z > pos.z + parentSize.z * 0.5f) return false;
        if (gobj.transform.position.z < pos.z - parentSize.z * 0.5f) return false;
        return true;
    }

    float findOptimalSplitPos(Node p_node, List<object> p_objects,AxisMark p_axis, Vector3 p_currentSize, Vector3 p_currentPos)
    {
        float bestpos = 0.0f;
        float bestcost = float.MaxValue;
        Vector3 axis = p_axis.getVec();
        foreach (object obj in p_objects)
        {
            GameObject gobj = obj as GameObject;
	        float left_extreme = getLeftExtreme(gobj, axis);
	        float right_extreme = getRightExtreme(gobj, axis);
            float cost = calculatecost(p_node, p_objects, left_extreme, axis, p_currentSize,p_currentPos);
	        if (cost < bestcost)
            {
	            bestcost = cost; bestpos = left_extreme;
            }
            //if (cost >= 1000000) Debug.Log("L!!! " + cost);
            cost = calculatecost(p_node, p_objects, right_extreme, axis, p_currentSize, p_currentPos);
            if (cost < bestcost)
            {
	            bestcost = cost; bestpos = right_extreme;
            }
            //if (cost >= 1000000) Debug.Log("R!!! " + cost);
        }
        return bestpos;
    }

    float getLeftExtreme(GameObject p_obj, Vector3 p_axis)
    {
        Vector3 pos=entrywiseMul(p_obj.collider.bounds.extents, p_axis);
        return pos.x + pos.y + pos.z;
    }

    float getRightExtreme(GameObject p_obj, Vector3 p_axis)
    {
        Vector3 pos = -entrywiseMul(p_obj.collider.bounds.extents, p_axis);
        return pos.x + pos.y + pos.z;
    }

    float calculatecost(Node p_node, List<object> p_objects,float p_splitpos, Vector3 p_axis, Vector3 p_currentSize, Vector3 p_currentPos)
    {
        Vector3 lsize;
        Vector3 rsize;
        getChildVoxelsMeasurement(p_splitpos, p_axis, p_currentSize,
                                  out lsize, out rsize);
        float leftarea = calculateArea(lsize);
        float rightarea = calculateArea(rsize);
	    int leftcount, rightcount;
        Vector3 leftBoxPos = p_currentPos + 0.5f * entrywiseMul(lsize, p_axis);
        Vector3 rightBoxPos = p_currentPos - 0.5f * entrywiseMul(rsize, p_axis);
        calculatePrimitiveCount(p_node, p_objects, lsize,rsize,
                                leftBoxPos, rightBoxPos,
                                out leftcount,out rightcount);

        return m_traversalCost + m_intersectionCost * (leftarea * (float)leftcount + rightarea * (float)rightcount);
    }

    void calculatePrimitiveCount(Node p_node, List<object> p_objects, Vector3 p_leftBox,Vector3 p_rightBox,
                                 Vector3 p_leftBoxPos, Vector3 p_rightBoxPos,
                                 out int outLeftCount, out int outRightCount)
    {
        outLeftCount=0;
        outRightCount=0;
        foreach (object obj in p_objects)
        {
            if (objIntersectNode(obj, p_leftBoxPos, p_leftBox))
            {
                outLeftCount++;
            }
            if (objIntersectNode(obj, p_rightBoxPos, p_rightBox))
            {
                outRightCount++;
            }
        }
    }

    float calculateArea(Vector3 p_extents)
    {
        return 2.0f * p_extents.x * p_extents.y * p_extents.z;
    }

    void getChildVoxelsMeasurement(float p_inSplitpos, Vector3 p_axis, Vector3 p_inParentSize,
                                   out Vector3 outLeftSz, out Vector3 outRightSz)
    {
//         if (p_inSplitpos < 0.0f)
//             Debug.Log(p_inSplitpos);
        Vector3 offset = p_axis * p_inSplitpos; // where to split from origo

        Vector3 splitH = 0.5f * p_axis; // half divider 
        // subtract offset from amound of parent-voxel in set direction 
        Vector3 lsize = (p_inParentSize - offset*2.0f);
        lsize = lsize - entrywiseMul(lsize, splitH);
        // Now we thus have the sizes, but we only want the size for the relevant axis:
        // lsize = lsize - entrywiseMul(lsize, p_axis); // Masking 
        // The other is thus the remainder, along active axes
        // The other axes are the same. So use masking:
        Vector3 rsize = (p_inParentSize - entrywiseMul(lsize, p_axis));
        //Vector3 lsize = (p_inParentSize - offset * 2.0f); 
        //lsize = lsize - entrywiseMul(lsize, splitH);
        //Vector3 rsize = (p_inParentSize - offset * 2.0f);
        //rsize = rsize - entrywiseMul(rsize, splitH);

        //if (rsize != lsize)
        //    Debug.Log("L: " + lsize.ToString() + " R: " + rsize.ToString());

        outLeftSz = lsize;
        outRightSz = rsize;
    }

    void OnDrawGizmos()
    {


        Vector3 parentOrigo = transform.position;

        if (m_tree!=null && m_tree.Length>0)
            drawNode(parentOrigo, m_box, 1); 
        
        Gizmos.color = Color.white;
        Gizmos.DrawWireCube(transform.position, m_box);
    }

    void drawNode(Vector3 pos, Vector3 parentSize,int idx)
    {
        // offset to get split plane
        if (idx < m_tree.Length && m_tree[idx]!=null)
        {
            Vector3 split=m_tree[idx].m_split.getVec();
            Vector3 offset=split * m_tree[idx].m_position;
            Vector3 currentOrigo = pos + offset;
            Gizmos.color = new Color(0.5f,0.5f,0.5f,0.0f)+new Color(split.z,split.y,split.x,0.5f);

            // draw split plane
            Vector3 drawSize=parentSize-entrywiseMul(parentSize, split);
            Gizmos.DrawWireCube(m_tree[idx].pos, m_tree[idx].size*0.99f);
            Gizmos.DrawWireCube(m_tree[idx].pos, m_tree[idx].size*0.999f);
            Gizmos.DrawWireCube(m_tree[idx].pos, m_tree[idx].size);
            //if (!m_tree[idx].m_isLeaf) 
            //Gizmos.DrawWireCube(currentOrigo, drawSize);

            foreach (object obj in m_tree[idx].m_objects)
            {
                Gizmos.DrawWireSphere(((GameObject)obj).transform.position, 2.0f);
            }

            if (!m_tree[idx].m_isLeaf)
            {
                //Vector3 dir = new Vector3(0.0f, vertSplit, horzSplit);
                Vector3 splitH = 0.5f * split;
                Vector3 lsize = (parentSize-offset*2.0f);
                lsize=lsize-entrywiseMul(lsize, splitH);
                Vector3 rsize = (parentSize-offset*2.0f);
                rsize=rsize-entrywiseMul(rsize, splitH);
                drawNode(currentOrigo + 0.5f * entrywiseMul(lsize, split), lsize, m_tree[idx].m_leftChildIdx);
                drawNode(currentOrigo - 0.5f * entrywiseMul(rsize, split), rsize, m_tree[idx].m_leftChildIdx + 1);
            }
        }

        return;
    }

    Vector3 entrywiseMul(Vector3 p_a, Vector3 p_b)
    {
        return new Vector3(p_a.x * p_b.x, p_a.y * p_b.y, p_a.z * p_b.z);
    }
}
