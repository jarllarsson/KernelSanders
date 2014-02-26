using UnityEngine;
using System.Collections;
using System.Collections.Generic;

public class kdTree : MonoBehaviour 
{
    public List<GameObject> m_objects;
    public Node[] m_tree;
    public Vector3 m_box;
    public float m_firstSplitDist = 0.0f;
	// Use this for initialization
	void Start () 
    {
        Random.seed=(int)(Time.time*1000.0f);
        m_tree = new Node[8024]; // pre-allocate

        Node root = new Node();
        foreach (GameObject obj in m_objects)
        {
            for (int i=0;i<3;i++)
            {
                float absa=2.0f*Mathf.Abs(obj.transform.position[i]-transform.position[i]);
                Debug.Log(absa);
                if (absa>m_box[i])
                   m_box[i]=absa;
            }
            root.m_objects.Add(obj as object);
        }
        m_tree[1] = root;
        buildTree(root, 0, 0, 1, transform.position,m_box); // start at 1

        /*
        tree = new Node();
        tree.m_split.setVec(AxisMark.AXIS.Y);
        tree.m_isLeaf = false;
        tree.m_position = 0.2f;
        tree.m_leftChildIdx = 3;
        m_tree.Add(tree);

        tree = new Node();
        tree.m_split.setVec(AxisMark.AXIS.Y);
        tree.m_isLeaf = true;
        tree.m_position = -0.2f;
        tree.m_leftChildIdx = 0;
        m_tree.Add(tree);

        tree = new Node();
        tree.m_split.setVec(AxisMark.AXIS.X);
        tree.m_isLeaf = false;
        tree.m_position = 0.2f;
        tree.m_leftChildIdx = 5;
        m_tree.Add(tree);

        tree = new Node();
        tree.m_split.setVec(AxisMark.AXIS.X);
        tree.m_isLeaf = true;
        tree.m_position = -0.2f;
        tree.m_leftChildIdx = 0;
        m_tree.Add(tree);

        tree = new Node();
        tree.m_split.setVec(AxisMark.AXIS.Z);
        tree.m_isLeaf = true;
        tree.m_position = 0.1f;
        tree.m_leftChildIdx = 0;
        m_tree.Add(tree);

        tree = new Node();
        tree.m_split.setVec(AxisMark.AXIS.Z);
        tree.m_isLeaf = true;
        tree.m_position = -0.3f;
        tree.m_leftChildIdx = 0;
        m_tree.Add(tree);
         * */
	}
	
	// Update is called once per frame
	void Update () 
    {
        //m_tree[1].m_position = m_firstSplitDist;
	}

    void buildTree(Node p_node, int p_dimsz, int p_dim, int p_idx, Vector3 pos, Vector3 parentSize)
    {

        if (p_dimsz > 24 || p_node.m_objects.Count < 1 || p_idx<<1>m_tree.Length-2) 
        {
            p_node.m_isLeaf = true;
            return;
        }
        Debug.Log("dimsz: "+p_dimsz);
        Debug.Log("idx: " + p_idx);
        if (p_dim > 2) p_dim = 0;
        
        AxisMark splitPlane=new AxisMark();
        splitPlane.setVec(p_dim);
	    float splitpos = findOptimalSplitPos(p_node);

        Vector3 split = splitPlane.getVec();
        Vector3 offset = split * splitpos;
        Vector3 currentOrigo = pos + offset;

        Vector3 splitH = 0.5f * split;
        Vector3 lsize = (parentSize - offset * 2.0f);
        lsize = lsize - entrywiseMul(lsize, splitH);
        Vector3 rsize = (parentSize - offset * 2.0f);
        rsize = rsize - entrywiseMul(rsize, splitH);

        Vector3 leftBoxPos = currentOrigo + 0.5f * entrywiseMul(lsize, split);
        Vector3 rightBoxPos = currentOrigo - 0.5f * entrywiseMul(rsize, split);
        Vector3 leftBox = lsize;
        Vector3 rightBox = rsize;
        Debug.Log("lbp" + leftBox);
        Debug.Log("rbp" + rightBox);
        //Debug.DrawLine(pos, leftBoxPos, Color.red, 100.0f);
        //Debug.DrawLine(leftBoxPos, leftBoxPos + lsize * 0.5f, Color.yellow, 100.0f);
        //Debug.DrawLine(pos, rightBoxPos, Color.green, 100.0f);
        //Debug.DrawLine(rightBoxPos, rightBoxPos + rsize * 0.5f, Color.cyan, 100.0f);

	    Node leftnode = new Node();
        Node rightnode = new Node(); 
        p_node.m_leftChildIdx = p_idx << 1;
        p_node.m_split = splitPlane;
        p_node.m_position = splitpos;
        //
        m_tree[p_node.m_leftChildIdx] = leftnode;
        m_tree[p_node.m_leftChildIdx+1] = rightnode;        
        //
	    foreach (object obj in p_node.m_objects)
	    {
            if (objIntersectNode(true, splitpos, splitPlane, obj, leftBoxPos, leftBox)) 
            {
                //p_node.m_objects.Remove(obj);
                leftnode.m_objects.Add(obj);
                Debug.Log("left");
            }
            if (objIntersectNode(false, splitpos, splitPlane, obj, rightBoxPos, rightBox))
            {
                //p_node.m_objects.Remove(obj);
                rightnode.m_objects.Add(obj);
                Debug.Log("right");
            }
	    }
        buildTree(leftnode, p_dimsz + 1, p_dim + 1, p_node.m_leftChildIdx, leftBoxPos, leftBox); // power of two structure
        buildTree(rightnode, p_dimsz + 1, p_dim + 1, p_node.m_leftChildIdx + 1, rightBoxPos, rightBox);
    }

    bool objIntersectNode(bool p_isLeft, float p_splitpos, AxisMark p_axis, object p_obj, Vector3 pos, Vector3 parentSize)
    {
        GameObject gobj = p_obj as GameObject;
        //Debug.DrawLine(pos, pos + entrywiseMul(parentSize,Vector3.right)*0.5f, Color.white, 100.0f);
        //Debug.Log((gobj.transform.position.x) + " > " + (pos.x + parentSize.x * 0.5f));
        if (gobj.transform.position.x > pos.x + parentSize.x * 0.5f) return false;
        //Debug.DrawLine(pos, pos + entrywiseMul(parentSize, -Vector3.right) * 0.5f, Color.blue, 100.0f);
        if (gobj.transform.position.x < pos.x - parentSize.x * 0.5f) return false;
        //Debug.DrawLine(pos, pos + entrywiseMul(parentSize, Vector3.up) * 0.5f, Color.white, 100.0f);
        if (gobj.transform.position.y > pos.y + parentSize.y * 0.5f) return false;
        //Debug.DrawLine(pos, pos + entrywiseMul(parentSize, -Vector3.up) * 0.5f, Color.blue, 100.0f);
        if (gobj.transform.position.y < pos.y - parentSize.y * 0.5f) return false;
        //Debug.DrawLine(pos, pos + entrywiseMul(parentSize, Vector3.forward) * 0.5f, Color.white, 100.0f);
        if (gobj.transform.position.z > pos.z + parentSize.z * 0.5f) return false;
        //Debug.DrawLine(pos, pos + entrywiseMul(parentSize, -Vector3.forward) * 0.5f, Color.blue, 100.0f);
        if (gobj.transform.position.z < pos.z - parentSize.z * 0.5f) return false;
        Debug.Log("!");
        return true;
        /*
        Vector3 axis = p_axis.getVec();
        //float val = Vector3.Dot(Vector3.Normalize(gobj.transform.position - transform.position - p_axis.getVec() * p_splitpos), axis);
        //Vector3 scaled = axis * p_splitpos;
        // entrywiseMul(axis, p_accDist);
        Vector3 objlen = entrywiseMul(axis,gobj.transform.position-transform.position);
        float val = (scaled.x + scaled.y + scaled.z);
        float valcomp = (objlen.x + objlen.y + objlen.z);
        Debug.Log("val"+val);
        Debug.Log("valcmp" + valcomp);
        if ((p_isLeft && val < valcomp ) || (!p_isLeft && val>=valcomp))
            return true;
        return false;
        */
    }

    float findOptimalSplitPos(Node p_node)
    {
        return 0.2f;
    }

    void OnDrawGizmos()
    {
        Gizmos.DrawWireCube(transform.position, m_box);

        Vector3 parentOrigo = transform.position;

        if (m_tree!=null && m_tree.Length>0)
            drawNode(parentOrigo, m_box,1);
    }

    void drawNode(Vector3 pos, Vector3 parentSize,int idx)
    {
        // offset to get split plane
        if (idx < m_tree.Length && m_tree[idx]!=null)
        {
            Vector3 split=m_tree[idx].m_split.getVec();
            Vector3 offset=split * m_tree[idx].m_position;
            Vector3 currentOrigo = pos + offset;
            Gizmos.color = new Color(0.5f,0.5f,0.5f)+new Color(split.x,split.y,(float)idx/(float)m_tree.Length);

            // draw split plane
            Vector3 drawSize=parentSize-entrywiseMul(parentSize, split);
            if (!m_tree[idx].m_isLeaf) Gizmos.DrawWireCube(currentOrigo, drawSize);

            foreach (object obj in m_tree[idx].m_objects)
            {
                Gizmos.DrawWireSphere(((GameObject)obj).transform.position, 0.1f);
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
