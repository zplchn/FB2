import java.util.ArrayList;
import java.util.List;

/**
 * Created by zplchn on 12/25/16.
 */


class Node {

    int val;
    ArrayList<Node>children;
    public Node(int val){
        this.val=val;
        children=new ArrayList<Node>();
    }
}
public class Solution1 {
    static class SumCount
    {
        double sum;
        int count;
        public SumCount(double sum, int count)
        {
            this.sum = sum;
            this.count = count;
        }
    }

    public static Node find(Node root)
    {
        if(root==null) return null;
        List<Double>avg=new ArrayList<>(); //avg stores global max avg
        List<Node>mroot=new ArrayList<>(); //mroot stores global max avg subtree root
        avg.add(Double.MIN_VALUE);
        mroot.add(root);
        dfs(root,avg,mroot); //postorder to find max subtree
        return mroot.get(0);
    }

    private static SumCount dfs(Node root,List<Double>avg,List<Node>mroot) {
        SumCount sc = new SumCount(0, 0); //prepare return (sum, count)
        if (root == null)
            return sc;
        sc.sum = root.val;
        sc.count = 1;
        if (root.children.isEmpty()) //stop recursion when at leaf
            return sc;

        for (Node itr : root.children) {
            SumCount rsc = dfs(itr, avg, mroot); //postorder accumulate sum and count
            sc.sum += rsc.sum;
            sc.count += rsc.count;
        }
        if (sc.count > 1 && sc.sum / sc.count > avg.get(0)) { //update global avg and mroot
            avg.set(0, sc.sum / sc.count);
            mroot.set(0, root);
        }
        return sc; //return current SunCount
    }

    public static void main(String[] args){
        Node root = new Node(100);
        Node c1 = new Node(20);
        Node c2 = new Node(20);
        Node c3 = new Node(20);
        Node c4 = new Node(30);
        Node c5 =new Node(30);
        Node c6= new Node(30);


        root.children.add(c1);
        root.children.add(c4);
        c1.children.add(c2);
        c1.children.add(c3);
        c4.children.add(c5);
        c4.children.add(c6);
        Node n=Solution1.find(root);
        System.out.println(n.val);
    }








}
