import java.util.*;

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

    public Map<Integer, Integer> top5(int[][] scores){
        Map<Integer, Integer> res = new HashMap<>();
        if (scores == null || scores.length == 0 || scores[0].length == 0)
            return res;
        //create a key - pq map
        Map<Integer, Queue<Integer>> hmscore = new HashMap<>();
        //populate the hm
        for (int[] s: scores){
            //If (!hm.containsKey(s[0]))
            //	hm.put(s[0], new PriorityQueue<>());
            hmscore.putIfAbsent(s[0], new PriorityQueue<>());
            Queue<Integer> pq = hmscore.get(s[0]);
            if (pq.size() < 5)
                pq.offer(s[1]);
            else if (pq.peek() < s[1]){
                pq.poll();
                pq.offer(s[1]);
            }
        }
        //calculate the avg per id
        for (Map.Entry<Integer, Queue<Integer>> entry: hmscore.entrySet()){
            Queue<Integer> pq = entry.getValue();
            Iterator<Integer> iter = pq.iterator();
            int sum = 0;
            while (iter.hasNext()){
                sum += iter.next();
            }
            res.put(entry.getKey(), sum / pq.size());

        }
        return res;
    }




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
        dfs(root,avg,mroot); //postorder to find max subtree
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

    //5
    public String longestPalindrome(String s) {
        if (s == null || s.length() == 0)
            return s;
        boolean[][] dp = new boolean[s.length()][s.length()]; //dp[i][j] indicates whether substring s[i...j] is a palindrome
        int max = 0, l = 0, r = 0; //log for global longest along with the start l and end r indices
        for (int i = s.length() - 1; i >= 0; --i){
            for (int j = i; j < s.length(); ++j){ //left side starts off right to left, right side from start to the end
                if (s.charAt(i) == s.charAt(j) && (j - i <= 2 || dp[i+1][j-1])){ //the inner side is already calculated
                    dp[i][j] = true;
                    if (j - i + 1 > max){ //log global longest
                        max = j - i + 1;
                        l = i;
                        r = j;
                    }
                }
            }
        }
        return s.substring(l, r + 1); //take the longest substring
    }








}
