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

    //305
    public List<Integer> numIslands2(int m, int n, int[][] positions) {
        //Union-Find problem. Union - combine two connected set; Find - check if two obj belong same connected set
        //complxity o(MlogN) M -#of unions N - total # of obj
        List<Integer> res = new ArrayList<>();
        if (m <= 0 || n <= 0 || positions == null || positions.length == 0 || positions[0].length == 0)
            return res;
        //union find needs to create an extra space id[] size = input
        int[] id = new int[m * n];
        Arrays.fill(id, -1); //initially all nodes belong to a dark -1 set - water
        int count = 0;

        for (int[] p : positions){
            int idx = p[0] * n + p[1];
            id[idx] = idx; //doesnt matter what id we give this time. we will union later
            ++count;

            //now find all 4 neighbours. Note we just need to look the 4 neighbours.
            //NOT DFS. because they all have an asscociate id. we just see if can union - one island
            int[][] off = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
            for (int[] o: off){
                int x = p[0] + o[0], y = p[1] + o[1], newIdx = x * n + y;
                //still do valid check of indices and if is valid set
                if (x >= 0 && x < m && y >= 0 && y < n && id[newIdx] != -1) { //not water set
                    //union. make their index = mine
                    int root = root(id, newIdx);
                    if (root != idx) { //may have dup input positions
                        id[root] = idx; //union. set as children under idx
                        --count; //every union decrease the count(number of set) by 1
                    }
                }
            }
            res.add(count); //every time log count
        }
        return res;
    }

    private int root(int[] id, int i){ //quick union + path compression
        while (id[i] != i){
            id[i] = id[id[i]]; //path compression
            i = id[i];
        }
        return i;
    }








}
