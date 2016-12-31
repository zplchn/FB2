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

    //Minimum spanning tree
    class Connection{
        String node1;
        String node2;
        int cost;
        public Connection(String a, String b, int c){
            node1 = a;
            node2 = b;
            cost = c;
        }
    }

    public static List<Connection> getLowCost(List<Connection> connections) {
        List<Connection> res = new ArrayList<>();
        if (connections == null || connections.isEmpty())
            return res;
        // first sort the given connections according to the cost, small to large
        Collections.sort(connections, new Comparator<Connection>(){
            @Override
            public int compare(Connection c1, Connection c2) {
                return c1.cost - c2.cost;
            }
        });
        // make unionfind structure. assign unique key to each node
        Map<String, String> hm = new HashMap<>();
        for (Connection c : connections) {
            hm.put(c.node1, c.node1);
            hm.put(c.node2, c.node2);
        }

        int total = hm.size(), index = 0;
        while (index < connections.size() && res.size() < total - 1) {
            Connection cur = connections.get(index++);
            String root1 = root(cur.node1, hm); //find
            String root2 = root(cur.node2, hm);
            if (root1.equals(root2))
                continue; // already a connected component
            hm.put(root1, root2); //union
            res.add(cur);
        }
        //insufficient connections to link all nodes. return null
        if (res.size() != total - 1)
            return null;
        //sort result by node1 ascending
        Collections.sort(res, new Comparator<Connection>(){
            @Override
            public int compare(Connection c1, Connection c2){
                return c1.node1.equals(c2.node1)? c1.node2.compareTo(c2.node2): c1.node1.compareTo(c2.node1);
            }
        });
        return res;
    }

    private static String root(String s, Map<String, String> hm){
        while (!s.equals(hm.get(s))) {
            s = hm.get(s);
        }
        return s;
    }




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

    //order
    class Order{
        String name;
        public Order (String name){
            this.name = name;
        }
    }

    class OrderDependency{
        Order dependent;
        Order independent;
        public OrderDependency(Order dependent, Order independent){
            this.dependent = dependent;
            this.independent = independent;
        }
    }

    public List<Order> findOrder(List<OrderDependency> dependencies){
        List<Order> res = new ArrayList<>();
        if (dependencies == null || dependencies.size() == 0)
            return res;
        //Initialize two maps
        Map<String, Integer>      inmap  = new HashMap<>(); //inmap (order, indegree)
        Map<String, List<String>> outmap = new HashMap<>(); //outmap (order, children_list)

        //construct graph
        for (OrderDependency i : dependencies){
            inmap.put(i.dependent.name, inmap.getOrDefault(i.dependent.name, 0) + 1);
            inmap.put(i.independent.name, 0);
            outmap.putIfAbsent(i.independent.name, new ArrayList<>());
            outmap.get(i.independent.name).add(i.dependent.name);
        }

        //topological sorting use bfs
        int total = inmap.size();
        Queue<String>q = new LinkedList<>();
        for (String s : inmap.keySet()){
            if (inmap.get(s) == 0)
                q.offer(s); //start with indegree == 0
        }
        while (!q.isEmpty()){
            String s = q.poll();
            res.add(new Order(s));
            if (outmap.containsKey(s)){
                for (String o: outmap.get(s)){
                    inmap.put(o, inmap.get(o) - 1);
                    if (inmap.get(o) == 0)
                        q.offer(o);
                }
            }
            outmap.remove(s);
        }
        if (res.size() != total)
            return new ArrayList<Order>();
        return res;
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
