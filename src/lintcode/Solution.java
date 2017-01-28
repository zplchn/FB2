package lintcode;

import java.util.*;

public class Solution {

    //Subarray Sum
    public ArrayList<Integer> subarraySum(int[] nums) {
        ArrayList<Integer> res = new ArrayList<>();
        if (nums == null || nums.length == 0)
            return res;
        Map<Integer, Integer> hm = new HashMap<>();
        hm.put(0, -1);
        int sum = 0;
        for (int i = 0; i < nums.length; ++i){
            sum += nums[i];
            if (hm.containsKey(sum)){
                res.add(hm.get(sum) + 1);
                res.add(i);
                break;
            }
            else
                hm.put(sum, i);
        }
        return res;
    }

    //inorder
    public ArrayList<Integer> inorderTraversal(TreeNode root) {
        ArrayList<Integer> res = new ArrayList<>();
        if (root == null)
            return res;
        Deque<TreeNode> st = new ArrayDeque<>();
        while (root != null || !st.isEmpty()){
            if (root != null){
                st.push(root);
                root = root.left;
            }
            else {
                TreeNode tn = st.pop();
                res.add(tn.val);
                root = tn.right;
            }
        }
        return res;
    }

    //preorder
    public ArrayList<Integer> preorderTraversal(TreeNode root) {
        ArrayList<Integer> res = new ArrayList<>();
        if (root == null)
            return res;
        Deque<TreeNode> st = new ArrayDeque<>();
        while (root != null || !st.isEmpty()){
            if (root != null){
                res.add(root.val);
                if (root.right != null) //Deque cannot store null val!
                    st.push(root.right);
                root = root.left;
            }
            else {
                root = st.pop();
            }
        }
        return res;
    }

    //postorder
    public ArrayList<Integer> postorderTraversal(TreeNode root) {
        ArrayList<Integer> res = new ArrayList<>();
        if (root == null)
            return res;
        Deque<TreeNode> st = new ArrayDeque<>();
        TreeNode pre = null;
        while (root != null || !st.isEmpty()){
            if (root != null){
                st.push(root);
                root = root.left;
            }
            else {
                TreeNode tn = st.peek().right;
                if (tn != null && pre != tn)
                    root = tn;
                else {
                    pre = st.pop();
                    res.add(pre.val);
                }
            }
        }
        return res;
    }

    //single number 2
    public List<Integer> singleNumberIII(int[] A) {
        List<Integer> res = new ArrayList<>();
        if (A == null || A.length == 0)
            return res;
        int xor = 0;
        for (int x : A)
            xor ^= x;
        xor &= -xor;
        int c1 = 0, c2 = 0;
        for (int x : A){
            if ((x & xor) == 0)
                c1 ^= x;
            else
                c2 ^= x;
        }
        return Arrays.asList(c1, c2);
    }

    //sorted arr to bst
    public TreeNode sortedArrayToBST(int[] A) {
        if (A == null || A.length == 0)
            return null;
        return sortedHelper(A, 0, A.length - 1);
    }

    private TreeNode sortedHelper(int[] A, int l, int r){
        if (l > r)
            return null;
        int m = l + ((r - l) >> 1);
        TreeNode tl = sortedHelper(A, l, m - 1);//here use l to m - 1
        TreeNode root = new TreeNode(A[m]);
        root.left = tl;
        root.right = sortedHelper(A, m + 1, r);
        return root;
    }

    //minimum subarray
    public int minSubArray(ArrayList<Integer> nums) {
        if (nums == null || nums.size() == 0)
            return -1;
        int lmin = nums.get(0), min = lmin;
        for (int i = 1; i < nums.size(); ++i){
            lmin = Math.min(nums.get(i), lmin + nums.get(i));
            min = Math.min(min, lmin);
        }
        return min;
    }

    //maximum subarray 2
    public int maxTwoSubArrays(ArrayList<Integer> nums) { //like stock 2
        if (nums == null || nums.size() == 0)
            return 0;
        int[] left = new int[nums.size()];
        int[] right = new int[nums.size()];
        int lmax = nums.get(0);
        left[0] = lmax;
        for (int i = 1; i < left.length; ++i){
            lmax = Math.max(lmax + nums.get(i), nums.get(i));
            left[i] = Math.max(left[i-1], lmax); //left keeps all max till current i
        }
        lmax = right[right.length - 1] = nums.get(nums.size() - 1);
        for (int i = nums.size() - 2; i >= 0; --i){
            lmax = Math.max(lmax + nums.get(i), nums.get(i));
            right[i] = Math.max(right[i+1], lmax);
        }
        int res = Integer.MIN_VALUE;
        for (int i = 0; i < left.length -1; ++i){ //END must result in two subarray!!
            res = Math.max(res, left[i] + right[i+1]); //right till the next one
        }
        return res;
    }

    //binary tree max path sum
    public int maxPathSum(TreeNode root) {
        if (root == null)
            return 0;
        max = root.val;
        maxHelper(root);
        return max;
    }

    private int max;

    private int maxHelper(TreeNode root){ //need return int!!
        if (root == null)
            return 0;
        int lsum = Math.max(maxHelper(root.left), 0);
        int rsum = Math.max(maxHelper(root.right), 0);
        max = Math.max(lsum + rsum + root.val, max);
        return root.val + Math.max(lsum, rsum);
    }

    //merge k sorted lists

    public ListNode mergeKLists(List<ListNode> lists) {
        // write your code here
        if (lists == null || lists.size() == 0)
            return null;
        //Queue<ListNode> pq = new PriorityQueue<>((l1, l2)->l1.val - l2.val);
        Queue<ListNode> pq = new PriorityQueue<>(lists.size(), new Comparator<ListNode>(){
            public int compare(ListNode l1, ListNode l2){
                return l1.val - l2.val;
            }
        });
        for (ListNode ln : lists){
            if (ln != null)
                pq.offer(ln);
        }
        ListNode dummy = new ListNode(0), pre = dummy;
        while (!pq.isEmpty()){
            ListNode ln = pq.poll();
            pre.next = ln;
            pre = pre.next;
            if (ln.next != null)
                pq.offer(ln.next);
        }
        pre.next = null;
        return dummy.next;
    }

    //unique bt
    public List<TreeNode> generateTrees(int n) {
        // write your code here
        return generateHelper(1, n);
    }

    private List<TreeNode> generateHelper(int l, int r){
        List<TreeNode> res = new ArrayList<>(); //when invalid case need to output null
        if (l > r){
            res.add(null);
            return res;
        }
        for (int i = l; i <= r; ++i){
            List<TreeNode> ll = generateHelper(l, i - 1);
            List<TreeNode> rl = generateHelper(i + 1, r);
            for (TreeNode lln : ll){
                for (TreeNode rln : rl){
                    TreeNode root = new TreeNode(i);
                    root.left = lln;
                    root.right = rln;
                    res.add(root);
                }
            }
        }
        return res;
    }

    //topological sorting
    public ArrayList<DirectedGraphNode> topSort(ArrayList<DirectedGraphNode> graph) {
        // write your code here
        ArrayList<DirectedGraphNode> res = new ArrayList<>();
        if (graph == null || graph.size() == 0)
            return res;
        Map<DirectedGraphNode, Integer> hm = new HashMap<>();
        for (DirectedGraphNode dgn : graph){
            if (!hm.containsKey(dgn))
                hm.put(dgn, 0);
            for (DirectedGraphNode n : dgn.neighbors){
                hm.put(n, hm.containsKey(n)? hm.get(n) + 1: 1);
            }
        }
        Queue<DirectedGraphNode> queue = new LinkedList<>();
        for (DirectedGraphNode n : hm.keySet()){
            if (hm.get(n) == 0)
                queue.offer(n);
        }
        while (!queue.isEmpty()){
            DirectedGraphNode n = queue.poll();
            res.add(n);
            for (DirectedGraphNode nbr : n.neighbors){
                hm.put(nbr, hm.get(nbr) - 1);
                if (hm.get(nbr) == 0)
                    queue.offer(nbr);
            }
        }
        return res;
    }

    //flatten binary tree
    private TreeNode pre;
    public void flatten(TreeNode root) {
        if (root == null)
            return;
        if (pre != null){
            pre.left = null;
            pre.right = root;
        }
        pre = root;
        TreeNode t = root.right;
        flatten(root.left);
        flatten(t);
    }

    //first missing positive
    public int firstMissingPositive(int[] A) {
        if (A == null || A.length == 0)
            return 1;
        for (int i = 0; i < A.length; ++i){ //ignore 0 only consider 1, 2,...!
            if (A[i] != i + 1 && A[i] > 0 && A[i] <= A.length && A[A[i] -1] != A[i]){
                int t = A[A[i] - 1];
                A[A[i] - 1] = A[i];
                A[i] = t;
                --i;
            }
        }
        for (int i = 0; i< A.length; ++i){
            if (A[i] != i + 1)
                return i + 1;
        }
        return A.length + 1;
    }

    //largest histogram
    public int largestRectangleArea(int[] height) {
        if (height == null || height.length == 0)
            return 0;
        int res = 0;
        Deque<Integer> st = new ArrayDeque<>();
        for (int i = 0; i <= height.length; ++i){
            while (!st.isEmpty() && (i == height.length || height[st.peek()] > height[i])){ //st is index. all compare uses height[st.peek()]
                res = Math.max(res, height[st.pop()] * (st.isEmpty()? i: i - st.peek() - 1));
            }
            if (i != height.length)
                st.push(i);
        }
        return res;
    }

    //graph valid tree
    public boolean validTree(int n, int[][] edges) {
        if (n <= 0 || edges == null || edges.length != n - 1)
            return false;
        if (n == 1) //when 1. no indegree == 1 leaf
            return true;
        int[] indegree = new int[n];
        List<Integer>[] children = new List[n];
        for (int i = 0; i < children.length; ++i)
            children[i] = new ArrayList<>();
        for (int[] e : edges){
            ++indegree[e[0]];
            ++indegree[e[1]];
            children[e[0]].add(e[1]);
            children[e[1]].add(e[0]);
        }
        Queue<Integer> queue = new LinkedList<>();
        int cnt = 0;
        for (int i = 0; i < indegree.length; ++i){
            if (indegree[i] == 1) //undirected so start from leaf. only leaf is 1 indegree
                queue.offer(i);
        }
        while (!queue.isEmpty()){
            int x = queue.poll();
            ++cnt;
            for (int c : children[x]){
                if (--indegree[c] == 1)
                    queue.offer(c);
            }
        }
        return cnt == n;
    }

    //combination sum
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> res = new ArrayList<>();
        if (candidates == null || candidates.length == 0)
            return res;
        Arrays.sort(candidates);
        combiHelper(candidates, target, 0, 0, new ArrayList<Integer>(), res);
        return res;
    }

    private void combiHelper(int[] candi, int target, int i, int sum, List<Integer> combi, List<List<Integer>> res){
        if (sum == target){
            res.add(new ArrayList<>(combi));
            return;
        }
        for (int j = i; j < candi.length; ++j){
            if (j > i && candi[j] == candi[j-1]) //if dup, the dup one will not get chosen at any level!
                continue;
            if (sum + candi[j] <= target){
                combi.add(candi[j]);
                combiHelper(candi, target, j, sum + candi[j], combi, res);
                combi.remove(combi.size() - 1);
            }
        }
    }

    //candy
    public int candy(int[] ratings) {
        if (ratings == null || ratings.length == 0)
            return 0;
        int[] num = new int[ratings.length];
        Arrays.fill(num, 1);
        for (int i = 1; i < ratings.length; ++i){
            if (ratings[i] > ratings[i-1])
                num[i] = num[i-1] + 1;
        }
        int res = num[num.length - 1];
        for (int i = num.length - 2; i >= 0; --i){
            if (ratings[i] > ratings[i+1] && num[i] <= num[i+1]){
                num[i] = num[i+1] + 1;
            }
            res += num[i]; // this needs to add regardless
        }
        return res;
    }

    //pow(x, n)
    public double myPow(double x, int n) {
        if (n == 0)
            return 1;
        double half = myPow(x, n / 2);
        if (n % 2 == 0)
            return half * half;
        else if (n > 0)
            return half * half * x;
        else
            return half * half / x;
    }

    //sort color
    public void sortColors(int[] nums) {
        if (nums == null || nums.length == 0)
            return;
        for (int i0 = 0, i1 = 0, i2 = nums.length - 1; i1 <= i2; ++i1){
            if (nums[i1] == 0){
                nums[i1] = nums[i0];
                nums[i0++] = 0;
            }
            else if (nums[i1] == 2){
                nums[i1] = nums[i2];
                nums[i2--] = 2;
                --i1;
            }
        }
    }

    //minimum window substring
    public String minWindow(String source, String target) {
        if (source == null || source.isEmpty() || target == null || target.isEmpty() || source.length() < target.length())
            return "";
        Map<Character, Integer> hm = new HashMap<>();
        for (char c : target.toCharArray()){
            if (hm.containsKey(c))
                hm.put(c, hm.get(c) + 1);
            else
                hm.put(c, 1);
        }
        int min = source.length() + 1, cnt = 0; //min should initialize to source.length() + 1
        String res = "";
        for (int l = 0, r = 0; r < source.length(); ++r){
            char rc = source.charAt(r);
            if (!hm.containsKey(rc))
                continue;
            hm.put(rc, hm.get(rc) - 1);
            if (hm.get(rc) >= 0)
                ++cnt;
            while (cnt == target.length()){
                char lc = source.charAt(l);
                if (!hm.containsKey(lc))
                    ++l;
                else if (hm.get(lc) < 0){
                    hm.put(lc, hm.get(lc) + 1);
                    ++l;
                }
                else {
                    if (r - l + 1 < min){
                        min = r - l + 1;
                        res = source.substring(l, r + 1);
                    }
                    break;
                }
            }
        }
        return res;
    }

    //minimum size subarray sum
    public int minimumSize(int[] nums, int s) {
        if (nums == null || nums.length == 0)
            return -1;
        int res = nums.length + 1, sum = 0;
        for (int l = 0, r = 0; r < nums.length; ++r){
            sum += nums[r];
            while (l <= r && sum >= s){
                res = Math.min(res, r - l + 1);
                sum -= nums[l++];
            }
        }
        return res > nums.length? -1: res;
    }

    //paint house 2
    public int minCostII(int[][] costs) {
        if (costs == null || costs.length == 0 || costs[0].length == 0)
            return 0;
        int m1 = 0, m2 = 0, i1 = -1; //only need to record the min1's index, ifnot equal, use min2
        for (int i =0; i < costs.length; ++i){
            int tm1 = Integer.MAX_VALUE, ti1 = -1, tm2 = Integer.MAX_VALUE;
            for (int j = 0; j < costs[0].length; ++j){
                costs[i][j] += j == i1? m2: m1;
                if (costs[i][j] < tm1){
                    tm2 = tm1;
                    tm1 = costs[i][j];
                    ti1 = j;
                }
                else if (costs[i][j] < tm2){
                    tm2 = costs[i][j];
                }
            }
            m1 = tm1; // this happens outside j loop
            i1 = ti1;
            m2 = tm2;
        }
        return m1;
    }

    //majaority 2
    public int majorityNumber(List<Integer> nums) {
        if (nums == null || nums.size() == 0)
            return -1;
        int m1 = -1,c1 = 0, m2 = -1, c2 = 0;
        for (int x : nums){
            if (x == m1)
                ++c1;
            else if (x == m2)
                ++c2;
            else if (c1 == 0){ //cnt == 0 is the signal we need a new candidate!
                m1 = x;
                c1 = 1;
            }
            else if (c2 == 0){
                m2 = x;
                c2 = 1;
            }
            else {
                --c1;
                --c2;
            }
        }
        c1 = 0;
        c2 = 0;

        for (int x : nums){
            if (x == m1)
                ++c1;
            else if (x == m2)
                ++c2;
        }
        if (c1 > nums.size() / 3) //both m1 and m2 could possilby be a candidate, no order guarantee!
            return m1;
        if (c2 > nums.size() / 3)
            return m2;
        return -1;
    }

    //bt zigzag
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null)
            return res;
        Queue<TreeNode> queue = new LinkedList<>();
        boolean rev = false;
        queue.offer(root); //dont forget to offer root!!!
        int cur = 1, next = 0;
        List<Integer> lvl = new ArrayList<>();

        while (!queue.isEmpty()){
            TreeNode tn = queue.poll();
            lvl.add(tn.val);
            if (tn.left != null){
                queue.offer(tn.left);
                ++next;
            }
            if (tn.right != null){
                queue.offer(tn.right);
                ++next;
            }
            if (--cur == 0){
                if (rev){
                    Collections.reverse(lvl);
                }
                rev = !rev;
                res.add(lvl);
                lvl = new ArrayList<>();
                cur = next;
                next = 0;
            }
        }
        return res;
    }

    //restore ip
    public List<String> restoreIpAddresses(String s) {
        List<String> res = new ArrayList<>();
        if (s == null || s.length() < 4)
            return res;
        restoreHelper(s, 0, 0, "", res);
        return res;
    }

    private void restoreHelper(String s, int i, int k, String pre, List<String> res){
        if (k == 3){
            String rest = s.substring(i);
            if (isValidIp(rest))
                res.add(pre + rest);
            return;
        }
        for (int j = i + 1; j <= i + 3 && j <= s.length(); ++j){ //j can = i + 3
            String sub = s.substring(i, j);
            if (isValidIp(sub)){
                restoreHelper(s, j, k + 1, pre + sub + ".", res);
            }
        }
    }

    private boolean isValidIp(String s){
        return s.length() > 0 && s.length() <= 3 && (s.length() > 1 ? s.charAt(0) != '0': true) && Integer.parseInt(s) < 256;
    }

    //Route exist between two nodes in a graph
    public boolean hasRoute(ArrayList<DirectedGraphNode> graph,
                            DirectedGraphNode s, DirectedGraphNode t) {
        //the graph is totally useless
        if (graph == null || s == null || t == null)
            return false;
        return hasRouteHelper(s, t, new HashSet<DirectedGraphNode>());
    }

    private boolean hasRouteHelper(DirectedGraphNode s, DirectedGraphNode t, HashSet<DirectedGraphNode> visited){
        if (s == t)
            return true;
        visited.add(s);
        for (DirectedGraphNode n : s.neighbors){
            if (!visited.contains(n) && hasRouteHelper(n, t, visited))
                return true;
        }
        return false;
    }

    //maximum subarray recording start and end
    public ArrayList<Integer> continuousSubarraySum(int[] A) {
        ArrayList<Integer> res = new ArrayList<>();
        if (A == null || A.length == 0)
            return res;
        int lmax = A[0], max = A[0]; //still intialize to 0
        res.add(0);
        res.add(0);
        for (int l = 0, r = 1; r < A.length; ++r){ //l should start at 0!!!
            if (A[r] > lmax + A[r]){
                l = r; //cannot directly set res since may not be a new max
                lmax = A[r];
            }
            else
                lmax += A[r];
            if (lmax > max){
                max = lmax;
                res.set(0, l);//only when a new max, set both l and r
                res.set(1, r);
            }
        }
        return res;
    }

    //house robber 2
    public int houseRobber2(int[] nums) {
        if (nums == null || nums.length == 0)
            return 0;
        if (nums.length == 1) //must have, since will use start index = 1 as one case
            return nums[0];
        return Math.max(robHelper(nums, 0, nums.length - 2), robHelper(nums, 1, nums.length - 1));
    }

    private int robHelper(int[] nums, int l, int r){ //same subproblems separate as a function
        int[] dp = new int[3];
        dp[1] = dp[2] = nums[l];
        for (int i = l+1; i <= r; ++i){
            dp[2] = Math.max(dp[1], nums[i] + dp[0]);
            dp[0] = dp[1];
            dp[1] = dp[2];
        }
        return dp[2];
    }

    //partition list
    public ListNode partition(ListNode head, int x) {
        if (head == null || head.next == null)
            return head;
        ListNode sd = new ListNode(0), sp = sd;
        ListNode bd = new ListNode(0), bp = bd;
        while (head != null){
            if (head.val < x){
                sp.next = head;
                sp = sp.next;
            }
            else {
                bp.next = head;
                bp = bp.next;
            }
            head = head.next;
        }
        bp.next = null; // this is must since big can be not the last one and needs point null
        sp.next = bd.next;
        return sd.next;
    }

    //word search 2
    class TrieNode{
        boolean isWord;
        TrieNode[] children;

        TrieNode(){
            children = new TrieNode[26];
        }
    }

    public ArrayList<String> wordSearchII(char[][] board, ArrayList<String> words) {
        ArrayList<String> res = new ArrayList<>();
        if (board == null || board.length == 0 || board[0].length == 0 || words == null || words.size() == 0)
            return res;
        TrieNode root = buildTrie(words);
        Set<String> hs = new HashSet<>();
        for (int i = 0; i < board.length; ++i){
            for (int j = 0; j < board[0].length; ++j){
                searchHelper(board, i, j, root, new StringBuilder(), hs);
            }
        }
        res.addAll(hs);
        return res;
    }

    private TrieNode buildTrie(ArrayList<String> words){
        TrieNode root = new TrieNode(), tr;
        for (String w : words){
            tr = root;
            for (char c : w.toCharArray()){
                int off = c - 'a';
                if (tr.children[off] == null)
                    tr.children[off] = new TrieNode();
                tr = tr.children[off];
            }
            tr.isWord = true;
        }
        return root;
    }
    private int[][] off1 = {{-1,0}, {1,0}, {0,-1}, {0,1}};
    private void searchHelper(char[][] board, int i, int j, TrieNode root, StringBuilder sb, Set<String> hs){
        if (root == null)
            return;
        if (root.isWord){
            hs.add(sb.toString());
        }
        if (i < 0 || i >= board.length || j < 0 || j >= board[0].length || (board[i][j] & 256) != 0)
            return;
        root = root.children[board[i][j] - 'a'];
        sb.append(board[i][j]);
        board[i][j] ^= 256; // this must be the last step after root move and sb append since it changes the char itself!!
        for (int[] o : off1){
            searchHelper(board, i + o[0], j + o[1], root, sb, hs);
        }
        sb.deleteCharAt(sb.length() - 1);
        board[i][j] ^= 256;
    }

    //number of island
    public int numIslands(boolean[][] grid) {
        if (grid == null || grid.length == 0 || grid[0].length == 0)
            return 0;
        int res = 0;
        for (int i = 0; i < grid.length; ++i){
            for (int j = 0; j < grid[0].length; ++j){
                if (grid[i][j]){
                    ++res;
                    islandHelper(grid, i, j);
                }
            }
        }
        return res;
    }
    private final int[][] off = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

    private void islandHelper(boolean[][] grid, int i, int j){
        grid[i][j] = false;
        for (int[] o : off){
            int x = i + o[0], y = j + o[1];
            if (x >= 0 && x < grid.length && y >= 0 && y < grid[0].length && grid[x][y])
                islandHelper(grid, x, y);
        }
    }

}
