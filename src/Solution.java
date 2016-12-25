import java.util.*;

/**
 * Created by zplchn on 12/17/16.
 */
public class Solution {

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





    //76
    public String minWindow(String s, String t) {
        if (s == null || s.length() == 0 || t == null || t.length() == 0)
            return "";
        Map<Character, Integer> hm = new HashMap<>();
        int cnt = 0, max = s.length() + 1;
        String res = "";
        for (int i = 0; i < t.length(); ++i)
            hm.put(t.charAt(i), hm.getOrDefault(t.charAt(i), 0) + 1);


        for (int i = 0, j = 0; j < s.length(); ++j){
            char c = s.charAt(j);
            if (!hm.containsKey(c))
                continue;

            hm.put(c, hm.get(c) - 1);
            if (hm.get(c) >= 0)
                ++cnt;
            while (cnt == t.length()){
                char l = s.charAt(i);
                if (!hm.containsKey(l))
                    ++i;
                else if (hm.get(l) < 0){
                    hm.put(l, hm.get(l) + 1);
                    ++i;
                }
                else {
                    if (j - i + 1 < max){
                        max = j - i + 1;
                        res = s.substring(i, j + 1);
                        //break;
                    }
                    break; //note break should be here even if we dont find a new min we still cannot move any more
                }
            }
        }
        return res;
    }

    //90
    public List<List<Integer>> subsetsWithDup(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        if (nums == null)
            return res;
        Arrays.sort(nums); //dont forget to sort!!!
        res.add(new ArrayList<>());
        int size = 1, start = 0;
        for (int i = 0; i < nums.length; ++i){
            start = (i > 0 && nums[i] == nums[i-1])? size: 0;
            size = res.size();
            for (int j = start; j < size; ++j){
                List<Integer> list = new ArrayList<>(res.get(j));
                list.add(nums[i]);
                res.add(list);
            }
        }
        return res;
    }

    //111
    public int minDepth(TreeNode root) {
        if (root == null)
            return 0;
        if (root.left == null)
            return minDepth(root.right) + 1; //dont forget + 1
        if (root.right == null)
            return minDepth(root.left) + 1;
        return Math.min(minDepth(root.left), minDepth(root.right)) + 1;
    }

    //116
    class TreeLinkNode {
        int val;
        TreeLinkNode left, right, next;
        TreeLinkNode(int x){
            val = x;
        }
    }
    public void connect(TreeLinkNode root) {
        if (root == null)
            return;
        if (root.right != null) //always need this cuz the tree can be a sigle node anyways
            root.right.next = root.next == null? null: root.next.left;
        if (root.left != null)
            root.left.next = root.right;
        connect(root.left);
        connect(root.right);
    }

    //155
    public class MinStack {
        private Deque<Integer> st;
        private Deque<Integer> mt;

        /** initialize your data structure here. */
        public MinStack() {
            st = new ArrayDeque<>();
            mt = new ArrayDeque<>();
        }

        public void push(int x) {
            st.push(x);
            if (mt.isEmpty() || x <= mt.peek())
                mt.push(x);
        }

        public void pop() {
            if (st.pop().equals(mt.peek())) //ok, here is use st.pop() == mt.peek() then it never equals. Integer a == Integer b compares refe, not value!
                mt.pop();
        }

        public int top() {
            return st.peek();
        }

        public int getMin() {
            return mt.peek();
        }
    }

    //200 Union find's complexity is o(klogmn) k is # of unions mn is total input. while dfs is o(mn)
    public int numIslands(char[][] grid) {
        if (grid == null || grid.length == 0 || grid[0].length == 0)
            return 0;
        int m = grid.length, n = grid[0].length;
        int[] id = new int[m * n];
        Arrays.fill(id , -1);
        int cnt = 0;
        int[][] offset = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        for (int i = 0; i < m; ++i){
            for (int j = 0; j < n; ++j){
                if (grid[i][j] == '1'){
                    ++cnt;
                    int idx = i * n + j;
                    id[idx] = idx;
                    for (int[] o : offset){
                        int x = i + o[0], y = j + o[1];
                        if (x >= 0 && x < m && y >= 0 & y < n && id[x * n + y] != -1){
                            int root = root(id, x * n + y);
                            if (root != idx){
                                id[root] = idx;
                                --cnt;
                            }
                        }
                    }
                }
            }
        }
        return cnt;
    }

    private int root(int[] id, int x){
        while (id[x] != x){
            id[x] = id[id[x]];
            x = id[x];
        }
        return x;
    }

    //211
    public class WordDictionary {
        class TrieNode{
            boolean isWord;
            TrieNode[] children;

            TrieNode(){
                children = new TrieNode[26];
            }
        }
        private TrieNode root = new TrieNode();
        // Adds a word into the data structure.
        public void addWord(String word) {
            if (word == null)
                return;
            TrieNode tr = root;
            for (int i = 0; i < word.length(); ++i){
                int off = word.charAt(i) - 'a';
                if (tr.children[off] == null)
                    tr.children[off] = new TrieNode();
                tr = tr.children[off];
            }
            tr.isWord = true;
        }

        // Returns if the word is in the data structure. A word could
        // contain the dot character '.' to represent any one letter.
        public boolean search(String word) {
            if (word == null)
                return false;
            return searchHelper(word, root, 0);
        }

        private boolean searchHelper(String word, TrieNode tr, int i){
            if (i == word.length())
                return tr.isWord; //note here returns isWord!!!

            if (word.charAt(i) != '.'){
                if (tr.children[word.charAt(i) - 'a'] == null)
                    return false;
                return searchHelper(word, tr.children[word.charAt(i) - 'a'], i + 1);
            }

            for (TrieNode tn : tr.children){
                if (tn != null && searchHelper(word, tn, i + 1)){
                    return true;
                }
            }
            return false;
        }
    }

    //226
    public TreeNode invertTree(TreeNode root) {
        if (root == null)
            return null;
        TreeNode t = root.left;
        root.left = root.right;
        root.right = t;
        invertTree(root.left);
        invertTree(root.right);
        return root;
    }

    //230
    public int kthSmallest(TreeNode root, int k) {
        if (root == null || k < 1)
            return -1;
        kthHelper(root, k);
        return res;
    }
    private Integer res;
    private int id; //note primitive int will be initialized to 0. Integer x will be initialzied to null

    private void kthHelper(TreeNode root, int k){
        if (root == null)
            return;
        if (res == null)
            kthHelper(root.left, k);
        if (++id == k)
            res = root.val;
        if (res == null)
            kthHelper(root.right, k);
    }

    //238
    public int[] productExceptSelf(int[] nums) {
        if (nums == null || nums.length == 0)
            return nums;
        int[] res = new int[nums.length];
        res[0] = 1;
        for (int i = 1; i < res.length; ++i){
            res[i] = res[i-1] * nums[i-1];
        }
        int t = 1;
        for (int i = res.length - 1; i >= 0; --i){
            res[i] *= t;
            t *= nums[i];
        }
        return res;
    }

    //265
    public int minCostII(int[][] costs) {
        if (costs == null || costs.length == 0 || costs[0].length == 0)
            return 0;
        int min1 = 0, min2 = 0, min1id = -1, tmin1, tmin2, tmin1id = -1; //need to maintain last row and this row, each a set var
        for (int i = 0; i < costs.length; ++i){
            tmin1 = tmin2 = Integer.MAX_VALUE;

            for (int j = 0; j < costs[0].length; ++j){
                costs[i][j] += min1id == j? min2: min1;
                if (costs[i][j] < tmin1){
                    tmin1id = j;
                    tmin2 = tmin1;
                    tmin1 = costs[i][j];
                }
                else if (costs[i][j] < tmin2)
                    tmin2 = costs[i][j];
            }
            min1 = tmin1;
            min2 = tmin2;
            min1id = tmin1id;
        }
        return min1;
    }

    //297
    public class Codec {

        // Encodes a tree to a single string.
        public String serialize(TreeNode root) {
            StringBuilder sb = new StringBuilder();
            serializeHelper(root, sb);
            return sb.toString();
        }

        private void serializeHelper(TreeNode root, StringBuilder sb){
            if (root == null){
                sb.append("#,");
                return;
            }
            sb.append(root.val);
            sb.append(',');
            serializeHelper(root.left, sb);
            serializeHelper(root.right, sb);
        }

        // Decodes your encoded data to tree.
        public TreeNode deserialize(String data) {
            if (data == null || data.length() == 0)
                return null;
            String[] tokens = data.split(",");
            Queue<String> queue = new LinkedList<>();
            for (String t: tokens)
                queue.offer(t);
            return deserializeHelper(queue);
        }

        private TreeNode deserializeHelper(Queue<String> queue){
            String s = queue.poll();
            if (s.equals("#"))
                return null;
            TreeNode root = new TreeNode(Integer.parseInt(s));
            root.left = deserializeHelper(queue);
            root.right = deserializeHelper(queue);
            return root;
        }
    }

    //301
//    public List<String> removeInvalidParentheses(String s) {
//
//    }

    //314
    public List<List<Integer>> verticalOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null)
            return res;
        Map<Integer, List<Integer>> hm = new HashMap<>();
        Queue<TreeNode> qt = new LinkedList<>();
        Queue<Integer> qo = new LinkedList<>();
        qt.offer(root);
        qo.offer(0);
        int min = 0, max = 0;
        while (!qt.isEmpty()){
            TreeNode tn = qt.poll();
            int x = qo.poll();
            min = Math.min(min, x);
            max = Math.max(max, x);
            hm.putIfAbsent(x, new ArrayList<>());
            hm.get(x).add(tn.val);
            if (tn.left != null){
                qt.offer(tn.left);
                qo.offer(x - 1);
            }
            if (tn.right != null){
                qt.offer(tn.right);
                qo.offer(x + 1);
            }
        }

        for (int i = min; i <= max; ++i){
            res.add(hm.get(i));
        }
        return res;
    }

    //334
    public boolean increasingTriplet(int[] nums) {
        // need increasing triplet. maintain a min1 and min2. when x > both min1 and min2 meaning it's a triplet. Note min1 needs to be ahead of min2. so when find a new min, min1 cannot flow to min2, simply update. so left to right.
        if (nums == null || nums.length < 3)
            return false;
        int min1, min2;
        min1 = min2 = Integer.MAX_VALUE;
        for (int x : nums){
            if (x <= min1)
                min1 = x;
            else if (x <= min2) //Note here both need =. it should absorb a dup!!
                min2 = x;
            else
                return true;
        }
        return false;
    }

    //382
    public class Solution2 {
        private ListNode head;
        private Random random;
        /** @param head The linked list's head.
        Note that the head is guaranteed to be not null, so it contains at least one node. */
        public Solution2(ListNode head) {
            this.head = head;
            random = new Random();
        }

        /** Returns a random node's value. */
        public int getRandom() {
            ListNode cur = head.next;
            int res = head.val, n = 1;
            while (cur != null){
                ++n;
                if (random.nextInt(n) == 0)
                    res = cur.val;
                cur = cur.next; //dont forget to advance ll
            }
            return res;
        }
    }

    //398
    public class Solution1 {
        private int[] nums;
        private Random random;
        public Solution1(int[] nums) {
            this.nums = nums;
            random = new Random();
        }

        public int pick(int target) {
            int res = -1, n = 0;
            for (int i = 0; i < nums.length; ++i){
                if (nums[i] == target){
                    ++n;
                    if (random.nextInt(n) == 0) //nextInt(bound) here bound must be positive number, othewise IllegalArgumentException.
                        res = i;
                }
            }
            return res;
        }
    }
}
