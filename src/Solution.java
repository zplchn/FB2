import java.util.*;

/**
 * Created by zplchn on 12/17/16.
 */
public class Solution {

    //9
    public boolean isPalindrome(int x) {
        if (x < 0)
            return false;
        int div = 1;
        while (x / div >= 10)
            div *= 10;
        while (x > 0){ //here must > 0 like 1021 , 1000021 should return false
            if (x / div != x % 10)
                return false;
            x = x % div / 10;
            div /= 100;
        }
        return true;
    }

    //38
    public String countAndSay(int n) {
        if (n < 1)
            return "";
        StringBuilder sb = new StringBuilder("1"), t;

        while (n-- > 1){ //first one is 1. then 11
            t = new StringBuilder();
            char c = sb.charAt(0);
            int cnt = 1;

            for (int i = 1; i < sb.length(); ++i){
                if (sb.charAt(i) == c)
                    ++cnt;
                else {
                    t.append(cnt);
                    t.append(c);
                    c = sb.charAt(i);
                    cnt = 1;
                }
            }
            t.append(cnt);
            t.append(c);
            sb = t;
        }
        return sb.toString();
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

    //93
    public List<String> restoreIpAddresses(String s) {
        List<String> res = new ArrayList<>();
        if (s == null || s.length() == 0)
            return res;
        restoreHelper(s, 0, 0, "", res);
        return res;
    }

    private void restoreHelper(String s, int i, int k, String pre, List<String> res){
        if (k == 3){
            String left = s.substring(i);
            if (isValidIp(left))
                res.add(pre + left);
            return;
        }
        for (int j = i+1; j <= Math.min(s.length(), i + 3); ++j){
            String str = s.substring(i, j);
            if (isValidIp(str))
                restoreHelper(s, j, k + 1, pre + str + ".", res);
        }
    }

    private boolean isValidIp(String s){
        return s.length() > 0 && s.length() <= 3 && (s.length() > 1? s.charAt(0) != '0': true) && Integer.parseInt(s) <= 255;
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

    //138
    class RandomListNode{
        int label;
        RandomListNode next, random;
        RandomListNode(int x){label = x;}
    }
    public RandomListNode copyRandomList(RandomListNode head) {
        // use 3 steps: 1. create copy node right after each current node 2. set random link 3. split
        if (head == null)
            return head;

        // 1. create copy as node1 -> copy1 -> node2-> copy2...
        RandomListNode cur = head;
        while (cur != null){
            RandomListNode next = cur.next;
            cur.next = new RandomListNode(cur.label);
            cur.next.next = next;
            cur = next;
        }
        // 2. set random pointer
        cur = head;
        while (cur != null){
            cur.next.random = cur.random == null? null: cur.random.next;
            cur = cur.next.next;
        }

        // 3. split
        RandomListNode dummy = new RandomListNode(0), pre = dummy;
        cur = head;
        while (cur != null){
            pre.next = cur.next;
            pre = pre.next;
            cur.next = cur.next.next;
            cur = cur.next;
        }
        return dummy.next;
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

    //202
    public boolean isHappy(int n) {
        if (n < 1)
            return false;
        Set<Integer> hs = new HashSet<>();
        while (n != 1){
            int t = 0;
            while (n != 0){
                int x = n % 10;
                t += x * x;
                n /= 10;
            }
            if (hs.contains(t))
                return false;
            hs.add(t);
            n = t;
        }
        return true;
    }

    //203
    public ListNode removeElements(ListNode head, int val) {
        if (head == null)
            return head;
        ListNode dummy = new ListNode(0), pre = dummy;
        dummy.next = head; //always when use dummy, first see if need dummy.next = head!!!!
        while (pre.next != null){
            if (pre.next.val == val)
                pre.next = pre.next.next;
            else
                pre = pre.next;
        }
        return dummy.next;
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

    //212
    public List<String> findWords(char[][] board, String[] words) {
        List<String> res = new ArrayList<>();
        if (board == null || board.length == 0 || board[0].length == 0 || words == null || words.length == 0)
            return res;
        TrieNode troot = buildTrie(words);
        Set<String> hs = new HashSet<>();
        for (int i = 0; i < board.length; ++i){
            for (int j = 0; j < board[0].length; ++j){
                findHelper(board, i, j, troot, new StringBuilder(), hs);
            }
        }
        res.addAll(hs);
        return res;
    }

    private TrieNode buildTrie(String[] words){
        TrieNode root = new TrieNode(), tr = root;
        for (String w : words){
            tr = root;
            for (int i = 0; i < w.length(); ++i){
                int off = w.charAt(i) - 'a';
                if (tr.children[off] == null)
                    tr.children[off] = new TrieNode();
                tr = tr.children[off];
            }
            tr.isWord = true;
        }
        return root;
    }
    private final int[][] off = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    private void findHelper(char[][] board, int i, int j, TrieNode root, StringBuilder sb, Set<String> hs){
        if (root.isWord){
            hs.add(sb.toString());
        }
        if (i < 0 || i >= board.length || j < 0 || j >= board[0].length || (board[i][j] & 256) != 0 || root.children[board[i][j] - 'a'] == null)
            return;
        sb.append(board[i][j]);
        root = root.children[board[i][j] - 'a']; //this must before xor 256!!
        board[i][j]^= 256; //Need visited!!!
        for (int[] o : off){
            int x = i + o[0], y = j + o[1];
            findHelper(board, x, y, root, sb, hs);
        }
        sb.deleteCharAt(sb.length() - 1);
        board[i][j]^= 256;
    }

    //220
    public boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
        //use sliding window size = k, find at any give nums[i] if exists a ceiling or floor number within t's range
        if (nums == null || nums.length == 0 || k <= 0)
            return false;
        TreeSet<Integer> ts = new TreeSet<>();
        for (int i = 0; i < nums.length; ++i){
            //check
            Integer ceiling = ts.ceiling(nums[i]);
            Integer floor = ts.floor(nums[i]);
            if ((ceiling != null && ceiling <= nums[i] + t) || (floor != null && nums[i] <= floor + t)) //note here use + not - to prevent INF - neg overflow!
                return true;
            ts.add(nums[i]); //dont forget add in
            //shrink
            if (i >= k)
                ts.remove(nums[i-k]);
        }
        return false;
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

    //243
    public int shortestDistance(String[] words, String word1, String word2) {
        if (words == null || words.length < 2 || word1 == null || word2 == null)
            return -1;
        int l1 = -1, l2 = -1, res = words.length + 1;
        for (int i = 0; i < words.length; ++i){
            if (words[i].equals(word1)){
                if (l2 != -1)
                    res = Math.min(res, i - l2);
                l1 = i; //dont forget to update the found index!
            }
            else if (words[i].equals(word2)){
                if (l1 != -1)
                    res = Math.min(res, i - l1);
                l2 = i;
            }
        }
        return res;
    }

    //247
    public List<String> findStrobogrammatic(int n) {
        List<String> res = new ArrayList<>();
        if (n < 1)
            return res;
        findHelper(new char[n], 0, n - 1, res);
        return res;
    }
    private final char[][] stro = {{'0', '0'}, {'1', '1'}, {'8', '8'}, {'6', '9'}, {'9', '6'}};
    private void findHelper(char[] combi, int l, int r, List<String> res){
        if (l >= r){
            if (l == r){
                for (int i = 0; i < 3; ++i){
                    combi[l] = stro[i][0];
                    res.add(new String(combi));
                }
            }
            else
                res.add(new String(combi));
            return;
        }
        int start = l == 0? 1: 0;
        for (int i = start; i < stro.length; ++i){
            combi[l] = stro[i][0];
            combi[r] = stro[i][1];
            findHelper(combi, l + 1, r - 1, res);
        }
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
    public List<String> removeInvalidParentheses(String s) {
        List<String> res = new ArrayList<>();
        if (s == null) //""is a valid output
            return res;
        int l = 0, r = 0;
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(')
                ++l;
            else if (s.charAt(i) == ')'){
                if (l > 0)
                    --l;
                else
                    ++r; //")(" int this case when right is more, we also increase r and the subsequent l will also be counted
            }
        }

        dfs(s, 0, res, new StringBuilder(), l, r, 0);
        Set<String> hs = new HashSet<>(res);

        return new ArrayList<String>(hs);
    }

    public void dfs(String s, int i, List<String> res, StringBuilder sb, int rml, int rmr, int open) {
        if (open < 0 || rml < 0 || rmr < 0)
            return;
        if (i == s.length()) {
            if (rml == 0 && rmr == 0 && open == 0) {
                res.add(sb.toString());
            }
            return;
        }
        char c = s.charAt(i);
        int len = sb.length();
        if (c == '(') {
            dfs(s, i + 1, res, sb, rml - 1, rmr, open);
            dfs(s, i + 1, res, sb.append(c), rml, rmr, open + 1);
        }
        else if (c == ')') {
            dfs(s, i + 1, res, sb, rml, rmr - 1, open);
            dfs(s, i + 1, res, sb.append(c), rml, rmr, open - 1);
        }
        else
            dfs(s, i + 1, res, sb.append(c), rml, rmr, open);
        sb.setLength(len);
    }

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

    //346
    public class MovingAverage {
        private Queue<Integer> queue;
        private double sum;
        private int size;
        /** Initialize your data structure here. */
        public MovingAverage(int size) {
            this.size = size;
            queue = new LinkedList<>();
        }

        public double next(int val) {
            if (queue.size() == size){
                sum -= queue.poll();//dont adjust size
            }
            queue.offer(val);
            sum += val;
            return sum / queue.size();
        }
    }

    //356
    public boolean isReflected(int[][] points) {
        //for a y symmetry axis. it would be x = (minx + maxx)/2, so we should find minx and maxx. and for each point check if the symmeric point exist. so we need to store them into a hashset
        if (points == null || points.length == 0)
            return true;
        //note new int[] cannot be used as hashkey. cuz not compare content; use string instead
        int minx = points[0][0], maxx = points[0][0];
        Set<String> hs = new HashSet<>();
        for (int[] p : points){
            minx = Math.min(minx, p[0]);
            maxx = Math.max(maxx, p[0]);
            hs.add(p[0] + ":" + p[1]);
        }
        long t = minx + maxx;
        //java's hashmap iterator can only get from entrySet()
        for (int[] p: points){
            int sk = (int)(t - p[0]);
            if (!hs.contains(sk + ":" + p[1]))
                return false;
        }
        return true;
    }

    //360
    public int[] sortTransformedArray(int[] nums, int a, int b, int c) {
        if (nums == null || nums.length == 0)
            return nums;
        int[] res = new int[nums.length];
        int i = a > 0? nums.length - 1: 0, l = 0, r = res.length - 1;
        while (l <= r){
            int lc = calc(nums[l], a, b, c);
            int rc = calc(nums[r], a, b, c);
            if (a > 0){
                if (lc < rc){
                    res[i--] = rc;
                    --r;
                }
                else {
                    res[i--] = lc;
                    ++l;
                }
            }
            else {
                if (lc < rc){
                    res[i++] = lc;
                    ++l;
                }
                else {
                    res[i++] = rc;
                    --r;
                }
            }
        }
        return res;
    }

    private int calc(int x, int a, int b, int c){
        return a * x * x + b * x + c;
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
