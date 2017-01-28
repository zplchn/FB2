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

    //19
    public ListNode removeNthFromEnd(ListNode head, int n) {
        if (head == null)
            return null;
        ListNode dummy = new ListNode(0), pre = dummy, cur = dummy; // cur start from dummy!
        dummy.next = head;
        while (n > 0 && cur != null && cur.next != null){
            cur = cur.next;
            --n;
        }
        if (n > 0)
            return head;
        while (cur != null && cur.next != null){
            cur = cur.next;
            pre = pre.next;
        }
        pre.next = pre.next.next;
        return dummy.next;
    }

    //24
    public ListNode swapPairs(ListNode head) {
        if (head == null)
            return head;
        ListNode dummy = new ListNode(0), pre = dummy, cur = head;
        dummy.next = head;
        while (cur != null && cur.next != null){
            ListNode next = cur.next.next;
            pre.next = cur.next;
            cur.next.next = cur;
            cur.next = next; //dont miss this!
            pre = cur;
            cur = next;
        }
        return dummy.next;
    }

    //29
    public int divide(int dividend, int divisor) {
        if (divisor == 0 || (dividend == Integer.MIN_VALUE && divisor == -1))
            return Integer.MAX_VALUE;
        //cannot carry sign, like -6 / 2. so first remove sign, Math.abs(int.min_value) == Integer.MIN_VALUE, a neg number !!!
        long dvd = Math.abs((long)dividend);
        long dvs = Math.abs((long)divisor);
        int res = 0;
        //dvd = dvs * ([0|1]2^n + [2^n-1] + ... + [2] + 1), any number can be expressed as a combi of power of 2

        while (dvd >= dvs){
            int i = 0;
            while (dvd >= (dvs << i))
                ++i;
            --i;
            res += 1 << i;
            dvd -= dvs << i;
        }
        return ((dividend ^ divisor) >>> 31) == 1? -res: res; //note >>> moves sign bit!!
    }

    //31
    public void nextPermutation(int[] nums) {
        if (nums == null || nums.length <= 1)
            return;
        int i = nums.length - 2;
        while (i >= 0 && nums[i] >= nums[i + 1])
            --i;
        if (i < 0){
            reverse(nums, 0, nums.length - 1);
            return;
        }
        int j = nums.length - 1;
        while (nums[j] <= nums[i])
            --j;
        swap(nums, i, j);
        reverse(nums, i + 1, nums.length - 1);
    }

    private void reverse(int[] nums, int l, int r){
        while (l < r){
            swap(nums, l++, r--);
        }
    }

    private void swap(int[] nums, int i, int j){
        int t = nums[i];
        nums[i] = nums[j];
        nums[j] = t;
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

    //58
    public int lengthOfLastWord(String s) {
        if (s == null)
            return 0;
        s = s.trim();
        if (s.isEmpty())
            return 0;
        String[] tokens = s.split("\\s+"); //when all space, return an empty array
        return tokens[tokens.length - 1].length();
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

    //84
    public int largestRectangleArea(int[] heights) {
        if (heights == null || heights.length == 0)
            return 0;
        Deque<Integer> st = new ArrayDeque<>();
        int res = 0;
        for (int i = 0; i <= heights.length; ++i){
            while (!st.isEmpty() && (i == heights.length || heights[i] < heights[st.peek()])){ //heights[peek] since its index
                res = Math.max(res, heights[st.pop()] * (st.isEmpty()? i: i - st.peek() - 1));
            }
            if (i != heights.length)
                st.push(i);
        }
        return res;
    }

    //89
    public List<Integer> grayCode(int n) {
        List<Integer> res = new ArrayList<>();
        if (n < 0)
            return res;
        for (int i = 0; i < (1 << n); ++i){
            res.add(i ^ (i >> 1)); //it's xor i right shift by 1
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

    //97
    public boolean isInterleave(String s1, String s2, String s3) {
        if (s1 == null || s2 == null || s1.length() + s2.length() != s3.length())
            return false;
        boolean[][] dp = new boolean[s1.length() + 1][s2.length() + 1];

        for (int i = 0; i <= s1.length(); ++i){
            for (int j = 0; j <= s2.length(); ++j){
                if (i == 0 && j == 0)
                    dp[i][j] = true;
                else if (i == 0)
                    dp[0][j] = dp[0][j-1] && s2.charAt(j-1) == s3.charAt(j-1);
                else if (j == 0)
                    dp[i][0] = dp[i-1][0] && s1.charAt(i-1) == s3.charAt(i-1);
                else
                    dp[i][j] = (s1.charAt(i-1) == s3.charAt(i + j - 1) && dp[i-1][j]) || (s2.charAt(j-1) == s3.charAt(i + j - 1) && dp[i][j-1]);
            }
        }
        return dp[dp.length - 1][dp[0].length - 1];
    }

    //101
    public boolean isSymmetric(TreeNode root) {
        if (root == null)
            return true;
        return isSymHelper(root.left, root.right);
    }

    private boolean isSymHelper(TreeNode l, TreeNode r){ //mirror by itself
        if (l == null)
            return r == null;
        if (r == null)
            return false;
        return l.val == r.val && isSymHelper(l.left, r.right) && isSymHelper(l.right, r.left);
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

    //123
    public int maxProfit(int[] prices) {
        if (prices == null || prices.length <= 1)
            return 0;
        int buy1 = Integer.MIN_VALUE, buy2 = Integer.MIN_VALUE, sell1 = 0, sell2 = 0;
        for (int p : prices){
            buy1 = Math.max(buy1, -p); //buy1, buy2, sell1, sell2 means after do the transaction, the money left in our account
            sell1 = Math.max(sell1, p + buy1);
            buy2 = Math.max(buy2, sell1 - p);
            sell2 = Math.max(sell2, buy2 + p);
        }
        return sell2;
    }

    //132
    public int minCut(String s) {
        if (s == null || s.length() == 0)
            return 0;
        boolean[][] dp = new boolean[s.length()][s.length()];
        for (int i = s.length() - 1; i >= 0; --i){
            for (int j = i; j < s.length(); ++j){
                if (s.charAt(i) == s.charAt(j) && (j - i <= 2 || dp[i+1][j-1]))
                    dp[i][j] = true;
            }
        }
        int[] dp1 = new int[s.length() + 1];
        Arrays.fill(dp1, s.length() + 1);
        dp1[0] = 0;
        for (int i = 1; i < dp1.length; ++i){
            for (int j = i; j < dp1.length; ++j){
                if (dp[i-1][j-1])
                    dp1[j] = Math.min(dp1[j], dp1[i-1] + 1);
            }
        }
        return dp1[dp1.length - 1] - 1;
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

    //145
    public List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        if (root == null)
            return res;
        Deque<TreeNode> st = new ArrayDeque<>();
        TreeNode pre = null; //last outputed node, when root null, check if peek's right is the last outputed node.if,then pop
        while (!st.isEmpty() || root != null){
            if (root != null){
                st.push(root);
                root = root.left;
            }
            else {
                TreeNode tn = st.peek();
                if (tn.right != null && tn.right != pre){
                    root = tn.right;
                }
                else {
                    pre = st.pop();
                    res.add(pre.val);
                }
            }
        }
        return res;
    }

    //154
    public int findMin(int[] nums) {
        if (nums == null || nums.length == 0)
            return -1;
        int l = 0, r = nums.length - 1, m;
        while (l < r){ //this q needs not having = for cases like [3,1,2]
            m = l + ((r - l) >> 1);
            if (nums[m] < nums[r])
                r = m;
            else if (nums[m] > nums[r])
                l = m + 1;
            else
                --r;
        }
        return nums[l];
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

    //188
    public int maxProfit(int k, int[] prices) {
        if (k <= 0 || prices == null || prices.length <= 1)
            return 0;
        if (k >= prices.length / 2){
            int res = 0;
            for (int i = 1; i < prices.length; ++i)
                res += prices[i] > prices[i-1]? prices[i] - prices[i-1]: 0;
            return res;
        }
        int[] buy = new int[k+1];
        int[] sell = new int[k+1];
        Arrays.fill(buy, Integer.MIN_VALUE);
        for (int p: prices){
            for (int i = 1; i <= k; ++i){
                buy[i] = Math.max(buy[i], sell[i-1] - p);
                sell[i] = Math.max(sell[i], buy[i] + p);
            }
        }
        return sell[sell.length - 1];
    }

    //198
    public int rob1(int[] nums) {
        if (nums == null || nums.length == 0)
            return 0;
        int[] dp = new int[3];
        dp[1] = dp[2] = nums[0];
        for (int i = 1; i < nums.length; ++i) {
            dp[2] = Math.max(dp[1], dp[0] + nums[i]);
            dp[0] = dp[1];
            dp[1] = dp[2];
        }
        return dp[2];
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

    //209
    public int minSubArrayLen(int s, int[] nums) {
        if (nums == null || nums.length == 0)
            return 0;
        int res = nums.length + 1, sum = 0;
        for (int l = 0, r = 0; r < nums.length; ++r){
            sum += nums[r];
            while (l <= r && sum >= s){
                res = Math.min(res, r - l + 1);
                sum -= nums[l++];
            }
        }
        return res > nums.length ? 0 : res;
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

    //213
    public int rob(int[] nums) {
        if (nums == null || nums.length == 0)
            return 0;
        if (nums.length == 1) // this is must, otherwise dp len = 1, cannot set dp[1]
            return nums[0];
        int c1 = robHelper(nums, 0, nums.length - 2);
        int c2 = robHelper(nums, 1, nums.length - 1);
        return Math.max(c1, c2);
    }

    private int robHelper(int[] nums, int l, int r){
        int[] dp = new int[3];
        dp[1] = dp[2] = nums[l]; //when only 1 element, still need to initialize dp[2]
        for (int i = l + 1; i <= r; ++i){
            dp[2] = Math.max(dp[1], dp[0] + nums[i]);
            dp[0] = dp[1];
            dp[1] = dp[2];
        }
        return dp[2];
    }

    //215
    public int findKthLargest(int[] nums, int k) {
        if (nums == null || nums.length == 0 || k <= 0)
            return -1;
        return kthSmallest(nums, 0, nums.length - 1, nums.length - k);
    }

    private int kthSmallest(int[] nums, int l, int r, int k){
        if (l > r) //k is larger than maximum number of elements in the array
            return -1;
        int pos = partition(nums, l, r);
        if (pos == k)
            return nums[pos];
        else if (pos < k)
            return kthSmallest(nums, pos + 1, r, k);
        else
            return kthSmallest(nums, l, pos - 1, k);
    }

    private int partition(int[] nums, int l, int r){
        int i = 0;
        for (int j = 0; j < r; ++j){
            if (nums[j] <= nums[r])
                swap1(nums, i++, j);
        }
        swap1(nums, i, r);
        return i;
    }

    private void swap1(int[] nums, int i, int j){
        int t = nums[i];
        nums[i] = nums[j];
        nums[j] = t;
    }

    //216
    public List<List<Integer>> combinationSum3(int k, int n) {
        List<List<Integer>> res = new ArrayList<>();
        if (k <= 0 || n <= 0)
            return res;
        combiHelper(k, n, 1, 0, new ArrayList<Integer>(), res);
        return res;
    }

    private void combiHelper(int k, int n, int i, int sum, List<Integer> combi, List<List<Integer>> res){
        if (k == 0){
            if (sum == n) //only when sum == target
                res.add(new ArrayList<>(combi));
            return;
        }
        for (int j = i; j <= 9; ++j){
            if (sum + j <= n){
                combi.add(j);
                combiHelper(k - 1, n, j + 1, sum + j, combi, res);
                combi.remove(combi.size() - 1);
            }
        }
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

    //225
    class MyStack {
        private Queue<Integer> q1 = new LinkedList<>();
        private Queue<Integer> q2 = new LinkedList<>();

        // Push element x onto stack.
        public void push(int x) {
            q1.offer(x);
        }

        // Removes the element on top of the stack.
        public void pop() {
            while (q1.size() > 1)
                q2.offer(q1.poll());
            q1.poll();
            Queue<Integer> t = q1; //only when pop the last one in q1 need swap q1 and q2
            q1 = q2;
            q2 = t;
        }

        // Get the top element.
        public int top() { //top does not need to swap
            while (q1.size() > 1)
                q2.offer(q1.poll());
            return q1.peek();
        }

        // Return whether the stack is empty.
        public boolean empty() {
            return q1.isEmpty();
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

    //228
    public List<String> summaryRanges(int[] nums) {
        List<String> res = new ArrayList<>();
        if (nums == null || nums.length == 0)
            return res;
        for (int l = 0, r = 1; r <= nums.length; ++r){
            if (r == nums.length || nums[r] != nums[r-1] + 1){
                res.add(summaryHelper(nums[l], nums[r-1]));
                l = r;
            }
        }
        return res;
    }

    private String summaryHelper(int l, int r){
        if (l == r)
            return String.valueOf(l);
        return l + "->" + r;
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

    //239
    public int[] maxSlidingWindow(int[] nums, int k) {
        if (nums == null || nums.length == 0 || k <= 0)
            return new int[0];
        int[] res = new int[nums.length - k + 1];
        Deque<Integer> deque = new ArrayDeque<>();
        for (int i = 0, j = 0; i < nums.length; ++i){
            while (!deque.isEmpty() && nums[i] > deque.peekLast()) //if equal, do not poll
                deque.pollLast();
            deque.offer(nums[i]);
            if (i >= k - 1){
                res[j++] = deque.peekFirst();
                if (nums[i - k + 1] == deque.peekFirst())
                    deque.pollFirst();
            }
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

    //244
    public class WordDistance {
        private Map<String, List<Integer>> hm;

        public WordDistance(String[] words) {
            hm = new HashMap<>();
            if (words == null || words.length == 0)
                return;
            for (int i = 0; i < words.length; ++i) {
                hm.putIfAbsent(words[i], new ArrayList<>());
                hm.get(words[i]).add(i);
            }
        }

        public int shortest(String word1, String word2) {
            if (word1 == null || word2 == null)
                return -1;
            List<Integer> l1 = hm.get(word1);
            List<Integer> l2 = hm.get(word2);
            int i1 = 0, i2 = 0, min = Integer.MAX_VALUE;
            while (i1 < l1.size() && i2 < l2.size()){
                int id1 = l1.get(i1), id2 = l2.get(i2); //compare value not i1, i2!
                min = Math.min(min, Math.abs(id1 - id2));
                if (id1 < id2)
                    ++i1;
                else
                    ++i2;
            }
            return min;
        }
    }

    //245
    public int shortestWordDistance(String[] words, String word1, String word2) {
        if (words == null || words.length == 0 || word1 == null || word2 == null)
            return -1;
        int i1 = -1, i2 = -1, min = Integer.MAX_VALUE;
        boolean isSame = word1.equals(word2);

        for (int i = 0; i < words.length; ++i){
            if (words[i].equals(word1)){
                if (isSame && i1 != -1)
                    min = Math.min(min, i - i1);
                else if (!isSame && i2 != -1)
                    min = Math.min(min, i - i2);
                i1 = i; //dont forget to set this after compare
            }
            else if (words[i].equals(word2)){
                if (i1 != -1)
                    min = Math.min(min, i - i1);
                i2 = i;
            }
        }
        return min;
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

    //249
    public List<List<String>> groupStrings(String[] strings) {
        List<List<String>> res = new ArrayList<>();
        if (strings == null || strings.length == 0)
            return res;
        Map<String, List<String>> hm = new HashMap<>();
        for (String s : strings){
            String pattern = getPattern(s);
            hm.putIfAbsent(pattern, new ArrayList<>());
            hm.get(pattern).add(s);
        }
        res.addAll(hm.values());
        return res;
    }

    private String getPattern(String s){
        if (s == null || s.length() == 0)
            return s;
        char c = s.charAt(0);
        StringBuilder sb = new StringBuilder();
        for (char cc : s.toCharArray()){
            sb.append((cc - c + 26) % 26); //"za", "ba" consider rotation
            sb.append(",");
        }
        return sb.toString();
    }

    //254
    public List<List<Integer>> getFactors(int n) {
        List<List<Integer>> res = new ArrayList<>();
        if (n <= 1)
            return res;
        factorHelper(n, 2, new ArrayList<Integer>(), res);
        return res;
    }

    private void factorHelper(int n, int start, List<Integer> combi, List<List<Integer>> res){
        if (n == 1){
            res.add(new ArrayList<>(combi));
            return;
        }
        for (int i = start; i * i <= n; ++i){
            if (n % i == 0){
                combi.add(i);
                factorHelper(n / i, i, combi, res); //combination, cannot have dup , must ascending; 12, 223 ok, 322 no
                combi.remove(combi.size() - 1);
            }
        }
        if (!combi.isEmpty()){
            combi.add(n);
            factorHelper(1, n, combi, res);
            combi.remove(combi.size() - 1);
        }
    }

    //256
    public int minCost(int[][] costs) {
        if (costs == null || costs.length == 0 || costs[0].length != 3)
            return 0;
        for (int i = 1; i < costs.length; ++i){
            for (int j = 0; j < costs[0].length; ++j){
                costs[i][j] += Math.min(costs[i-1][(j + 1)%3], costs[i-1][(j+2)%3]); //last row!
            }
        }
        int res = Integer.MAX_VALUE;
        for (int j = 0; j < costs[0].length; ++j){
            res = Math.min(costs[costs.length - 1][j], res);
        }
        return res;
    }

    //259
    public int threeSumSmaller(int[] nums, int target) {
        if (nums == null || nums.length == 0)
            return 0;
        Arrays.sort(nums);
        int res = 0;
        for (int i = 0; i < nums.length -2; ++i){
            int l = i + 1, r = nums.length - 1;
            while (l < r){
                int sum = nums[i] + nums[l] + nums[r];
                if (sum < target){
                    res += r - l; //this q requires index is unique. not filter dup by value!!!
                    ++l;
                }
                else {
                    --r;
                }
            }
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

    //268
    public int missingNumber(int[] nums) {
        if (nums == null || nums.length == 0)
            return 0;
        for (int i = 0; i < nums.length; ++i){
            if (nums[i] != i && nums[i] < nums.length && nums[nums[i]] != nums[i]){ //Need to check the nums[i] < len cuz it can be
                int t = nums[nums[i]];
                nums[nums[i]] = nums[i];
                nums[i] = t;
                --i;
            }
        }
        for (int i = 0; i < nums.length; ++i){
            if (nums[i] != i)
                return i;
        }
        return nums.length;
    }

    //269
    public String alienOrder(String[] words) {
        if (words == null || words.length < 1)
            return "";
        //construct graph
        Map<Character, Integer> indegree = new HashMap<>();
        Map<Character, List<Character>> children = new HashMap<>();
        for (String w : words){ //initialize array for "z" , "z" we still need output "z"; also indegree=0 need have key; no children need have empty list
            for (char c : w.toCharArray()){
                indegree.putIfAbsent(c, 0);
                children.putIfAbsent(c, new ArrayList<>());
            }
        }

        for (int i = 0; i < words.length - 1; ++i){
            int min = Math.min(words[i].length(), words[i+1].length()), j = 0;
            while (j < min && words[i].charAt(j) == words[i+1].charAt(j))
                ++j;
            if (j == min && words[i].length() > words[i+1].length()) //"abc", "ab" in this case, no valid sorting order!!
                return "";
            else if (j < min){ //Note: at this point, the first word's char must smaller than the second!!!
                char c1 = words[i].charAt(j), c2 = words[i+1].charAt(j);
                indegree.put(c1, indegree.get(c1) + 1);
                children.get(c2).add(c1);
            }
        }
        //topo sort
        Queue<Character> queue = new LinkedList<>();
        StringBuilder sb = new StringBuilder();
        for (Character k : indegree.keySet()){
            if (indegree.get(k) == 0)
                queue.offer(k);
        }
        while (!queue.isEmpty()){
            Character p = queue.poll();
            sb.append(p);
            for (Character c : children.get(p)){
                indegree.put(c, indegree.get(c) - 1);
                if (indegree.get(c) == 0)
                    queue.offer(c);
            }
        }
        return sb.length() == indegree.size()? sb.reverse().toString(): ""; //the sb order is big to small, needs reverse
    }

    //270
    public int closestValue(TreeNode root, double target) {
        if (root == null)
            return -1;
        double min = Math.abs(target - root.val);
        int res = root.val;

        while (root != null){
            //System.out.println(Math.abs(target - root.val) + " " + min);   4
            //if (Math.abs(target - root.val) > min)                      1
            //    break;                                                    3
            if (Math.abs(target - root.val) < min){ //if we want to find 3.2. at 1 is larger diff, but we cannot stop
                min = Math.abs(target - root.val);
                res = root.val;
            }
            if (target > root.val)
                root = root.right;
            else
                root = root.left;
        }
        return res;
    }

    //275
    public int hIndex(int[] citations) {
        if (citations == null || citations.length == 0)
            return 0;
        int l = 0, r = citations.length - 1, m;
        while (l <= r){
            m = l + ((r - l) >> 1);
            if (citations[m] > citations.length - m) //len is 1 more than index
                r = m - 1;
            else if (citations[m] < citations.length - m)
                l = m + 1;
            else
                return citations.length - m;
        }
        return citations.length - l;
    }

    //279
    public int numSquares(int n) {
        if (n <= 0)
            return 0;
        int[] dp = new int[n + 1];
        Arrays.fill(dp, n + 1);
        dp[0] = 0; // dont forget these intial
        for (int i = 1; i < dp.length; ++i){
            for (int j = 1; i - j * j >= 0; ++j){
                dp[i] = Math.min(dp[i], dp[i- j * j] + 1);
            }
        }
        return dp[dp.length - 1];
    }

    //281
    public class ZigzagIterator {
        private List<Iterator<Integer>> iters;
        private int idx;

        public ZigzagIterator(List<Integer> v1, List<Integer> v2) {
            iters = new ArrayList<>();
            if (v1 != null && !v1.isEmpty()) iters.add(v1.iterator());
            if (v2 != null && !v2.isEmpty()) iters.add(v2.iterator());
        }

        public int next() {
            return iters.get(idx++).next();
        }

        public boolean hasNext() {
            if (iters.isEmpty())
                return false;
            idx %= iters.size(); //mod size here first!!! solve next() case, also solve when remove last one %0's divide by 0 exp
            if (iters.get(idx).hasNext())
                return true;
            iters.remove(idx);
            return hasNext();
        }
    }

    //282
    public List<String> addOperators(String num, int target) {
        List<String> res = new ArrayList<>();
        if (num == null || num.length() == 0)
            return res;
        addHelper(num, target, 0, 0, 0,"", res);
        return res;
    }

    private void addHelper(String num, int target, int i,  long pre, long last, String combi, List<String> res){
        if (i == num.length()){
            if (pre == target)
                res.add(combi);
            return;
        }
        for (int j = i + 1; j <= num.length(); ++j){
            String sub = num.substring(i, j);
            if (sub.length() > 1 && sub.charAt(0) == '0') //multi-length string to int always check leading 0
                break;
            long x = Long.parseLong(sub); //multi-length string to int always consider overflow
            if (x > Integer.MAX_VALUE)
                break;
            if (i == 0)
                addHelper(num, target, j, x, x, sub, res);
            else {
                addHelper(num, target, j, pre + x, x, combi + "+" + sub, res);
                addHelper(num, target, j, pre - x, -x, combi + "-" + sub, res);
                addHelper(num, target, j, pre - last + last * x, last * x, combi + "*" + sub, res);
            }
        }
    }

    //286
    public void wallsAndGates(int[][] rooms) {
        if (rooms == null || rooms.length == 0 || rooms[0].length == 0)
            return;
        //bfs start from all empty in the queue, with a level = 1. whoever grab first means it's closer
        Queue<int[]> queue = new LinkedList<>();
        int cur = 0, next = 0, lvl = 1;
        for (int i = 0; i < rooms.length; ++i){
            for (int j = 0; j < rooms[0].length; ++j){
                if (rooms[i][j] == 0){
                    queue.offer(new int[]{i, j});
                    ++cur;
                }
            }
        }
        int[][] off = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        while (!queue.isEmpty()){
            int[] p = queue.poll();
            for (int[] o : off){
                int x = p[0] + o[0], y = p[1] + o[1];
                if (x >= 0 && x < rooms.length && y >= 0 && y < rooms[0].length && rooms[x][y] == Integer.MAX_VALUE){
                    rooms[x][y] = lvl;
                    queue.offer(new int[]{x, y});
                    ++next;
                }
            }
            if (--cur == 0){
                cur = next;
                next = 0;
                ++lvl;
            }
        }
    }

    //288
    public class ValidWordAbbr {
        private Map<String, Set<String>> hm;

        public ValidWordAbbr(String[] dictionary) {
            hm = new HashMap<>();
            if (dictionary == null || dictionary.length == 0)
                return;
            for (String s : dictionary){
                String abbr = getAbbr(s);
                hm.putIfAbsent(abbr, new HashSet<>());
                hm.get(abbr).add(s);
            }
        }

        private String getAbbr(String s){
            if (s == null || s.length() <= 2)
                return s;
            StringBuilder sb = new StringBuilder();
            sb.append(s.charAt(0));
            sb.append(s.length() - 2);
            sb.append(s.charAt(s.length() - 1));
            return sb.toString();
        }

        public boolean isUnique(String word) {
            String abbr = getAbbr(word);
            if (!hm.containsKey(abbr)) //no other words in dict ==> abbr not exist in the dict; exist but is the word itself
                return true;
            else if (hm.get(abbr).size() == 1 && hm.get(abbr).contains(word))
                return true;
            return false;
        }
    }

    //290
    public boolean wordPattern(String pattern, String str) {
        if (pattern == null || str == null || pattern.length() == 0 || str.length() == 0)
            return false;
        String[] tokens = str.split("\\s+");
        if (pattern.length() != tokens.length) //need to first check this to prevent array length go off bound
            return false;
        Map<Character, String> hm = new HashMap<>();
        for (int i = 0; i < pattern.length(); ++i){
            if (hm.containsKey(pattern.charAt(i))){
                if (!hm.get(pattern.charAt(i)).equals(tokens[i]))
                    return false;
            }
            else {
                if (hm.containsValue(tokens[i]))
                    return false;
                hm.put(pattern.charAt(i), tokens[i]);
            }
        }
        return true;
    }

    //295
    public class MedianFinder {
        private Queue<Integer> minq = new PriorityQueue<>();
        private Queue<Integer> maxq = new PriorityQueue<>(Collections.reverseOrder()); //note maxq

        // Adds a number into the data structure.
        public void addNum(int num) {
            if (maxq.isEmpty() || num <= maxq.peek())
                maxq.offer(num);
            else
                minq.offer(num);
            if (maxq.size() > minq.size() + 1)
                minq.offer(maxq.poll());
            if (minq.size() > maxq.size())
                maxq.offer(minq.poll());
        }

        // Returns the median of current data stream
        public double findMedian() {
            if (maxq.size() == minq.size())
                return (maxq.peek() + minq.peek()) / 2.0;
            else
                return maxq.peek();
        }
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

    //304
    public class NumMatrix {
        private int[][] dp ;
        public NumMatrix(int[][] matrix) {
            if (matrix == null || matrix.length == 0 || matrix[0].length == 0)
                return;
            dp = new int[matrix.length + 1][matrix[0].length + 1];

            for (int i = 0; i < matrix.length; ++i){
                int t = 0;
                for (int j = 0; j < matrix[0].length; ++j){
                    dp[i+1][j+1] = matrix[i][j] + t + dp[i][j+1];
                    t += matrix[i][j]; //must use seperate t to sum up to the left ONLY to this row!!!
                }
            }
        }

        public int sumRegion(int row1, int col1, int row2, int col2) {
            if (dp == null)
                return 0;
            return dp[row2+1][col2+1] - dp[row2+1][col1] - dp[row1][col2+1] + dp[row1][col1];
        }
    }

    //311
    public int[][] multiply(int[][] A, int[][] B) {
        if (A == null || B == null || A.length == 0 || B.length == 0)
            return new int[0][]; //note how 2-d empty array is created
        int[][] res = new int[A.length][B[0].length];
        for (int i = 0; i < A.length; ++i){
            for (int j = 0; j < A[0].length; ++j){
                if (A[i][j] != 0){
                    for (int k = 0; k < B[0].length; ++k){
                        if (B[j][k] != 0)
                            res[i][k] += A[i][j] * B[j][k];
                    }
                }
            }
        }
        return res;
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

    //323
    public int countComponents(int n, int[][] edges) {
        if (n <= 0 || edges == null || edges.length == 0 || edges[0].length == 0)
            return n;
        boolean[] visited = new boolean[n];
        List<Integer>[] children = new List[n];
        for (int i = 0; i < n; ++i)
            children[i] = new ArrayList<>();
        for (int[] e : edges){
            children[e[0]].add(e[1]);
            children[e[1]].add(e[0]);
        }
        int res = 0;
        for (int i = 0; i < visited.length; ++i){
            if (!visited[i]){
                ++res;
                countHelper(i, visited, children);
            }
        }
        return res;
    }

    private void countHelper(int i, boolean[] visited, List<Integer>[] children){
        visited[i] = true; //dfs mask self, go to children
        for (int c : children[i]){
            if (!visited[c]){
                countHelper(c, visited, children);
            }
        }
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

    //337
    public int rob(TreeNode root) {
        if (root == null)
            return 0;
        int[] res = robHelper(root);
        return res[1];
    }

    private int[] robHelper(TreeNode root){
        int[] res = {0, 0}; //{pre, cur}
        if (root == null)
            return res;
        int[] lr = robHelper(root.left);
        int[] rr = robHelper(root.right);
        res[0] = lr[1] + rr[1]; //pre is last cur sum
        res[1] = Math.max(res[0], lr[0] + rr[0] + root.val);
        return res;
    }

    //340
    public int lengthOfLongestSubstringKDistinct(String s, int k) {
        if (s == null || s.length() == 0 || k <= 0)
            return 0;
        Map<Character, Integer> hm = new HashMap<>();
        int res = 0;
        for (int l = 0, r = 0; r < s.length(); ++r){
            hm.put(s.charAt(r), hm.getOrDefault(s.charAt(r), 0) + 1);

            while (hm.size() > k){
                char c = s.charAt(l++);
                int cnt = hm.get(c) - 1;
                if (cnt == 0)
                    hm.remove(c);
                else
                    hm.put(c, cnt);
            }
            res = Math.max(res, r - l + 1);
        }
        return res;
    }

    //341
    public class NestedIterator implements Iterator<Integer> {
        private Deque<Iterator<NestedInteger>> st;
        private Iterator<NestedInteger> iter;
        private NestedInteger ni;

        public NestedIterator(List<NestedInteger> nestedList) {
            if (nestedList == null)
                return;
            st = new ArrayDeque<>();
            iter = nestedList.iterator(); //iter is possibly null
        }

        @Override
        public Integer next() {
            //after hasNext(), either return false, or ni is set
            int x = ni.getInteger();
            ni = null;
            return x;
        }

        @Override
        public boolean hasNext() {
            if (iter == null)
                return false;
            if (!iter.hasNext()) {
                if (st.isEmpty())
                    return false;
                iter = st.pop();
                return hasNext();
            }
            ni = iter.next();
            if (ni.isInteger())
                return true;
            else {
                st.push(iter);
                iter = ni.getList().iterator();
                return hasNext(); //iter starts same problem when find a next iter
            }
        }
    }

    //344
    public String reverseString(String s) {
        if (s == null || s.length() <= 1)
            return s;
        char[] ca = s.toCharArray();
        int l = 0, r = ca.length - 1;
        while (l < r){
            char t = ca[l];
            ca[l] = ca[r];
            ca[r] = t;
            ++l;
            --r;
        }
        return new String(ca);
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

    //347
    public List<Integer> topKFrequent(int[] nums, int k) {
        List<Integer> res = new ArrayList<>();
        if (nums == null || nums.length == 0 || k <= 0)
            return res;
        Map<Integer, Integer> hm = new HashMap<>();
        for (int x : nums)
            hm.put(x, hm.getOrDefault(x, 0) + 1);
        Queue<Map.Entry<Integer, Integer>> pq = new PriorityQueue<>((e1, e2) -> e1.getValue() - e2.getValue());
        for (Map.Entry<Integer, Integer> e: hm.entrySet()){
            if (pq.size() < k)
                pq.offer(e);
            else if (e.getValue() > pq.peek().getValue()){
                pq.poll();
                pq.offer(e);
            }
        }
        for (Map.Entry<Integer, Integer> e : pq)
            res.add(e.getKey());
        return res;
    }

    //352
    public class SummaryRanges {
        private TreeMap<Integer, Interval> tm;

        /** Initialize your data structure here. */
        public SummaryRanges() {
            tm = new TreeMap<>();
        }

        public void addNum(int val) {
            if (tm.containsKey(val))
                return;
            Integer lower = tm.lowerKey(val);
            Integer higher = tm.higherKey(val);

            //4 cases, if join left and right intervals; join left; join right; new interval itself
            if (lower != null && higher != null && tm.get(lower).end + 1 == val && val + 1 == higher){
                tm.get(lower).end = tm.get(higher).end;
                tm.remove(higher);
            }
            else if (lower != null && val <= tm.get(lower).end + 1){
                tm.get(lower).end = Math.max(tm.get(lower).end, val);
            }
            else if (higher != null && val + 1 == higher){
                tm.put(val, new Interval(val, tm.get(higher).end));
                tm.remove(higher);
            }
            else
                tm.put(val, new Interval(val, val));
        }

        public List<Interval> getIntervals() {
            return new ArrayList<>(tm.values()); //from Collection to List
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

    //358
    public String rearrangeString(String str, int k) {
        if (str == null || str.length() == 0 || k <= 1)
            return str;
        //count freq of each char, and need to output high-freq char first using pq
        Map<Character, Integer> hmfreq = new HashMap<>();
        for (char c: str.toCharArray()){
            hmfreq.put(c, hmfreq.getOrDefault(c, 0) + 1);
        }
        Queue<Map.Entry<Character, Integer>> pq = new PriorityQueue<>((e1, e2)->e2.getValue() - e1.getValue());
        pq.addAll(hmfreq.entrySet());
        //use another hm to store when the next available time(index) that a previous char can insert again
        Map<Integer, Map.Entry<Character, Integer>> hmnext = new HashMap<>();
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < str.length(); ++i){
            if (hmnext.containsKey(i))
                pq.offer(hmnext.get(i));
            if (pq.isEmpty())
                return ""; //not enough space
            Map.Entry<Character, Integer> e = pq.poll();
            sb.append(e.getKey());
            if (e.getValue() > 1){
                e.setValue(e.getValue() - 1); //entry use setValue()
                hmnext.put(i + k, e);
            }
        }
        return sb.toString();
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

    //362
    public class HitCounter {
        private TreeMap<Integer, Integer> tm;
        private int cnt;

        /** Initialize your data structure here. */
        public HitCounter() {
            tm = new TreeMap<>();
        }

        /** Record a hit.
         @param timestamp - The current timestamp (in seconds granularity). */
        public void hit(int timestamp) {
            tm.put(timestamp, tm.getOrDefault(timestamp, 0) + 1);
            ++cnt;
        }

        /** Return the number of hits in the past 5 minutes.
         @param timestamp - The current timestamp (in seconds granularity). */
        public int getHits(int timestamp) {
            while (!tm.isEmpty() && timestamp - tm.firstKey() >= 300) //need to first check tm is empty before get firstKey()
                cnt -= tm.pollFirstEntry().getValue();
            return cnt;
        }
    }

    //367
    public boolean isPerfectSquare(int num) {
        if (num < 0)
            return false;
        if (num == 0)
            return true;
        long l = 1, r = num, m; //change to long prevent overflow
        while (l <= r){
            m = l + ((r - l) >> 1);
            if (m * m < num) //cannot use m < num / m when m = 2, num = 5. lost by integer division
                l = m + 1;
            else if (m * m > num)//l cannot start from 0, here will divide by 0
                r = m - 1;
            else
                return true;
        }
        return false;
    }

    //377
    public int combinationSum4(int[] nums, int target) {
        if (nums == null || nums.length == 0)
            return 0;
        Arrays.sort(nums); //this is must
        int[] dp = new int[target + 1];
        dp[0] = 1;
        for (int i = 1; i <= target; ++i){
            for (int j = 0; j < nums.length && nums[j] <= i; ++j){ //nums[j] <= i!! and requires sort!
                dp[i] += dp[i - nums[j]];
            }
        }
        return dp[target];
    }

    //379
    public class PhoneDirectory {
        private BitSet bs;
        private int size, cap;

        /** Initialize your data structure here
         @param maxNumbers - The maximum numbers that can be stored in the phone directory. */
        public PhoneDirectory(int maxNumbers) {
            cap = maxNumbers;
            bs = new BitSet();
        }

        /** Provide a number which is not assigned to anyone.
         @return - Return an available number. Return -1 if none is available. */
        public int get() {
            if (size >= cap)
                return -1;
            int x = bs.nextClearBit(0); //nextClearBit(int fromIndex) return the first available unset bit's index
            bs.set(x); //set the bit!!
            ++size;
            return x;
        }

        /** Check if a number is available or not. */
        public boolean check(int number) {
            return number >= 0 && number < cap && !bs.get(number);
        }

        /** Recycle or release a number. */
        public void release(int number) {
            if (number >= 0 && number < cap && bs.get(number)){
                bs.clear(number);
                --size;
            }
        }
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

    //389
    public char findTheDifference(String s, String t) {
        if (s == null || t == null)
            return 0;
        char res = 0; // char c = 0; is a valid statement!
        for (char c : s.toCharArray())
            res ^= c;
        for (char c : t.toCharArray())
            res ^= c;
        return res;
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

    //401
    public List<String> readBinaryWatch(int num) {
        //directly count a number's bit by concat hh<<6|mm and use Integer.bitCount(i)
        List<String> res = new ArrayList<>();
        if (num < 0)
            return res;
        for (int h = 0; h < 12; ++h){ //hh from 0 - 11
            for (int m = 0; m < 60; ++m){
                int x = h << 6 | m;
                if (Integer.bitCount(x) == num)
                    res.add(String.format("%d:%02d", h, m));
            }
        }
        return res;
    }

    //438
    public List<Integer> findAnagrams(String s, String p) {
        List<Integer> res = new ArrayList<>();
        if (s == null || s.length() == 0 || p == null || p.length() == 0)
            return res;
        Map<Character, Integer> hm = new HashMap<>();
        for (char c : p.toCharArray()){
            hm.put(c, hm.getOrDefault(c, 0) + 1);
        }
        int cnt = 0;
        for (int l = 0, r = 0; r < s.length(); ++r){
            char rc = s.charAt(r);
            if (!hm.containsKey(rc))
                continue;
            else
                hm.put(rc, hm.get(rc) - 1);
            if (hm.get(rc) >= 0)
                ++cnt;
            while (cnt == p.length()){
                char lc = s.charAt(l);
                if (!hm.containsKey(lc))
                    ++l;
                else if (hm.get(lc) < 0) {
                    hm.put(lc, hm.get(lc) + 1);
                    ++l;
                }
                else {
                    if (r - l + 1 == p.length())
                        res.add(l);
                    break;
                }
            }
        }
        return res;
    }

    //442
    public List<Integer> findDuplicates(int[] nums) {
        List<Integer> res = new ArrayList<>();
        //mark the dest index to be negative. for every nums[i], take the abs and check dest is neg. if yes, add res. else negate dest.
        for (int i = 0; i < nums.length; ++i){
            int index = Math.abs(nums[i]) - 1;
            if (nums[index] < 0)
                res.add(Math.abs(nums[i]));
            else
                nums[index] = -nums[index];
        }
        for (int i = 0; i < nums.length; ++i){
            nums[i] = Math.abs(nums[i]);
        }
        return res;
    }

    //448
    public List<Integer> findDisappearedNumbers(int[] nums) {
        List<Integer> res = new ArrayList<>();
        if (nums == null || nums.length == 0)
            return res;
        for (int i = 0; i < nums.length; ++i){
            int index = Math.abs(nums[i]) - 1; //must do this, the dest can be already negative
            if (nums[index] > 0)
                nums[index] = -nums[index];
        }
        for (int i = 0; i < nums.length; ++i){
            if (nums[i] > 0)
                res.add(i + 1);
            else
                nums[i] = -nums[i];
        }
        return res;
    }

    //463
    public int islandPerimeter(int[][] grid) {
        //check a 1 is at edge
        if (grid == null || grid.length == 0 || grid[0].length == 0)
            return 0;
        int res = 0;
        for (int i = 0; i < grid.length; ++i){
            for (int j = 0; j < grid[0].length; ++j){
                if (grid[i][j] == 1){
                    if (i == 0 || grid[i-1][j] == 0) ++res; //must individually check every direction cuz they can all be true
                    if (i == grid.length - 1 || grid[i+1][j] == 0) ++res;
                    if (j == 0 || grid[i][j-1] == 0) ++res;
                    if (j == grid[0].length - 1 || grid[i][j+1] == 0) ++res;
                }
            }
        }
        return res;
    }
}
