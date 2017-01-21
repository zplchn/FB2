package lintcode;

import java.util.*;


/**
 * Created by zplchn on 1/20/17.
 */
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
    public ListNode mergeKLists(ListNode[] lists) {
        if (lists == null || lists.length == 0)
            return null;
        Queue<ListNode> pq = new PriorityQueue<>((l1, l2)->l1.val - l2.val);
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
}
