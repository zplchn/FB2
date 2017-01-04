import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by zplchn on 1/1/17.
 */
public class Thumbtack {

    //input data stream range [0, 1000), implement constant space getMean, insert, getMedian operation.
    public class MedianFinder{
        private int[] nums;
        private int size;

        public MedianFinder(){
            nums = new int[1000];
        }

        public void insert(int x){
            ++nums[x];
            ++size;
        }

        public double getMean(){
            double avg = 0;
            for (int i = 1; i < nums.length; ++i) {
                avg += (double) i * nums[i] / size; //need cast
            }
            return avg;
        }

        public double getMedian() {
            int half = size / 2, m1 = -1, m2 = -1;
            for (int i = 0; i < nums.length; ++i) {
                if (nums[i] != 0) {
                    half -= nums[i];
                    if (half == 0)
                        m1 = i;
                    else if (half < 0) {
                        m1 = m1 < 0? i: m1;
                        m2 = i;
                        break;
                    }
                }
            }
            return size % 2 == 0 ? (m1 + m2) / 2.0 : m2;
        }
    }

    //auto-complete
    class TrieNode{
        List<String> startsWith;
        TrieNode[] children;

        TrieNode(){
            startsWith = new ArrayList<>();
            children = new TrieNode[26];
        }
    }
    public class AutoComplete{
        private TrieNode root;

        public AutoComplete(){
            root = new TrieNode();
        }

        public AutoComplete(String[] strs){
            this(); //chain ctor from same class
            if (strs == null)
                return;
            for (String s : strs){
                TrieNode tr = root;
                for (int i = 0; i < s.length(); ++i){
                    int off = s.charAt(i) - 'a';
                    if (tr.children[off] == null)
                        tr.children[off] = new TrieNode();
                    tr = tr.children[off];
                    tr.startsWith.add(s);
                }
            }
        }

        public List<String> findByPrefix(String prefix){
            List<String> res = new ArrayList<>();
            if (prefix == null || prefix.length() == 0)
                return res;
            TrieNode tr = root;
            for (int i = 0; i < prefix.length(); ++i){
                int off = prefix.charAt(i) - 'a';
                if (tr.children[off] == null)
                    return res;
                tr = tr.children[off];
            }
            return tr.startsWith;
        }
    }

    /*
    The following iterative sequence is defined for the set of positive integers:
n → n/2 (n is even)
n → 3n + 1 (n is odd)
Using the rule above and starting with 13, we generate the following sequence:
13 → 40 → 20 → 10 → 5 → 16 → 8 → 4 → 2 → 1
It can be seen that this sequence (starting at 13 and finishing at 1) contains 10 terms. Although it has not been proved yet (Collatz Problem), it is thought that all starting numbers finish at 1.
Which starting number, under one million, produces the longest chain?

        13 → 40 → 20 → 10 → 5 → 16 → 8 → 4 → 2 → 1
     */
    public int longestCollatzSequence(int n){
        if (n <= 0)
            return n;
        Map<Integer, Integer> hm = new HashMap<>();
        hm.put(1, 1);
        int max = 1, res = 1;
        long seq;
        for (int i = 2; i <= n; ++i){
            seq = i;
            int k = 0;
            while (seq != 1 && seq >= i) {
                if (seq % 2 == 0)
                    seq /= 2;
                else
                    seq = 3 * seq + 1;
                ++k;
            }
            hm.put(i, k + hm.get((int)seq));
            if (hm.get(i) > max) {
                max = hm.get(i);
                res = i;
            }
        }
        return res;
    }

    //nearest palindrome number
    public int nearestPalindrome(int x){
        if (x < 10)
            return x;
        String s = String.valueOf(x);
        if (isPalindrome(s))
            return x;
        String firstHalf = s.substring(0, s.length() / 2), firstMidHalf = firstHalf;
        StringBuilder sb = new StringBuilder();
        sb.append(firstHalf);
        if (s.length() % 2 == 1) {
            char c = s.charAt(s.length() / 2);
            sb.append(c);
            firstMidHalf += c;
        }
        sb.append(new StringBuilder(firstHalf).reverse());
        int nx = Integer.parseInt(sb.toString());
        int nfirstH = Integer.parseInt(firstMidHalf) + (nx < x? 1 : -1);
        String nfh = String.valueOf(nfirstH);
        String res = "";
        if (s.length() % 2 == 0)
            res = nfh + new StringBuilder(nfh).reverse();
        else {
            String sub = nfh.substring(0, nfh.length() - 1);
            res = sub + nfh.charAt(nfh.length() - 1) + new StringBuilder(sub).reverse();
        }
        int nx2 = Integer.parseInt(res);
        int diff = Math.abs(nx - x) - Math.abs(nx2 - x);
        return diff > 0 ? nx2: nx;
    }

    private boolean isPalindrome(String s){
        int l = 0, r = s.length() - 1;
        while (l < r){
            if (s.charAt(l++) != s.charAt(r--))
                return false;
        }
        return true;
    }


    public static void main(String[] args){
        Thumbtack tt = new Thumbtack();
        int[] t1 = {0};
        int[] t2 = {0, 1};
        int[] t3 = {1, 2, 4, 6};
        int[] t4 = {1, 1, 2, 3, 4};
        //int [][] tests = {t1, t2, t3, t4};
        int [][] tests = {t3};
        System.out.println(tt.nearestPalindrome(100));
        //System.out.println(tt.longestCollatzSequence(1000000));

//        for (int[] t: tests) {
//            MedianFinder tm = tt.new MedianFinder(); //an inner class object stays with the instance of the outer class object. Must use outer.new Inner() to create
//            for (int x : t)
//                tm.insert(x);
//            System.out.println(tm.getMean());
//            //System.out.println(tm.getMedian());
//            System.out.println("-------------");
//        }
        String[] strs = {"thumb", "thumbtack", "thu", "tmb", "t"};
        AutoComplete ac = tt.new AutoComplete(strs);
        for (String s: ac.findByPrefix("thumb")){ //this for will only executed once
            System.out.println(s);
        }



    }
}
