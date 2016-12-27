/**
 * Created by zplchn on 12/26/16.
 */
public class TrieNode {
    boolean isWord;
    TrieNode[] children;

    TrieNode(){
        children = new TrieNode[26];
    }
}
