package lintcode;

import java.util.ArrayList;

/**
 * Created by zplchn on 1/21/17.
 */
public class DirectedGraphNode {
    int label;
    ArrayList<DirectedGraphNode> neighbors;

    public DirectedGraphNode(int x){
        label = x;
        neighbors = new ArrayList<>();
    }
}
