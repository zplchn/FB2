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

    public static void main(String[] args){
        Thumbtack tt = new Thumbtack();
        int[] t1 = {0};
        int[] t2 = {0, 1};
        int[] t3 = {1, 2, 4, 6};
        int[] t4 = {1, 1, 2, 3, 4};
        //int [][] tests = {t1, t2, t3, t4};
        int [][] tests = {t3};

        for (int[] t: tests) {
            MedianFinder tm = tt.new MedianFinder(); //an inner class object stays with the instance of the outer class object. Must use outer.new Inner() to create
            for (int x : t)
                tm.insert(x);
            System.out.println(tm.getMean());
            //System.out.println(tm.getMedian());
            System.out.println("-------------");
        }

    }
}
