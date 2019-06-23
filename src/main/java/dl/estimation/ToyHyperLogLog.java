/**
 * 
 */
package dl.estimation;

import java.io.IOException;
import java.util.function.Function;
import java.util.stream.LongStream;

import com.google.common.hash.HashFunction;
import com.google.common.hash.Hashing;

/**
 * @author Laurence Cao
 *
 */
public class ToyHyperLogLog {

    // m > (1.30/E)^2
    final static Function<Double, Double> M = E -> Math.pow(1.30d / E, 2);
    final static HashFunction hashFunction = Hashing.murmur3_128();
    final static protected long LONG_MASK = 1L << (Long.SIZE - 1);

    protected int m;
    protected int[] buckets;
    protected int mask;
    protected double alpha;

    double computeAlpha() {
        double alpha;
        switch (m) {
            case (1 << 4):
                alpha = 0.673;
                break;
            case (1 << 5):
                alpha = 0.697;
                break;
            case (1 << 6):
                alpha = 0.709;
                break;
            default:
                alpha = (0.7213 / (1 + 1.079 / m));
        }
        return alpha;
    }

    public ToyHyperLogLog(double err) {
        m = Double.valueOf(Math.ceil(M.apply(err))).intValue();
        for (int i = 0; i < Integer.MAX_VALUE; i++) {
            if (Math.pow(2, i) >= m) {
                m = Double.valueOf(Math.pow(2, i)).intValue();
                break;
            }
        }
        System.out.println("using buckest: " + m);
        buckets = new int[m];
        mask = m - 1;
        alpha = computeAlpha();
    }

    public void add(long value) {
        long key = hashFunction.hashLong(value).asLong();
        int idx = (int)(key & mask);
        key |= LONG_MASK;
        // log2(buckets.length)
        int bits = Integer.numberOfTrailingZeros(m);
        key >>= bits;

        int lb = Long.numberOfTrailingZeros(key) + 1;
        if (buckets[idx] < lb) {
            buckets[idx] = lb;
        }
    }

    public double estimate() {
        double sum = 0;
        for (int i = 0; i < buckets.length; i++) {
            sum += 1.0 / (1L << buckets[i]);
        }

        double result = alpha * buckets.length * buckets.length / sum;
        return Math.round(result);
    }

    public static void main(String[] args) throws IOException {
        ToyHyperLogLog hll = new ToyHyperLogLog(0.001);
        long numberOfElements = 100_000_000;
        LongStream.range(0, numberOfElements).forEach(element -> {
            hll.add(element);
        });
        System.out.println(hll.estimate());
    }

}
