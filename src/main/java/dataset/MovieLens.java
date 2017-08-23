package dataset;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

public class MovieLens {

	public final static RealMatrix rates;

	public final static Map<Integer, Integer> dictionary = new HashMap<Integer, Integer>();
	static {
		List<Integer[]> items = new ArrayList<Integer[]>();
		int[] sz = initLines(items);
		rates = MatrixUtils.createRealMatrix(sz[0], sz[1]);
		initMatrix(items, rates);
	}

	static int[] initLines(List<Integer[]> lines) {
		int[] ret = new int[] { 0, 0 };
		List<Integer[]> r = new ArrayList<Integer[]>();
		try (InputStream is = MovieLens.class.getResourceAsStream("/movielens/ratings.csv")) {
			BufferedReader br = new BufferedReader(new InputStreamReader(is));
			String ln = null;
			String[] it;
			int userid = -1;
			ln = br.readLine();
			while ((ln = br.readLine()) != null) {
				it = ln.trim().split(",");
				try {
					Integer id = Integer.valueOf(it[0]);
					if (id > userid) {
						userid = id;
					}
					Integer id2 = Integer.valueOf(it[1]);
					if (!dictionary.containsKey(id2)) {
						dictionary.put(id2, dictionary.size());
					}
					id2 = dictionary.get(id2);
					Integer v = ((Float) (10 * Float.valueOf(it[2]))).intValue();
					r.add(new Integer[] { id, id2, v });
				} catch (Exception e) {
					e.printStackTrace();
					continue;
				}
			}
			br.close();
			ret[0] = userid;
			ret[1] = dictionary.size();
			lines.addAll(r);
		} catch (Exception e) {
			e.printStackTrace();
			throw new RuntimeException(e);
		}
		return ret;
	}

	static void initMatrix(List<Integer[]> items, RealMatrix r) {
		for (int i = 0; i < items.size(); i++) {
			Integer[] item = items.get(i);
			r.setEntry(item[0] - 1, item[1], 1d);
		}
	}

}
