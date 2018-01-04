package dl.perf;

public class ComparePerfTest {

	/**
	 * <pre>
	 * when not same size
	 * compare integer: 321ms
	 * compare string: 388ms
	 * 
	 * when same size:
	 * compare integer: 311ms
	 * compare string: 1852ms
	 * </pre>
	 * 
	 * @param args
	 */
	public static void main(String[] args) {
		long loop = 1000000000l; // 10 billion
		long ts = System.currentTimeMillis();
		compareInteger(loop, 10000, 20000);
		ts = System.currentTimeMillis() - ts;
		System.out.println("compare integer: " + ts + "ms");
		
		ts = System.currentTimeMillis();
		compareString(loop, "baaabb", "abbaab");
		ts = System.currentTimeMillis() - ts;
		System.out.println("compare string: " + ts + "ms");
	}

	public static boolean compareString(long N, String a, String b) {
		boolean ret = false;
		for (long i = 0; i < N; i++) {
			ret = a.equals(b);
		}
		return ret;
	}

	public static boolean compareInteger(long N, int a, int b) {
		boolean ret = false;
		for (long i = 0; i < N; i++) {
			ret = a == b;
		}
		return ret;
	}

}
