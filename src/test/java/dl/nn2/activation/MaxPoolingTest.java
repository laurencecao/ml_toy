package dl.nn2.activation;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import dl.nn2.graph.MatrixDataEdge;
import dl.nn2.graph.Upsampling;

public class MaxPoolingTest {

	@Before
	public void setUp() throws Exception {
	}

	@After
	public void tearDown() throws Exception {
	}

	@Test
	public void test() {
		MaxPooling pool = new MaxPooling(2);
		RealMatrix m = MatrixUtils
				.createRealMatrix(new double[][] { { 1, 2, 3, 4 }, { 5, 6, 7, 8 }, { 3, 4, 5, 6 }, { 9, 2, 7, 4 }, });
		System.out.println(MatrixDataEdge.pretty0(m));
		System.out.println("-------------------------");

		RealMatrix mm = pool.forward(m);
		System.out.println(MatrixDataEdge.pretty0(mm));

		Upsampling up = new Upsampling(2);
		MatrixDataEdge mm1 = up.eval(new MatrixDataEdge("", mm));
		System.out.println(MatrixDataEdge.pretty0(mm1.asMat(0)));

		System.out.println("-------------------------");

		mm = pool.backward(m);
		System.out.println(MatrixDataEdge.pretty0(mm));

	}

}
