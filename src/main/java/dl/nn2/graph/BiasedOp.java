package dl.nn2.graph;

import java.util.Arrays;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

public class BiasedOp extends TracedComputation {

	final static String ADD_NAME = "biased_data";
	final static String TRIM_NAME = "nobiased_data";
	protected boolean addBiased;
	protected String memo;

	public BiasedOp(boolean add, String memo) {
		this.addBiased = add;
		this.memo = memo;
	}

	@Override
	protected MatrixDataEdge eval0(MatrixDataEdge data) {
		RealMatrix m = data.asMat(0);
		RealMatrix ret = null;
		String n = null;
		int[] shape = { m.getRowDimension(), m.getColumnDimension() };

		if (addBiased) {
			ret = MatrixUtils.createRealMatrix(shape[0] + 1, shape[1]);
			double[] one = ret.getRow(0);
			Arrays.fill(one, 1d);
			ret.setRow(0, one);
			ret.setSubMatrix(m.getData(), 1, 0);
			n = ADD_NAME;
		} else {
			m = m.getSubMatrix(1, shape[0] - 1, 0, shape[1] - 1);
			ret = m.copy();
			n = TRIM_NAME;
		}

		return new MatrixDataEdge(n, ret);
	}

	@Override
	public String type() {
		return "Biased" + (addBiased ? "+" : "-") + " #" + memo + "#";
	}

	@Override
	public String name() {
		return type();
	}

	@Override
	public int[] inShape() {
		return new int[] { -1, -1 };
	}

	@Override
	public int[] outShape() {
		return new int[] { -1, -1 };
	}

}
