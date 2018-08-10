package dl.nn2.graph;

import java.util.Arrays;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

public class Mul2Op extends TracedComputation {

	protected MatrixDataEdge w;
	protected MatrixDataEdge d;
	protected boolean addBiased;
	protected boolean transposed;
	protected String memo;

	public Mul2Op(boolean biased, boolean transposed, MatrixDataEdge w, MatrixDataEdge d, String memo) {
		this.addBiased = biased;
		this.transposed = transposed;
		this.w = new MatrixDataEdge("mul2op", w.asMat(0));
		this.d = new MatrixDataEdge("mul2op", d.asMat(0));
		this.memo = memo;

		if (this.addBiased) {
			RealMatrix dd = this.d.asMat(0);
			RealMatrix d1 = MatrixUtils.createRealMatrix(dd.getRowDimension() + 1, dd.getColumnDimension());
			double[] one = d1.getRow(0);
			Arrays.fill(one, 1d);
			d1.setRow(0, one);
			d1.setSubMatrix(dd.getData(), 1, 0);
			this.d.update(d1);
		}
		if (this.transposed) {
			this.d.update(this.d.asMat(0).transpose());
		}
	}

	@Override
	public String name() {
		return "Mul2Op_" + memo;
	}

	@Override
	public int[] inShape() {
		return new int[] { -1, -1 };
	}

	@Override
	public int[] outShape() {
		return new int[] { w.shape()[0], d.shape()[1] };
	}

	@Override
	protected MatrixDataEdge eval0(MatrixDataEdge data) {
		RealMatrix r = w.asMat(0).multiply(d.asMat(0));
		return new MatrixDataEdge(name(), r);
	}

	@Override
	public MatrixDataEdge eval(MatrixDataEdge data, String rtMsg) {
		MatrixDataEdge ret = null;
		try {
			ret = eval0(data);
		} catch (Exception e) {
			throw e;
		} finally {
			trace(rtMsg, w, d, ret, null, 1);
		}
		return ret;
	}

}
