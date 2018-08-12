package dl.nn2.graph;

import org.apache.commons.math3.linear.RealMatrix;

public class MulVarOp extends TracedComputation {

	final protected Computation comp;
	final protected boolean transpose;
	final protected String memo;
	protected MatrixDataEdge w;

	public MulVarOp(Computation comp, String memo) {
		this(comp, false, memo);
	}

	public MulVarOp(Computation comp, boolean transpose, String memo) {
		this.comp = comp;
		this.memo = memo;
		this.transpose = transpose;
	}

	@Override
	protected MatrixDataEdge eval0(MatrixDataEdge data) {
		this.w = comp.eval(null);
		RealMatrix p = this.transpose ? w.asMat(0).transpose() : w.asMat(0);
		RealMatrix r = p.multiply(data.asMat(0));
		return new MatrixDataEdge(type() + "_" + name() + "_" + id(), r);
	}

	public MatrixDataEdge eval(MatrixDataEdge data) {
		return eval(data, "");
	}

	@Override
	public MatrixDataEdge eval(MatrixDataEdge data, String rtMsg) {
		MatrixDataEdge ret = null;
		try {
			ret = eval0(data);
		} catch (Exception e) {
			throw e;
		} finally {
			MatrixDataEdge wt = null;
			if (transpose) {
				wt = new MatrixDataEdge("transposed weight", w.asMat(0).transpose());
			} else {
				wt = w;
			}
			trace(rtMsg, wt, data, ret, null, 1);
		}
		return ret;
	}

	@Override
	public String name() {
		return (transpose ? "{T}" : " ");
	}

	@Override
	public String type() {
		return super.type() + name() + " #" + memo + "#";
	}

	@Override
	public int[] inShape() {
		RealMatrix p = this.transpose ? this.w.asMat(0).transpose() : this.w.asMat(0);
		return new int[] { p.getColumnDimension(), -1 };
	}

	@Override
	public int[] outShape() {
		RealMatrix p = this.transpose ? this.w.asMat(0).transpose() : this.w.asMat(0);
		return new int[] { p.getRowDimension(), -1 };
	}

}
