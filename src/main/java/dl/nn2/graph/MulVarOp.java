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
		MatrixDataEdge ret = new MatrixDataEdge(type() + "_" + name() + "_" + id());
		for (int i = 0; i < w.asMatList().size(); i++) {
			RealMatrix m = w.asMatList().get(i);
			RealMatrix p = this.transpose ? m.transpose() : m;
			RealMatrix r = p.multiply(data.asMatList().get(i));
			ret.addToMatList(r);
		}
		return ret;
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
