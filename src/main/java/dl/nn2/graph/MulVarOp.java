package dl.nn2.graph;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class MulVarOp extends TracedComputation {

	final protected Computation comp;
	final protected boolean transpose;
	final protected boolean preMul;
	final protected boolean byEle;
	final protected String memo;
	protected MatrixDataEdge w;

	public MulVarOp(Computation comp, String memo) {
		this(comp, false, false, true, memo);
	}

	public MulVarOp(Computation comp, boolean transpose, boolean preMul, boolean byEle, String memo) {
		this.comp = comp;
		this.preMul = preMul;
		this.memo = memo;
		this.transpose = transpose;
		this.byEle = byEle;
	}

	@Override
	protected MatrixDataEdge eval0(MatrixDataEdge data) {
		this.w = comp.eval(null);
		RealMatrix p = this.transpose ? w.asMat(0).transpose() : w.asMat(0);
		RealMatrix r = null;
		if (byEle) {
			r = MatrixUtils.createRealMatrix(p.getRowDimension(), p.getColumnDimension());
			for (int i = 0; i < p.getColumnDimension(); i++) {
				RealVector v0 = p.getColumnVector(i);
				RealVector v1 = data.asMat(0).getColumnVector(i);
				r.setColumnVector(i, v0.ebeMultiply(v1));
			}
		} else {
			if (!preMul) {
				r = p.multiply(data.asMat(0));
			} else {
				r = data.asMat(0).multiply(p);
			}
		}
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
			if (!preMul) {
				trace(rtMsg, wt, data, ret, null, 1);
			} else {
				trace(rtMsg, data, wt, ret, null, 0);
			}
		}
		return ret;
	}

	@Override
	public String name() {
		return (transpose ? "{T}" : " ") + (preMul ? "{preMul}" : "");
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
