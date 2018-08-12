package dl.nn2.graph;

import org.apache.commons.math3.linear.RealMatrix;

public class MulOp extends TracedComputation {

	final protected MatrixDataEdge w;
	final protected boolean transpose;
	final protected boolean preMul;
	final protected String memo;

	public MulOp(MatrixDataEdge w, boolean transpose, boolean preMul, boolean unBiased, String memo) {
		RealMatrix wt = w.asMat(0);
		if (unBiased) {
			wt = wt.getSubMatrix(0, wt.getRowDimension() - 1, 1, wt.getColumnDimension() - 1);
		}
		this.w = new MatrixDataEdge("MulOp", wt);
		this.transpose = transpose;
		this.preMul = preMul;
		this.memo = memo;
	}

	@Override
	protected MatrixDataEdge eval0(MatrixDataEdge data) {
		RealMatrix p = this.transpose ? this.w.asMat(0).transpose() : this.w.asMat(0);
		RealMatrix r = null;
		if (!preMul) {
			r = p.multiply(data.asMat(0));
		} else {
			r = data.asMat(0).multiply(p);
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
