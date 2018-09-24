package dl.nn2.graph;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class CrossProductOp extends TracedComputation {

	protected Computation weight;
	protected MatrixDataEdge w;
	protected int[] shape = { -1, -1 };
	protected String memo;

	public CrossProductOp(MatrixDataEdge w, String memo) {
		this.w = w;
		this.memo = memo;
	}

	public CrossProductOp(Computation comp, String memo) {
		this.weight = comp;
		this.memo = memo;
	}

	@Override
	public String name() {
		return CrossProductOp.class.getName();
	}

	@Override
	public int[] inShape() {
		return shape;
	}

	@Override
	public int[] outShape() {
		return shape;
	}

	@Override
	protected MatrixDataEdge eval0(MatrixDataEdge data) {
		MatrixDataEdge wt = null;
		if (this.w != null) {
			wt = w;
		} else {
			wt = this.weight.eval(null);
		}
		MatrixDataEdge ret = new MatrixDataEdge("CrossProduct");
		for (int i = 0; i < data.asMatList().size(); i++) {
			RealMatrix m = data.asMatList().get(i);
			RealMatrix ww = wt.asMatList().get(i);
			RealMatrix r = MatrixUtils.createRealMatrix(m.getRowDimension(), m.getColumnDimension());
			for (int x = 0; x < m.getRowDimension(); x++) {
				RealVector v = m.getRowVector(x);
				RealVector vv = ww.getRowVector(x);
				r.setRowVector(x, vv.ebeMultiply(v));
			}
			ret.addToMatList(r);
		}
		return ret;
	}

}
