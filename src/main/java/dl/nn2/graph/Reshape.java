package dl.nn2.graph;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

public class Reshape extends TracedComputation {

	protected int[] outShape;
	protected boolean dummy = false;
	protected boolean reverse = false;

	public Reshape() {
		// dummy reshape
		this(new int[] { -1, -1 });
	}

	public Reshape(int[] out) {
		this(out, false);
	}

	public Reshape(int[] out, boolean reverse) {
		if ((out == null) || (out[0] == out[1] && out[0] == -1)) {
			this.dummy = true;
		}
		this.outShape = out;
		this.reverse = reverse;
	}

	@Override
	public String name() {
		return Reshape.class.getName();
	}

	@Override
	public int[] inShape() {
		return new int[] { -1, -1 };
	}

	@Override
	public int[] outShape() {
		return outShape;
	}

	@Override
	protected MatrixDataEdge eval0(MatrixDataEdge data) {
		if (dummy) {
			return new MatrixDataEdge("no_reshape", data);
		}
		MatrixDataEdge ret = new MatrixDataEdge("reshape");
		MatrixDataEdge r;
		if (!reverse) {
			for (RealMatrix mm : data.asMatList()) {
				r = doCalc(data, new int[] { mm.getRowDimension(), mm.getColumnDimension() },
						new int[] { outShape[0], outShape[1] });
				ret.addToMatList(r.asMat(0));
			}
		} else {
			for (RealMatrix mm : data.asMatList()) {
				r = doCalc(data, new int[] { outShape[0], outShape[1] },
						new int[] { mm.getRowDimension(), mm.getColumnDimension() });
				ret.addToMatList(r.asMat(0));
			}
		}
		return ret;
	}

	protected MatrixDataEdge doCalc(MatrixDataEdge data, int[] in, int[] out) {
		RealMatrix m = data.asMat(0);
		int r = in[0];
		int c = in[1];
		int R = out[0];
		int C = out[1];
		if (R < 0) {
			double d = 1.0d * c * r / C;
			R = Double.valueOf(d).intValue();
		}
		if (C < 0) {
			double d = 1.0d * c * r / R;
			C = Double.valueOf(d).intValue();
		}

		if (C * R != c * r) {
			throw new IllegalArgumentException();
		}
		RealMatrix ret = MatrixUtils.createRealMatrix(R, C);
		int x = 0, y = 0;
		for (int i = 0; i < m.getRowDimension(); i++) {
			double[] d = m.getRow(i);
			for (int z = 0; z < d.length; z++) {
				ret.setEntry(x, y, d[z]);
				y++;
				if (y >= C) {
					x++;
					y = 0;
				}
			}
		}
		return new MatrixDataEdge("", ret);
	}

}
