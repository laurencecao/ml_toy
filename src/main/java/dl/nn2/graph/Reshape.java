package dl.nn2.graph;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

public class Reshape extends TracedComputation {

	protected int[] outShape;

	public Reshape(int[] out) {
		this.outShape = out;
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
		RealMatrix m = data.asMat(0);
		int c = m.getColumnDimension();
		int r = m.getRowDimension();
		int R = outShape[0];
		int C = outShape[1];
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
