package dl.nn2.graph;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

public class Upsampling extends TracedComputation {

	protected int kernel;

	public Upsampling(int kernel) {
		this.kernel = kernel;
	}

	@Override
	public String name() {
		return Upsampling.class.getName();
	}

	@Override
	public int[] inShape() {
		return new int[] { -1, -1 };
	}

	@Override
	public int[] outShape() {
		return new int[] { -1, -1 };
	}

	@Override
	protected MatrixDataEdge eval0(MatrixDataEdge data) {
		MatrixDataEdge ret = new MatrixDataEdge("upsampling");
		for (RealMatrix m : data.asMatList()) {
			RealMatrix r = MatrixUtils.createRealMatrix(m.getRowDimension() * kernel, m.getColumnDimension() * kernel);
			for (int i = 0; i < m.getRowDimension(); i++) {
				int x = i * kernel;
				for (int j = 0; j < m.getColumnDimension(); j++) {
					int y = j * kernel;
					double v = m.getEntry(i, j);
					for (int k = 0; k < kernel; k++) {
						for (int kk = 0; kk < kernel; kk++) {
							r.setEntry(x + k, y + kk, v);
						}
					}
				}
			}
			ret.addToMatList(r);
		}
		return ret;
	}

}
