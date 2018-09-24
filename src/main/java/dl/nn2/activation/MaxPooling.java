package dl.nn2.activation;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.util.FastMath;

public class MaxPooling implements GateFunction {

	public final static char WEIGHT = 'w'; // only weights

	protected int poolSize;
	protected char type;

	public MaxPooling(int pool) {
		this(pool, 'w');
	}

	public MaxPooling(int pool, char type) {
		this.poolSize = pool;
		this.type = type;
	}

	public MaxPooling copy(char type) {
		return new MaxPooling(this.poolSize, type);
	}

	@Override
	public RealMatrix forward(RealMatrix m) {
		int r = Double.valueOf(FastMath.ceil(1.0d * m.getRowDimension() / poolSize)).intValue();
		int c = Double.valueOf(FastMath.ceil(1.0d * m.getColumnDimension() / poolSize)).intValue();
		RealMatrix ret = MatrixUtils.createRealMatrix(r, c);
		int iii = 0;
		int jjj = 0;
		for (int i = 0; i < m.getColumnDimension(); i += poolSize) {
			for (int j = 0; j < m.getRowDimension(); j += poolSize) {
				int jj = j + poolSize - 1;
				jj = jj < m.getRowDimension() ? jj : m.getRowDimension() - 1;
				int ii = i + poolSize - 1;
				ii = ii < m.getColumnDimension() ? ii : m.getColumnDimension() - 1;
				RealMatrix m0 = m.getSubMatrix(j, jj, i, ii);
				double mm = Double.MIN_VALUE;
				for (int k = 0; k < m0.getRowDimension(); k++) {
					mm = FastMath.max(m0.getRowVector(k).getMaxValue(), mm);
				}
				ret.setEntry(iii, jjj, mm);
				iii++;
			}
			jjj++;
			iii = 0;
		}
		return ret;
	}

	@Override
	public RealMatrix backward(RealMatrix m) {
		RealMatrix ret = MatrixUtils.createRealMatrix(m.getRowDimension(), m.getColumnDimension());
		for (int i = 0; i < m.getColumnDimension(); i += poolSize) {
			for (int j = 0; j < m.getRowDimension(); j += poolSize) {
				int jj = j + poolSize - 1;
				jj = jj < m.getRowDimension() ? jj : m.getRowDimension() - 1;
				int ii = i + poolSize - 1;
				ii = ii < m.getColumnDimension() ? ii : m.getColumnDimension() - 1;
				double mm = Double.MIN_VALUE;
				int[] pos = null;
				for (int x = j; x <= jj; x++) {
					for (int y = i; y <= ii; y++) {
						if (m.getEntry(x, y) > mm) {
							pos = new int[] { x, y };
							mm = m.getEntry(x, y);
						}
					}
				}
				switch (type) {
				case 'w':
					ret.setEntry(pos[0], pos[1], 1);
				default:
					break;
				}
			}
		}
		return ret;
	}

	@Override
	public String name() {
		return MaxPooling.class.getName();
	}

}
