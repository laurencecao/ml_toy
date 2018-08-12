package dl.nn2.activation;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class Utils {

	public static RealMatrix gateCorr(RealMatrix data) {
		int sz = data.getRowDimension();
		RealMatrix ret = MatrixUtils.createRealMatrix(sz, sz);
		for (int i = 0; i < data.getColumnDimension(); i++) {
			RealVector v = data.getColumnVector(i);
			for (int j = 0; j < v.getDimension(); j++) {
				ret.setEntry(j, j, v.getEntry(j));
			}
		}
		ret = ret.scalarMultiply(1d / sz);
		return ret;
	}

}
