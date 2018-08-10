package dl.nn2.activation;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.util.FastMath;

public class Softmax implements GateFunction {

	@Override
	public RealMatrix forward(RealMatrix m0) {
		RealMatrix ret = m0.copy();
		for (int pos = 0; pos < ret.getColumnDimension(); pos++) {
			RealVector v = ret.getColumnVector(pos);
			double m = v.getMaxValue();
			double deno = 0d;
			double ele = 0d;
			for (int i = 0; i < v.getDimension(); i++) {
				ele = FastMath.exp(v.getEntry(i) - m);
				v.setEntry(i, ele);
				deno += ele;
			}
			ret.setColumnVector(pos, v.mapDivide(deno));
		}
		return ret;
	}

	@Override
	public RealMatrix backward(RealMatrix m) {
		throw new UnsupportedOperationException();
	}

	@Override
	public String name() {
		return Softmax.class.getSimpleName();
	}

}
