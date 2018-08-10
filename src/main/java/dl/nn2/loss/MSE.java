package dl.nn2.loss;

import org.apache.commons.math3.linear.RealMatrix;

public class MSE implements LossFunction {

	@Override
	public String name() {
		return MSE.class.getSimpleName();
	}

	@Override
	public double eval(RealMatrix target, RealMatrix y) {
		double ret = 0d;
		for (int i = 0; i < target.getColumnDimension(); i++) {
			double t = target.getEntry(0, i);
			double _y = y.getEntry(0, i);
			ret += 1.0d * (t - _y) * (t - _y) / 2;
		}
		return ret / target.getColumnDimension();
	}

	@Override
	public RealMatrix dif(RealMatrix target, RealMatrix y) {
		return y.subtract(target);
	}

}
