package dl.nn2.activation;

import org.apache.commons.math3.linear.RealMatrix;

public class NoOp implements GateFunction {

	@Override
	public RealMatrix forward(RealMatrix m) {
		return m;
	}

	@Override
	public RealMatrix backward(RealMatrix m) {
		RealMatrix ret = m.copy();
		return ret.scalarMultiply(0).scalarAdd(1);
	}

	@Override
	public String name() {
		return NoOp.class.getName();
	}

}
