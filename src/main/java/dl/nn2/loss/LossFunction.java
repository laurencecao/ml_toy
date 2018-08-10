package dl.nn2.loss;

import org.apache.commons.math3.linear.RealMatrix;

public interface LossFunction {

	double eval(RealMatrix target, RealMatrix y);

	RealMatrix dif(RealMatrix target, RealMatrix y);

	String name();

}
