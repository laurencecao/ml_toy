package dl.nn2.activation;

import org.apache.commons.math3.linear.RealMatrix;

public interface GateFunction {

	RealMatrix forward(RealMatrix m);

	RealMatrix backward(RealMatrix m);

	String name();

}
