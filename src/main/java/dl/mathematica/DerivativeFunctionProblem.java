package dl.mathematica;

import java.util.function.Function;

import org.apache.commons.math3.util.FastMath;

public class DerivativeFunctionProblem {

	public static void main(String[] args) {
		Function<Double, Double> sin = FastMath::sin;
		double ret = derivative(sin, 0);
		System.out.println("sin'(0) ==> " + ret);
	}

	static double derivative(Function<Double, Double> fn, double x0) {
		double delta = 1.0e-15; // or similar
		double x1 = x0 - delta;
		double x2 = x0 + delta;
		double y1 = fn.apply(x1);
		double y2 = fn.apply(x2);
		return (y2 - y1) / (x2 - x1);
	}

}
