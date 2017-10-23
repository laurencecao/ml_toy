package dl.numeric;

import org.apache.commons.math3.special.Beta;
import org.apache.commons.math3.special.Gamma;
import org.apache.commons.math3.util.FastMath;

public class BetaDistPlayer {

	public static void main(String[] args) {
		double a = 1, b = 1;

		double x = beta(a, b);
		double y = FastMath.exp(Beta.logBeta(a, b));
		System.out.println(x + " | " + y);

		a = 0.1;
		b = 0.5;
		x = beta(a, b);
		y = FastMath.exp(Beta.logBeta(a, b));
		System.out.println(x + " | " + y);

		a = 5;
		b = 0.2;
		x = beta(a, b);
		y = FastMath.exp(Beta.logBeta(a, b));
		System.out.println(x + " | " + y);

		a = 5;
		b = 8;
		x = beta(a, b);
		y = FastMath.exp(Beta.logBeta(a, b));
		System.out.println(x + " | " + y);

	}

	static double beta(double a, double b) {
		// B(a, b) = gamma(a) * gamma(b) / gamma(a+b)
		return Gamma.gamma(a) * Gamma.gamma(b) / Gamma.gamma(a + b);
	}

}
