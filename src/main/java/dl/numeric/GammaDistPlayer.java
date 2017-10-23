package dl.numeric;

import org.apache.commons.math3.special.Gamma;
import org.apache.commons.math3.util.FastMath;

public class GammaDistPlayer {

	public static void main(String[] args) {
		// Stirling's formula

		double a;
		double b;
		double c;
		a = gamma(100);
		b = Gamma.gamma(100);
		c = gamma0(100);

		System.out.println(a + " | " + b + " | " + c);
	}

	static double gamma(int x) {
		// n! = sqrt(2 * PI * n) * (n/e)^n
		double n = x - 1;
		double a = FastMath.sqrt(2 * FastMath.PI * n);
		double b = FastMath.pow((n / FastMath.E), n);
		double ret = a * b;
		return ret;
	}

	static double gamma0(int x) {
		int n = x - 1;
		double ret = 1d;
		for (int i = 1; i <= n; i++) {
			ret *= i;
		}
		return ret;
	}

}
