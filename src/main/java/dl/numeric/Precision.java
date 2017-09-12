package dl.numeric;

import java.math.BigDecimal;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.ThreadLocalRandom;

import org.apache.commons.math3.util.FastMath;

public class Precision {

	public static void main(String[] args) {
		doublePrecision();

		Double[][] r = double_array_multiple(10);
		System.out.println(Arrays.toString(r[0]));
		System.out.println(Arrays.toString(r[1]));
	}

	static public void doublePrecision() {
		double total = 0;
		total += 5.6;
		total += 5.8;
		System.out.println("5.6d + 5.8d => " + total);
		float t = 0;
		t += 5.6;
		t += 5.8;
		System.out.println("5.6f + 5.8f => " + t);
	}

	static Double[][] double_array_multiple(int size) {
		ThreadLocalRandom rnd = ThreadLocalRandom.current();
		double last = 0.1d;
		Double[] ele = new Double[size];
		double r0 = 1d, r1 = 0d, r2 = 1d, r3 = 1d;
		for (int i = 0; i < size; i++) {
			ele[i] = rnd.nextDouble(last);
			last *= 0.1d;
		}

		for (int i = 0; i < ele.length; i++) {
			r0 *= ele[i];
		}

		for (int i = 0; i < ele.length; i++) {
			r1 += FastMath.log(ele[i]);
		}
		r1 = FastMath.exp(r1);

		double deno = 0d;
		for (int i = 0; i < ele.length; i++) {
			double scaled = i * 2;
			deno += scaled;
			double s = FastMath.pow(10, scaled);
			r2 *= ele[i] * s;
		}
		r2 /= FastMath.pow(10, deno);

		BigDecimal b = new BigDecimal(1);
		for (int i = 0; i < ele.length; i++) {
			b = b.multiply(new BigDecimal(ele[i]));
		}
		r3 = b.doubleValue();
		return new Double[][] { ele, { r0, r1, r2, r3 } };
	}

	static public double sumAtLogSpace(Double[] arr) {
		List<Double> lst = Arrays.asList(arr);
		Double maximum = Collections.max(lst);
		Optional<Double> a = lst.stream().map(x -> FastMath.exp(x - maximum)).reduce(Double::sum);
		return FastMath.log(a.get()) + maximum;
	}

	static public Double elnproduct(Double x, Double y) {
		Double xx = FastMath.log(x);
		Double yy = FastMath.log(y);
		if (xx == null || yy == null) {
			return null;
		}
		return xx + yy;
	}

}
