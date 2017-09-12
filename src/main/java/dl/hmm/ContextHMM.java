package dl.hmm;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Optional;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.util.FastMath;

public class ContextHMM {

	final static double LOGZERO = 0.00000001d;

	final int batch_size;
	final int state_size;
	final int symbol_size;

	ContextHMM(int batch, int state, int symbol) {
		this.batch_size = batch;
		this.state_size = state;
		this.symbol_size = symbol;
	}

	RealVector initialStates; // π = state * 1
	RealMatrix transitionProbs; // t = state * state
	RealMatrix emissionProbs; // q = state * symbol

	RealMatrix alpha; // α = state * turn
	RealMatrix beta; // β = state * turn
	RealMatrix gamma; // γ = state * turn
	RealMatrix[] ksi; // ξ = (state * state) [turn]

	double like;

	public ContextHMM copy() {
		ContextHMM ret = new ContextHMM(batch_size, state_size, symbol_size);
		ret.initialStates = initialStates.copy();
		ret.transitionProbs = transitionProbs.copy();
		ret.emissionProbs = emissionProbs.copy();
		ret.alpha = alpha.copy();
		ret.beta = beta.copy();
		ret.gamma = gamma.copy();
		ret.ksi = new RealMatrix[ksi.length];
		for (int i = 0; i < ksi.length; i++) {
			ret.ksi[i] = ksi[i].copy();
		}
		ret.like = like;
		return ret;
	}

	static public double sumAtLogSpace(Double... arr) {
		List<Double> lst = Arrays.asList(arr);
		Double maximum = Collections.max(lst);
		Optional<Double> a = lst.stream().map(x -> FastMath.exp(x - maximum)).reduce(Double::sum);
		return FastMath.log(a.get()) + maximum;
	}

	static public double sumAtLogSpace(double[] arr) {
		return sumAtLogSpace(ArrayUtils.toObject(arr));
	}

	static public Double eexp(Double a) {
		// a == null if LOGZERO
		if (a == null) {
			return 0d;
		}
		return FastMath.exp(a);
	}

	static public Double eln(Double a) {
		// return LOGZERO if <= 0
		if (a == null || a <= 0) {
			return null;
		}
		return FastMath.log(a);
	}

	static public Double elnsum(Double x, Double y) {
		Double xx = eln(x);
		Double yy = eln(y);
		if (xx == null || yy == null) {
			return xx == null ? yy : xx;
		}
		if (xx > yy) {
			return xx + eln(1 + FastMath.exp(yy - xx));
		}
		return yy + eln(1 + FastMath.exp(xx - yy));
	}

	static public Double elnproduct(Double x, Double y) {
		Double xx = eln(x);
		Double yy = eln(y);
		if (xx == null || yy == null) {
			return null;
		}
		return xx + yy;
	}

	static public Double[] eln(double[] a) {
		Double[] input = ArrayUtils.toObject(a);
		Double[] ret = new Double[input.length];
		for (int i = 0; i < input.length; i++) {
			if (input[i] <= 0d) {
				input[i] = LOGZERO;
			}
			ret[i] = eln(input[i]);
		}
		return ret;
	}

	static public Double[] elnproduct(double[] a, double[] b) {
		Double[] ret = new Double[a.length];
		for (int i = 0; i < ret.length; i++) {
			ret[i] = elnproduct(a[i], b[i]);
		}
		return ret;
	}

	static public Double[] eproduct(double[] a, double[] b) {
		Double[] ret = new Double[a.length];
		for (int i = 0; i < ret.length; i++) {
			ret[i] = a[i] + b[i];
		}
		return ret;
	}

}
