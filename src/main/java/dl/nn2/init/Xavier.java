package dl.nn2.init;

import java.util.concurrent.ThreadLocalRandom;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.math.plot.utils.FastMath;

public class Xavier {

	public static double debug = -1d;

	final static double C = 1d;

	public RealMatrix initWeights(int in, int out) {
		if (debug <= 0) {
			return xavier(in, out);
		}
		return debug(in, out);
	}

	RealMatrix xavier(int in, int out) {
		double var = FastMath.sqrt(6d / (in + out));
		RealMatrix ret = MatrixUtils.createRealMatrix(out, in);
		for (int i = 0; i < out; i++) {
			double[] samples = new double[in];
			for (int j = 0; j < in; j++) {
				samples[j] = ThreadLocalRandom.current().nextDouble(-var, var);
			}
			RealVector v = MatrixUtils.createRealVector(samples);
			ret.setRowVector(i, v);
		}
		return ret;
	}

	RealMatrix debug(int in, int out) {
		RealMatrix ret = MatrixUtils.createRealMatrix(out, in);
		return ret.scalarAdd(debug);
	}

}
