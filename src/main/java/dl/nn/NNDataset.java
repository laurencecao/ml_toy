package dl.nn;

import java.util.Arrays;

import org.apache.commons.lang3.StringUtils;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealVector;

public class NNDataset {

	final static String XOR = "XOR";

	final static double[][] xor_inputs = new double[][] {

			{ -1, -1, -1 }, { -1, 1, 1 }, { 1, -1, 1 }, { 1, 1, -1 }

	};

	public static RealVector[] getData(String name) {
		RealVector[] ret = null;
		if (StringUtils.equalsIgnoreCase(XOR, name)) {
			ret = new RealVector[xor_inputs.length];
			for (int i = 0; i < xor_inputs.length; i++) {
				ret[i] = MatrixUtils.createRealVector(Arrays.copyOfRange(xor_inputs[i], 0, 2));
			}
		}
		return ret;
	}

	public static RealVector[] getLabel(String name) {
		RealVector[] ret = null;
		if (StringUtils.equalsIgnoreCase(XOR, name)) {
			ret = new RealVector[xor_inputs.length];
			for (int i = 0; i < xor_inputs.length; i++) {
				ret[i] = MatrixUtils.createRealVector(new double[] { xor_inputs[i][2] });
			}
		}
		return ret;
	}

}
