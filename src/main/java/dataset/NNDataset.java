package dataset;

import java.util.Arrays;

import org.apache.commons.lang3.StringUtils;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import dl.em.ThreeCoinProblemData;
import dl.hmm.FourCoinProblemData;

public class NNDataset {

	public final static String XOR = "XOR";
	public final static String CIRCLE = "CIRCLE";
	public final static String MOVIELENS = "MOVIELENS";
	public final static String DIGITAL = "DIGITAL";
	public final static String HOME = "HOME";
	public final static String TSP = "TSP";
	public final static String WHOLESALE = "WHOLESALE";
	public final static String THREECOINS = "THREECOINS";
	public final static String FOURCOINS = "FOURCOINS";
	public final static String WINGNUT = "WINGNUT";

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
		if (StringUtils.equalsIgnoreCase(CIRCLE, name)) {
			int sz = CircleRuleGame.count();
			ret = new RealVector[sz];
			for (int i = 0; i < sz; i++) {
				ret[i] = MatrixUtils.createRealVector(CircleRuleGame.getData(i));
			}
		}
		if (StringUtils.equalsIgnoreCase(MOVIELENS, name)) {
			int sz = MovieLens.rates.getRowDimension();
			ret = new RealVector[sz];
			for (int i = 0; i < sz; i++) {
				ret[i] = MovieLens.rates.getRowVector(i);
			}
		}
		if (StringUtils.equalsIgnoreCase(DIGITAL, name)) {
			int sz = Digital.data.length;
			ret = new RealVector[sz];
			for (int i = 0; i < sz; i++) {
				ret[i] = MatrixUtils.createRealVector(Digital.data[i]);
			}
		}
		if (StringUtils.equalsIgnoreCase(HOME, name)) {
			ret = HomeLocation.getData();
		}
		if (StringUtils.equalsIgnoreCase(WHOLESALE, name)) {
			ret = WholeSale.getData();
		}
		if (StringUtils.equalsIgnoreCase(THREECOINS, name)) {
			ret = ThreeCoinProblemData.load(ThreeCoinProblemData.dataPath);
		}
		if (StringUtils.equalsIgnoreCase(FOURCOINS, name)) {
			ret = FourCoinProblemData.load(FourCoinProblemData.dataPath);
		}
		if (StringUtils.equalsIgnoreCase(WINGNUT, name)) {
			ret = WingNut.getData();
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
		if (StringUtils.equalsIgnoreCase(CIRCLE, name)) {
			int sz = CircleRuleGame.count();
			ret = new RealVector[sz];
			for (int i = 0; i < sz; i++) {
				ret[i] = MatrixUtils.createRealVector(new double[] { CircleRuleGame.getLabel(i) });
				ret[i].mapSubtractToSelf(4).mapDivideToSelf(4);
			}
		}
		if (StringUtils.equalsIgnoreCase(HOME, name)) {
			ret = HomeLocation.getLabel();
		}
		return ret;
	}

	public static String[] getHeader(String name) {
		String[] ret = null;
		if (StringUtils.equalsIgnoreCase(HOME, name)) {
			ret = HomeLocation.getHeader();
		}
		if (StringUtils.equalsIgnoreCase(WHOLESALE, name)) {
			ret = WholeSale.getHeader();
		}
		return ret;
	}

	public static RealMatrix getWeights(String name) {
		RealMatrix ret = null;
		if (StringUtils.equalsIgnoreCase(TSP, name)) {
			ret = TSPCitys.cityCost;
		}
		return ret;
	}

}
