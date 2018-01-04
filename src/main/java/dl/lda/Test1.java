package dl.lda;

import java.util.List;

import org.apache.commons.math3.distribution.EnumeratedDistribution;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import umontreal.ssj.randvarmulti.DirichletGen;
import utils.ImageUtils;

public class Test1 {

	public static void main(String[] args) {
		double[][] p = new double[][] {
				{ 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
				{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
				{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255 } };
		RealMatrix m = MatrixUtils.createRealMatrix(p).transpose();
		String s = t1(m);
		System.out.println(s);
		s = t2(3, 20);
		System.out.println(s);
	}

	static String t1(RealMatrix p) {
		List<EnumeratedDistribution<Integer>> diceB = LDAGenerator.getDiceB(p);
		double[][] ret = new double[diceB.size()][];
		for (int i = 0; i < diceB.size(); i++) {
			EnumeratedDistribution<Integer> item = diceB.get(i);
			ret[i] = new double[item.getPmf().size()];
			for (int j = 0; j < item.getPmf().size(); j++) {
				ret[i][j] = item.getPmf().get(j).getValue();
			}
		}
		return ImageUtils.drawMultinomialLines(ret);
	}

	static String t2(int K, int T) {
		DirichletGen diceA = LDAGenerator.getDiceA(K);
		double[][] ret = new double[T][K];
		for (int i = 0; i < T; i++) {
			diceA.nextPoint(ret[i]);
		}
		return ImageUtils.drawMultinomialLines(ret);
	}

}
