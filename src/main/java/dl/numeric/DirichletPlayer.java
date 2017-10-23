package dl.numeric;

import cc.mallet.types.Dirichlet;
import cc.mallet.types.Multinomial;
import cern.colt.Arrays;
import umontreal.ssj.probdistmulti.DirichletDist;
import umontreal.ssj.randvarmulti.DirichletGen;
import umontreal.ssj.rng.WELL607;
import utils.ImageUtils;

public class DirichletPlayer {

	public static void main(String[] args) {
		int T = 10;
		double[][] data = new double[T][];
		Multinomial[] data0 = new Multinomial[T];
		double[] alpha = new double[] { 1, 1, 1 };
		DirichletDist dist = new DirichletDist(alpha);
		Dirichlet d = new Dirichlet(alpha);
		for (int i = 0; i < T; i++) {
			double[] arr = d.nextDistribution();
			System.out.println(Arrays.toString(arr));
			System.out.println(dist.density(arr));
			String s = ImageUtils.drawMultinomialLines(new double[][] { arr });
			System.out.println(s);
			data[i] = arr;
			data0[i] = new Multinomial(arr);
		}
		double[] o = DirichletDist.getMLE(data, data.length, 3);
		System.out.println("mle  -----> " + Arrays.toString(o));

		DirichletGen d2 = new DirichletGen(new WELL607(), alpha);
		for (int i = 0; i < T; i++) {
			double[] arr = new double[alpha.length];
			d2.nextPoint(arr);
			System.out.println(Arrays.toString(arr));
			System.out.println(dist.density(arr));
			String s = ImageUtils.drawMultinomialLines(new double[][] { arr });
			System.out.println(s);
			data[i] = arr;
			data0[i] = new Multinomial(arr);
		}
		System.out.println(d2.getDimension());
		o = DirichletDist.getMLE(data, data.length, 3);
		System.out.println("mle  -----> " + Arrays.toString(o));

		
	}

}
