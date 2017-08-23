package dl.ga;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.Function;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.distribution.EnumeratedDistribution;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.RealVectorChangingVisitor;
import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.util.Pair;

import com.google.common.collect.EvictingQueue;

import utils.DrawingUtils;

public class GeneticFunctionFit {

	static Function<double[], Integer> chromosome2Gene = x -> {
		Integer ret = 0;
		for (int i = 0; i < x.length; i++) {
			int flag = x[i] == 1d ? 1 : 0;
			ret = ret << 1;
			ret += flag;
		}
		return ret;
	};

	static Function<Integer, Double> costFunction = x -> {
		return -1 * (FastMath.pow(x, 2)) / 10 + 3 * x;
		// so derivation of this: -(x/5) + 3
	};

	final static int epoch = 5;
	final static double epi = 0.0000001;

	public static void main(String[] args) throws IOException {
		int r = 0;
		r = r << 1;
		r = r + 1;
		r = r << 1;
		r = r + 1;
		int chromosome_sz = Double.valueOf(FastMath.log(2, 4096)).intValue();
		int population_count = 1000;
		// ensure odd
		int mating = Double.valueOf(0.1d * population_count / 2 * 2).intValue();
		// expectation 1.5bit every time of mutation
		double mutateProb = 1.5d / chromosome_sz;

		Double BEST_X = 15d; // 0 = -(x/5) + 3
		double BEST = costFunction.apply(BEST_X.intValue());
		EvictingQueue<Double> buf = EvictingQueue.create(epoch);
		RealVector[] pop = generateSeeds(chromosome_sz, population_count);
		int i = 0;
		double error = 10d;
		while (error > epi || i < epoch) {
			pop = evolution(pop, mating, mutateProb);
			double[] best = evaluationBest(pop);
			buf.add(best[1] - BEST);
			error = Math.abs(best[1] - BEST);
			i++;
		}
		double[] err = ArrayUtils.toPrimitive(buf.toArray(new Double[buf.size()]));
		DrawingUtils.drawMSE(err, i, 1, "/tmp/GA_err.png");

		double[] b = evaluationBest(pop);
		System.out.println("The best fit is: " + b[0] + " -> " + b[1]);
	}

	static RealVector[] generateSeeds(int length, int count) {
		// data = column vector
		RNDBit half = new RNDBit(0.5d);
		RealVector[] ret = new RealVector[count];
		for (int i = 0; i < count; i++) {
			ret[i] = MatrixUtils.createRealVector(new double[length]);
			ret[i].walkInOptimizedOrder(half.setRNDBit);
		}
		return ret;
	}

	static RealVector[] evolution(RealVector[] population, int mating, double mutProb) {
		// evaluation first
		double total = 0;
		double[] p = new double[population.length];
		for (int i = 0; i < population.length; i++) {
			Integer gene = chromosome2Gene.apply(population[i].toArray());
			p[i] = costFunction.apply(gene);
			p[i] = p[i] > 0 ? p[i] : 0.0001d; // using little prob
			total += p[i];
		}
		List<Pair<RealVector, Double>> pmf = new ArrayList<Pair<RealVector, Double>>();
		for (int i = 0; i < p.length; i++) {
			p[i] /= total;
			pmf.add(new Pair<RealVector, Double>(population[i], p[i]));
		}

		EnumeratedDistribution<RealVector> elements = new EnumeratedDistribution<RealVector>(pmf);

		List<RealVector> ret = new ArrayList<RealVector>();
		// selection
		for (int i = 0; i < mating; i++) {
			RealVector father = elements.sample();
			RealVector mother = elements.sample();
			/**
			 * enable this you will see 0.1 residual here
			 */
			// while (father == mother) {
			// mother = elements.sample();
			// }

			// cross over
			int cutoff = ThreadLocalRandom.current().nextInt(0, population[0].getDimension());
			RealVector ch1 = null;
			RealVector ch2 = null;
			if (cutoff == 0) {
				ch1 = father;
				ch2 = mother;
			} else {
				ch1 = father.getSubVector(0, cutoff);
				ch1 = ch1.append(mother.getSubVector(cutoff, population[0].getDimension() - cutoff));
				ch2 = mother.getSubVector(0, cutoff);
				ch2 = ch2.append(father.getSubVector(cutoff, population[0].getDimension() - cutoff));
			}

			ret.add(ch1);
			ret.add(ch2);
		}

		// mutation
		MutateBit m = new MutateBit(mutProb);
		for (int i = 0; i < ret.size(); i++) {
			RealVector v = ret.get(i);
			v.walkInOptimizedOrder(m.setMutateBit);
		}

		// complete
		return ret.toArray(new RealVector[ret.size()]);
	}

	static double[] evaluationBest(RealVector[] population) {
		double ret = 0d;
		double x = 0d;
		for (int i = 0; i < population.length; i++) {
			Integer gene = chromosome2Gene.apply(population[i].toArray());
			double chrom = costFunction.apply(gene);
			ret = FastMath.max(chrom, ret);
			x = chrom == ret ? gene : x;
		}
		return new double[] { x, ret };
	}
}

class RNDBit {

	final double prob;

	RNDBit(double p) {
		this.prob = p;
	}

	RealVectorChangingVisitor setRNDBit = new RealVectorChangingVisitor() {
		@Override
		public void start(int dimension, int start, int end) {
		}

		@Override
		public double visit(int index, double value) {
			return FastMath.random() < prob ? 1 : 0;
		}

		@Override
		public double end() {
			return 0;
		}
	};

}

class MutateBit {

	final double prob;

	MutateBit(double p) {
		this.prob = p;
	}

	RealVectorChangingVisitor setMutateBit = new RealVectorChangingVisitor() {
		@Override
		public void start(int dimension, int start, int end) {
		}

		@Override
		public double visit(int index, double value) {
			double nv = value == 1 ? 0 : 0;
			return FastMath.random() < prob ? nv : value;
		}

		@Override
		public double end() {
			return 0;
		}
	};

}