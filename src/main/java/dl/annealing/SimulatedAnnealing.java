package dl.annealing;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.commons.math3.util.FastMath;

import dataset.NNDataset;

public class SimulatedAnnealing {

	final static double ENERGY_TARGET = 100d; // when temperature = 1
	final static double TEMPERATURE = 100d;
	final static double coolRate = 1.0d / TEMPERATURE; // FastMath.pow(TEMPERATURE,
														// 2);
	final static int debug = 10000000;

	// [-5, -3] => [0.006, 0.05]
	final static double very_small = 0.001d;
	final static double very_big = 0.01d;

	static class Paremeter {

		double p0;
		double p1;
		double element;
		double denominator;

		double base0;
		double base1;
		double m;
		double density;

		Paremeter(double p0, double p1, double mean) {
			this.p0 = p0 < p1 ? p0 : p1;
			this.p1 = p0 < p1 ? p1 : p0;
			this.m = mean;
		}

		double getEnergy(double currentCost, double nextCost, double tp) {
			base1 = FastMath.log(p0) * tp * -1;
			base0 = FastMath.log(p1) * tp * -1;
			int sign = nextCost >= currentCost ? 1 : -1;
			double inc = (nextCost - currentCost) / m;
			inc = base1 - FastMath.min(FastMath.abs(inc), 1) * density;
			return inc * sign;
		}

	}

	static double energy(State current, State next, double tp, Paremeter param) {
		// return (next.cost - current.cost) / current.cost;
		return param.getEnergy(current.cost, next.cost, tp);
	}

	static boolean acceptOrDecline(double dE, double tp) {
		if (dE <= 0) {
			return true;
		}
		double p = FastMath.exp(-dE / tp);
		// System.out.println("accept prob: " + p + ", " + dE);
		return ThreadLocalRandom.current().nextDouble() < p;
	}

	static double cool(int timeIdx, double dE, double tp) {
		// 1. timeIdx: little and little impact on temperature
		double a = 1.0d / timeIdx;

		// 2. dE with it's mean:
		// double b = FastMath.abs(dE) / mean.getMean();

		// 3. now temperature: faster at higher temperature
		double c = FastMath.log(2, tp);

		return coolRate * FastMath.exp(-1 * (a + c));
	}

	/**
	 * <pre>
	 * total cost = 708.0 ==> 
	 *     22[0.0] --> 16[27.0] --> 12[29.0] --> 13[35.0] --> 14[10.0] --> 15[27.0] 
	 * --> 17[26.0] --> 18[26.0] --> 19[22.0] --> 20[30.0] --> 27[31.0] --> 28[12.0] 
	 * --> 29[20.0] --> 30[8.0] --> 31[12.0] --> 32[11.0] --> 33[21.0] --> 34[9.0] 
	 * --> 35[18.0] --> 36[17.0] --> 37[12.0] --> 38[9.0] --> 39[6.0] --> 40[25.0] 
	 * --> 41[6.0] --> 0[5.0] --> 1[8.0] --> 2[45.0] --> 3[9.0] --> 4[15.0] --> 5[17.0] 
	 * --> 6[6.0] --> 7[10.0] --> 8[5.0] --> 9[20.0] --> 11[26.0] --> 10[11.0] --> 23[23.0] 
	 * --> 24[8.0] --> 25[11.0] --> 26[3.0] --> 21[32.0] --> 22[5.0]
	 * </pre>
	 * 
	 * @param args
	 */
	public static void main(String[] args) {
		RealMatrix weight = NNDataset.getWeights(NNDataset.TSP);
		State.weight = weight;
		State best = training(weight);
		printTravelPath(best.path, weight);
	}

	static State training(RealMatrix weight) {
		// initialization
		SummaryStatistics mean = new SummaryStatistics();
		double[][] w = weight.getData();
		for (int i = 0; i < w.length; i++) {
			for (int j = 0; j < w[0].length; j++) {
				mean.addValue(w[i][j]);
			}
		}
		Paremeter param = new Paremeter(very_small, very_big, mean.getMean());
		State best = new State(null);
		best.costIt();
		best.temperature = TEMPERATURE;

		State current = best.copy();
		State next;

		int timeIdx = 0;
		double tp = TEMPERATURE;
		// while not freeze
		while (tp > 1) { // freeze to 1
			// mutation
			next = current.copy();
			next.mutate();
			next.costIt();

			// calculate energy
			double dE = energy(current, next, tp, param);

			// accept or decline
			boolean accept = acceptOrDecline(dE, tp);

			if (accept) { // accept with no condition
				current = next;
				if (current.cost < best.cost) {
					best = current.copy();
				}
				tp = tp - cool(timeIdx, dE, tp);
				// System.out.println("temperature: " + tp);
			}

			timeIdx++;

			if (timeIdx % debug == 0) {
				System.out.println("temperature=" + tp + " at[" + timeIdx + "], best cost is: " + best.cost);
			}

		}
		System.out.println("Total mutation times: " + timeIdx);

		return best;

	}

	static void printTravelPath(Integer[] path, RealMatrix cost) {
		double total_cost = 0d;
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < path.length; i++) {
			int now = path[i];
			double c = 0d;
			if (i > 0) {
				int from = path[i - 1];
				int to = now;
				c = cost.getEntry(from, to);
			}
			total_cost += c;
			sb.append(now).append("[").append(c).append("]").append(" --> ");
		}
		double c = cost.getEntry(path[path.length - 1], path[0]);
		sb.append(path[0]).append("[").append(c).append("]");
		total_cost += c;
		sb.insert(0, " ==> ").insert(0, total_cost).insert(0, "total cost = ");
		System.out.println(sb.toString());
	}

}

/**
 * performance is discarded for comprehensibility
 * 
 * @author caowenjiong
 *
 */
class State {

	final static Random rng = new Random();
	static RealMatrix weight;

	Integer[] path;
	double cost;
	double temperature;

	private State() {
		// for copy
	}

	State(Integer[] path) {
		this.path = path;
		this.cost = 0d;
		if (path == null) {
			fullBuild();
		}
	}

	void costIt() {
		double ret = 0d;
		for (int i = 1; i < path.length; i++) {
			int from = this.path[i - 1];
			int to = this.path[i];
			ret += weight.getEntry(from, to);
		}
		int from = this.path[this.path.length - 1];
		int to = this.path[0];
		ret += weight.getEntry(from, to);

		this.cost = ret;
	}

	void mutate() {
		int r = this.path.length;

		int p1 = rng.nextInt(r);
		int p2 = rng.nextInt(r);

		// swap path[p1] && path[p2]
		int c = this.path[p1];
		this.path[p1] = this.path[p2];
		this.path[p2] = c;
	}

	void fullBuild() {
		int r = weight.getRowDimension();
		List<Integer> lst = new ArrayList<Integer>();
		for (int i = 0; i < r; i++) {
			lst.add(i);
		}
		// important things repeat 3 times
		Collections.shuffle(lst);
		Collections.shuffle(lst);
		Collections.shuffle(lst);
		this.path = lst.toArray(new Integer[lst.size()]);
	}

	double getCost() {
		return cost;
	}

	State copy() {
		State ret = new State();
		ret.path = Arrays.copyOfRange(this.path, 0, this.path.length);
		ret.cost = this.cost;
		return ret;
	}

}
