package dl.annealing;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.util.FastMath;

import dataset.NNDataset;

public class SimulatedAnnealing {

	final static double TEMPERATURE = 100d;
	final static double coolRate = 1.0d * 0.1d / TEMPERATURE;
	final static int debug = 10000000;

	static double energy(State current, State next) {
		return (next.cost - current.cost);
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

		// 2. now temperature: faster at higher temperature
		double c = FastMath.log(2, tp);

		return coolRate * FastMath.exp(-1 * (a + c));
	}

	/**
	 * <pre>
	 * total cost = 704.0 ==> 
	 *     12[0.0] --> 16[29.0] --> 21[30.0] --> 22[5.0] --> 11[36.0] --> 10[11.0] --> 23[23.0] 
	 * --> 26[9.0] --> 25[3.0] --> 24[11.0] --> 9[14.0] --> 8[20.0] --> 7[5.0] --> 6[10.0] 
	 * --> 5[6.0] --> 4[17.0] --> 3[15.0] --> 2[9.0] --> 1[45.0] --> 0[8.0] --> 41[5.0] 
	 * --> 40[6.0] --> 39[25.0] --> 38[6.0] --> 37[9.0] --> 36[12.0] --> 35[17.0] --> 34[18.0] 
	 * --> 33[9.0] --> 32[21.0] --> 31[11.0] --> 30[12.0] --> 29[8.0] --> 28[20.0] --> 27[12.0] 
	 * --> 20[31.0] --> 19[30.0] --> 18[22.0] --> 17[26.0] --> 15[26.0] --> 14[27.0] --> 13[10.0] --> 12[35.0]
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
			double dE = energy(current, next);

			// accept or decline
			boolean accept = acceptOrDecline(dE, tp);

			if (accept) { // accept with no condition
				current = next;
				if (current.cost < best.cost) {
					best = current.copy();
				}
				tp = tp - cool(timeIdx, dE, tp);
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
