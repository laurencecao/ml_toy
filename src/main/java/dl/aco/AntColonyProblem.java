package dl.aco;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

import org.apache.commons.math3.distribution.EnumeratedDistribution;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealMatrixChangingVisitor;
import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.util.Pair;

import dataset.NNDataset;

public class AntColonyProblem {

	final static int epoch = 1000;
	final static int ANTS_COUNT = 300;
	final static int debug = 10;

	final static double evaporation = 0.2d;
	final static double occurrent = 0.05d;
	final static double alpha = 1d;
	final static double beta = 5d;
	final static double CONST_PHEROMONE = 500;

	// CONST_PHEROMONE / ANTS_COUNT + 10 * Math.random();
	final static double pheromone = Math.random();

	public static void main(String[] args) {
		RealMatrix cost = NNDataset.getWeights(NNDataset.TSP);
		List<Ant> history = new ArrayList<Ant>();
		double best = train(cost, history);
		System.out.println("Best cost is: " + best);
		int sz = 5 < history.size() ? 5 : history.size();
		for (int i = 1; i <= sz; i++) {
			Ant ant = history.get(history.size() - i);
			printTravelPath(ant, cost);
		}
	}

	static double train(RealMatrix cost, List<Ant> history) {
		// best cost here
		double ret = Double.MAX_VALUE;

		// caution: large cost, small weight
		int r = cost.getRowDimension();
		int c = cost.getColumnDimension();
		RealMatrix withPheromone = MatrixUtils.createRealMatrix(r, c);
		withPheromone.walkInOptimizedOrder(new RealMatrixChangingVisitor() {
			@Override
			public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {
			}

			@Override
			public double end() {
				return 0;
			}

			@Override
			public double visit(int row, int column, double value) {
				return AntColonyProblem.pheromone;
			}
		});

		int turn = r - 1;

		for (int i = 0; i < epoch; i++) {
			List<Ant> ants = new ArrayList<Ant>();
			// reset ant
			int start = ThreadLocalRandom.current().nextInt(r);
			for (int j = 0; j < ANTS_COUNT; j++) {
				ants.add(new Ant(cost, withPheromone, occurrent, (start + j) % r));
			}

			// ant moving
			for (Ant ant : ants) {
				for (int j = 0; j < turn; j++) {
					ant.visitNext();
				}
			}

			// update pheromone
			for (Ant ant : ants) {
				updatePheromone(withPheromone, ant);
			}

			// pick up best
			Collections.sort(ants);
			if (ants.get(0).cost < ret) {
				ret = ants.get(0).cost;
				history.add(ants.get(0));
			}

			if (i % debug == 0) {
				System.out.println("at epoch " + i + ", the best cost: " + ret);
			}
		}

		return ret;
	}

	static void updatePheromone(RealMatrix withPheromone, Ant ant) {
		final double contriubted = 1.0d / ant.cost;
		final int[] path = ant.path;
		RealMatrixChangingVisitor updator = new RealMatrixChangingVisitor() {
			@Override
			public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {
			}

			@Override
			public double end() {
				return 0;
			}

			@Override
			public double visit(int row, int column, double oldPheromone) {
				double ret = oldPheromone;
				int from = path[row];
				int to = path[column];
				if (from + 1 == to) {
					ret = evaporation * oldPheromone + (1 - evaporation) * contriubted;
				}
				return ret;
			}
		};

		withPheromone.walkInOptimizedOrder(updator);
	}

	static void printTravelPath(Ant ant, RealMatrix cost) {
		List<City> road = new ArrayList<City>();
		for (int i = 0; i < ant.path.length; i++) {
			road.add(new City(ant.path[i], i));
		}
		Collections.sort(road);
		double total_cost = 0d;
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < road.size(); i++) {
			int now = road.get(i).idx;
			double c = 0d;
			if (i > 0) {
				int from = road.get(i - 1).idx;
				int to = now;
				c = cost.getEntry(from, to);
			}
			total_cost += c;
			sb.append(now).append("[").append(c).append("]").append(" --> ");
		}
		int sz = " --> ".length();
		sb.delete(sb.length() - sz, sb.length());
		sb.insert(0, " ==> ").insert(0, total_cost).insert(0, "total cost = ");
		System.out.println(sb.toString());
	}

}

class Ant implements Comparable<Ant> {

	final Random rnd = new Random(System.nanoTime() % 113);
	RealMatrix pheromone;
	RealMatrix weight;

	// crazy probability
	double crazy;

	// total cost, less is better
	double cost;

	int turn; // now turn at
	int[] path; // node visited path, -1 is never
	int currentPos; // now at node

	Ant(RealMatrix wt, RealMatrix withPheromone, double occurrent, int start) {
		this.pheromone = withPheromone;
		this.weight = wt;
		this.crazy = occurrent;
		this.cost = 0d;
		this.path = new int[withPheromone.getRowDimension()];
		Arrays.fill(this.path, -1);

		// random initialization
		this.turn = 0;
		this.currentPos = start;
		this.path[currentPos] = turn;
	}

	void visitNext() {
		turn++;
		if (rnd.nextDouble() < crazy) {
			// real crazy mode
			int next = -1;
			for (int i = rnd.nextInt(path.length); i < path.length * 2; i++) {
				if (path[i % path.length] == -1) {
					next = i % path.length;
					path[next] = turn;
					break;
				}
			}
			cost += this.weight.getEntry(currentPos, next);
			currentPos = next;
			return;
		}
		double[] p = new double[path.length];
		double total = 0d;
		for (int i = 0; i < p.length; i++) {
			p[i] = 0;
			if (path[i] != -1) {
				continue;
			}
			double ph = this.pheromone.getEntry(currentPos, i);
			double wt = FastMath.pow(1.0d / this.weight.getEntry(currentPos, i), AntColonyProblem.beta);
			p[i] = FastMath.pow(ph, AntColonyProblem.alpha) * wt;
			total += p[i];
		}
		List<Pair<Integer, Double>> probs = new ArrayList<Pair<Integer, Double>>();
		for (int i = 0; i < p.length; i++) {
			if (p[i] != 0d) {
				p[i] /= total;
				probs.add(new Pair<Integer, Double>(i, p[i]));
			}
		}
		EnumeratedDistribution<Integer> selector = new EnumeratedDistribution<>(probs);
		Integer next = selector.sample();
		cost += this.weight.getEntry(currentPos, next);
		path[next] = turn;
		currentPos = next;
	}

	@Override
	public int compareTo(Ant o) {
		return Double.valueOf(this.cost).compareTo(o.cost);
	}

}

class City implements Comparable<City> {

	int seq;
	int idx;

	City(int s, int idx) {
		this.seq = s;
		this.idx = idx;
	}

	@Override
	public int compareTo(City o) {
		return Integer.valueOf(seq).compareTo(o.seq);
	}

}
