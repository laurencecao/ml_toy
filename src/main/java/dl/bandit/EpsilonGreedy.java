package dl.bandit;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;

import utils.DrawingUtils;

public class EpsilonGreedy {

	final static double EPSILON = 0.1d;

	public static void main(String[] args) throws IOException {
		int TURN = 10000;
		// probability for 1
		SlotMachine machine = new SlotMachine(0.4d, 0.7d, 0.1d, 0.2d, 0.9d);
		sampling(machine, TURN);
		System.out.println("Total [" + TURN + "] reward: " + machine.getReward());
	}

	static void sampling(SlotMachine slot, int TURN) throws IOException {
		// initialization
		List<ContextMean> dist = new ArrayList<ContextMean>();
		for (int i = 0; i < slot.getArmCount(); i++) {
			dist.add(new ContextMean());
		}

		for (int i = 0; i < slot.getArmCount(); i++) {
			boolean hit = slot.next(i);
			greedyPolicy2(dist.get(i), hit);
		}

		// testing
		for (int i = 0; i < TURN - slot.getArmCount(); i++) {
			int idx = greedyPolicy1(dist, EPSILON); // decide which to trial
			boolean hit = slot.next(idx); // real world reward
			greedyPolicy2(dist.get(idx), hit); // feedback to trial model
		}
		drawing(slot, dist, TURN);
	}

	static int greedyPolicy1(List<ContextMean> ctx, double epsilon) {
		ThreadLocalRandom rnd = ThreadLocalRandom.current();
		if (rnd.nextDouble() < epsilon) { // greedy
			return rnd.nextInt(ctx.size());
		}
		int ret = 0;
		double max = -1d;
		for (int i = 0; i < ctx.size(); i++) {
			ContextMean m = ctx.get(i);
			if (m.summary.getMean() >= max) {
				max = m.summary.getMean();
				ret = i;
			}
		}
		return ret;
	}

	static void greedyPolicy2(ContextMean ctx, boolean hit) {
		ctx.count += 1;
		ctx.summary.addValue(hit ? 1d : 0d);
	}

	static void drawing(SlotMachine slot, List<ContextMean> ctx, int turn) throws IOException {
		List<String> title = new ArrayList<String>();
		for (int i = 0; i < ctx.size(); i++) {
			title.add("SlotMachine_" + i);
		}
		List<double[][]> xy = new ArrayList<double[][]>();
		for (int i = 0; i < slot.getArmCount(); i++) {
			xy.add(new double[][] { new double[turn + 1], // turn idx
					new double[turn + 1], // mean,
					new double[1]// total_count
			});
			xy.get(i)[0][0] = 0d; // initial zero point
			// initial zero point count
			xy.get(i)[1][0] = 0;
			xy.get(i)[2][0] = 0;
		}

		int[][] history = slot.getHistory();
		for (int i = 1; i <= history.length; i++) {
			for (int j = 0; j < xy.size(); j++) {
				xy.get(j)[0][i] = i;
				xy.get(j)[1][i] = xy.get(j)[1][i - 1];
			}
			int idx = history[i - 1][0];
			int reward = history[i - 1][1];
			// new_avg = avg0 + (new_x - avg0) / (size + 1)
			xy.get(idx)[2][0] += 1d;
			xy.get(idx)[1][i] = xy.get(idx)[1][i - 1] + (reward - xy.get(idx)[1][i - 1]) / (xy.get(idx)[2][0]);
		}
		DrawingUtils.drawMultiSeries(title, xy, "tmp/epsilon.png");
	}

}

class ContextMean {

	int count = 0;
	SummaryStatistics summary = new SummaryStatistics();

}
