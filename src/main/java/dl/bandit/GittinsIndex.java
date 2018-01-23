package dl.bandit;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

import org.apache.commons.math3.util.FastMath;

import utils.DrawingUtils;

/**
 * modified Gittins Index
 * 
 * @author caowenjiong
 *
 */
public class GittinsIndex {

	public static void main(String[] args) throws IOException {
		int TURN = 1000;
		double gamma = 0.6d;
		int step = 50;
		// probability for 1
		SlotMachine machine = new SlotMachine(0.5d, 0.73d, 0.6d, 0.45d, 0.75d);
		sampling(machine, TURN, gamma, step);
		System.out.println("Total [" + TURN + "] reward: " + machine.getReward());
	}

	static void sampling(SlotMachine slot, int TURN, double gamma, int step) throws IOException {
		// initialization
		// just take it easier
		List<ContextUCB> dist = new ArrayList<ContextUCB>();
		for (int i = 0; i < slot.getArmCount(); i++) {
			dist.add(new ContextUCB());
			boolean hit = slot.next(i);
			ledger(dist.get(i), hit);
		}

		// testing
		for (int i = slot.getArmCount(); i < TURN; i++) {
			// decide which to trial
			int idx = gittinsIndex(dist, i, gamma, step);
			boolean hit = slot.next(idx); // real world reward
			ledger(dist.get(idx), hit); // feedback to trial model
		}
		drawing(slot, dist, TURN);
	}

	static double calcSupremum(double probs, double gamma, int step) {
		Random rnd = ThreadLocalRandom.current();
		double ret = 0;
		for (int i = 0; i < step; i++) {
			int v = rnd.nextDouble() < probs ? 1 : 0;
			ret += v * FastMath.pow(gamma, step);
		}
		return ret;
	}

	static int gittinsIndex(List<ContextUCB> ctx, int currTurn, double gamma, int step) {
		int ret = 0;
		double max = -1d;
		for (int i = 0; i < ctx.size(); i++) {
			ContextUCB m = ctx.get(i);
			double p = m.summary.getMean();
			double vi = calcSupremum(p, gamma, step);
			if (vi >= max) {
				max = vi;
				ret = i;
			}
		}
		return ret;
	}

	static void ledger(ContextUCB ctx, boolean hit) {
		ctx.count += 1;
		ctx.summary.addValue(hit ? 1d : 0d);
	}

	static void drawing(SlotMachine slot, List<ContextUCB> ctx, int turn) throws IOException {
		List<String> title = new ArrayList<String>();
		for (int i = 0; i < ctx.size(); i++) {
			title.add("SlotMachine_" + i);
		}
		List<double[][]> xy = new ArrayList<double[][]>();
		for (int i = 0; i < slot.getArmCount(); i++) {
			xy.add(new double[][] { new double[turn + 1], // turn idx
					new double[turn + 1], // mean + bonus,
					new double[1], // total_count
					new double[turn + 1] // mean
			});
			xy.get(i)[0][0] = 0d; // initial zero point
			// initial zero point count
			xy.get(i)[1][0] = 0;
			xy.get(i)[2][0] = 0;
			xy.get(i)[3][0] = 0;
		}

		int[][] history = slot.getHistory();
		for (int i = 1; i <= slot.getArmCount(); i++) {
			for (int j = 0; j < xy.size(); j++) {
				xy.get(j)[0][i] = i;
				xy.get(j)[1][i] = xy.get(j)[1][i - 1];
				xy.get(j)[3][i] = xy.get(j)[1][i];
			}
			int idx = history[i - 1][0];
			int reward = history[i - 1][1];
			xy.get(idx)[2][0] += 1;
			xy.get(idx)[1][i] = xy.get(idx)[1][i - 1] + (reward - xy.get(idx)[1][i - 1]) / (xy.get(idx)[2][0]);
			xy.get(idx)[3][i] = xy.get(idx)[1][i];
		}

		for (int i = slot.getArmCount() + 1; i <= history.length; i++) {
			for (int j = 0; j < xy.size(); j++) {
				xy.get(j)[0][i] = i;
				xy.get(j)[3][i] = xy.get(j)[3][i - 1];
				double ct = xy.get(j)[2][0];
				xy.get(j)[1][i] = xy.get(j)[3][i] + FastMath.sqrt(2 * FastMath.log(ct) / (i - 1));
			}
			int idx = history[i - 1][0];
			int reward = history[i - 1][1];
			// new_avg = avg0 + (new_x - avg0) / (size + 1)
			xy.get(idx)[2][0] += 1d;
			xy.get(idx)[3][i] = xy.get(idx)[1][i - 1] + (reward - xy.get(idx)[1][i - 1]) / (xy.get(idx)[2][0]);
		}
		DrawingUtils.drawMultiSeries(title, xy, "tmp/gittins.png");
	}
}
