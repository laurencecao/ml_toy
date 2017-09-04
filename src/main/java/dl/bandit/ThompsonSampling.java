package dl.bandit;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

import org.apache.commons.math3.distribution.BetaDistribution;

import utils.DrawingUtils;

public class ThompsonSampling {

	final static double EXPLORE = 0.05d;

	public static void main(String[] args) throws IOException {
		int TURN = 10000;
		// probability for 1
		SlotMachine machine = new SlotMachine(0.4d, 0.7d, 0.1d, 0.2d, 0.9d);
		sampling(machine, TURN);
		System.out.println("Total [" + TURN + "] reward: " + machine.getReward());
	}

	static void sampling(SlotMachine slot, int TURN) throws IOException {
		// initialization
		List<ContextBeta> dist = new ArrayList<ContextBeta>();
		for (int i = 0; i < slot.getArmCount(); i++) {
			dist.add(new ContextBeta());
		}

		// testing
		for (int i = 0; i < TURN; i++) {
			int idx = thompsonStage1(dist); // decide which to trial
			boolean hit = slot.next(idx); // real world reward
			thompsonStage2(dist.get(idx), hit); // feedback to trial model
		}
		drawing(slot, dist, TURN);
	}

	static int thompsonStage1(List<ContextBeta> ctx) {
		ThreadLocalRandom rnd = ThreadLocalRandom.current();
		if (rnd.nextDouble() < EXPLORE) {
			return rnd.nextInt(ctx.size());
		}
		int ret = 0;
		double max = -1d;
		for (int i = 0; i < ctx.size(); i++) {
			ContextBeta c = ctx.get(i);
			BetaDistribution beta = new BetaDistribution(c.alphaP + c.alphaC, c.betaP + c.betaC);
			double pReward = beta.sample();
			if (pReward > max) {
				ret = i;
				max = pReward;
			}
		}
		return ret;
	}

	static void thompsonStage2(ContextBeta ctx, boolean hit) {
		ctx.alphaC += hit ? 1d : 0d;
		ctx.betaC += hit ? 0d : 1d;
	}

	static void drawing(SlotMachine slot, List<ContextBeta> ctx, int turn) throws IOException {
		List<String> title = new ArrayList<String>();
		for (int i = 0; i < ctx.size(); i++) {
			title.add("SlotMachine_" + i);
		}
		List<double[][]> xy = new ArrayList<double[][]>();
		for (int i = 0; i < slot.getArmCount(); i++) {
			xy.add(new double[][] { new double[turn + 1], // turn idx
					new double[turn + 1], // win probability,
					new double[1], // alphaC + alphaP
					new double[1] }); // total_count
			xy.get(i)[0][0] = 0d; // initial zero point
			// initial zero point probability
			xy.get(i)[1][0] = ctx.get(i).alphaP / (ctx.get(i).alphaP + ctx.get(i).betaP);
			xy.get(i)[2][0] = ctx.get(i).alphaP;
			xy.get(i)[3][0] = ctx.get(i).alphaP + ctx.get(i).betaP;
		}

		int[][] history = slot.getHistory();
		for (int i = 1; i <= history.length; i++) {
			for (int j = 0; j < xy.size(); j++) {
				xy.get(j)[0][i] = i;
				xy.get(j)[1][i] = xy.get(j)[1][i - 1];
			}
			int idx = history[i - 1][0];
			int reward = history[i - 1][1];
			xy.get(idx)[2][0] += reward;
			xy.get(idx)[3][0] += 1d;
			xy.get(idx)[1][i] = xy.get(idx)[2][0] / xy.get(idx)[3][0];
		}
		DrawingUtils.drawMultiSeries(title, xy, "tmp/thompson.png");
	}

}

class ContextBeta {

	double alphaP = 10d;
	double betaP = 10d;

	double alphaC = 0d;
	double betaC = 0d;

}