package dl.bandit;

import java.util.concurrent.ThreadLocalRandom;

import org.apache.commons.collections4.queue.CircularFifoQueue;

public class SlotMachine {

	final static int CAPACITY = 10000;
	double[] probs;
	CircularFifoQueue<Boolean> history;
	CircularFifoQueue<Integer> selectionHistory;

	public SlotMachine(double... probs) {
		this.probs = probs;
		history = new CircularFifoQueue<Boolean>(CAPACITY);
		selectionHistory = new CircularFifoQueue<>(CAPACITY);
	}

	public boolean next(int idx) {
		ThreadLocalRandom rnd = ThreadLocalRandom.current();
		boolean ret = rnd.nextDouble() < probs[idx];
		history.add(ret);
		selectionHistory.add(idx);
		return ret;
	}

	public int getArmCount() {
		return probs.length;
	}

	public int getReward() {
		int ret = 0;
		for (Boolean b : history) {
			ret += b ? 1 : 0;
		}
		return ret;
	}

	public int[][] getHistory() {
		int[][] ret = new int[history.size()][];
		for (int i = 0; i < history.size(); i++) {
			ret[i] = new int[] { 0, 0 };
			ret[i][0] = selectionHistory.get(i);
			ret[i][1] = history.get(i) ? 1 : 0;
		}
		return ret;
	}

}
