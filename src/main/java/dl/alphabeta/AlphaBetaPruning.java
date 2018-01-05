package dl.alphabeta;

import java.util.ArrayList;
import java.util.Arrays;

import org.apache.commons.math3.util.FastMath;

import utils.ImageUtils;

public class AlphaBetaPruning {

	final static int layer = 4;
	final static int leaves[] = {

			-3, -4, -4,

			1, 15, 2,

			-12, 11, -19,

			-12, -1, -2,

			5, -8, -11,

			-5, 0, -2,

			-15, -3, 15,

			-2, 11, -3,

			14, 1, -6

	};

	public static void main(String[] args) {
		Decision[] trie = init(leaves, layer);
		alphaBetaPlay(trie, layer);
		printTree(trie);
	}

	static Decision[] init(int[] leaf, int layer) {
		int total = 0;
		for (int i = 0; i < layer; i++) {
			total += Double.valueOf(FastMath.pow(3, i)).intValue();
		}
		Decision[] all = new Decision[total];
		Decision root = new Decision(0, true, -1);
		all[0] = root;
		for (int i = 1; i < total; i++) {
			int r = (i - 1) / 3;
			boolean isMax = !all[r].isMax;
			all[i] = new Decision(i, isMax, r);
			Integer[] s = all[r].successors;
			ArrayList<Integer> lst = new ArrayList<Integer>();
			if (s != null) {
				lst.addAll(Arrays.asList(s));
			}
			lst.add(i);
			all[r].successors = lst.toArray(new Integer[lst.size()]);
		}
		int base = 13;
		for (int i = base; i < total; i++) {
			all[i].score = leaf[i - base];
		}
		return all;
	}

	static void printTree(Decision[] tree) {
		for (int i = 0; i < tree.length; i++) {
			System.out.println(tree[i].id + "@" + tree[i].parent + " {" + (tree[i].isMax ? "MAX" : "MIN") + "} " + "==>"
					+ Arrays.toString(tree[i].successors) + "-->" + tree[i].score + "; alpha => " + tree[i].alpha
					+ "; beta => " + tree[i].beta + " ; pruning: " + tree[i].pruning);
		}
	}

	static void alphaBetaPlay(Decision[] tree, int layer) {
		Decision root = tree[0];
		int ret = maxValue(tree, root.id, Integer.MIN_VALUE, Integer.MAX_VALUE);
		System.out.println("Root: " + "alpha => " + ret);
		int w = 35;
		int h = 35;
		ImageUtils.drawAlphaBetaTree(tree, "tmp/ab_tree.jpeg", layer, leaves.length, w, h, new int[][] {

				{ 10 + (w + 10) * leaves.length / 2 - w / 2, 0 },

				{ 10 + (w + 10) * leaves.length / 6 - w / 2, (w + 10) * leaves.length / 6 * 2 },

				{ 10 + (w + 10) * leaves.length / 18 - w / 2, (w + 10) * leaves.length / 18 * 2 },

				{ 10, w + 10 }

		});
	}

	static int maxValue(Decision[] tree, int state, int alpha, int beta) {
		Decision node = tree[state];
		if (node.successors == null) {
			assert node.score != null;
			return node.score;
		}
		int a = alpha;
		int b = beta;
		updateAlpha(node, a, false);
		updateBeta(node, b, false);
		node.pruning.addAll(Arrays.asList(node.successors));
		for (Integer id : node.successors) {
			assert id != null;
			a = FastMath.max(a, minValue(tree, id, a, b));
			node.pruning.remove(id);
			if (a >= b) {
				updateAlpha(node, a, true);
				break;
			}
		}
		updateAlpha(node, a, false);
		return a;
	}

	static int minValue(Decision[] tree, int state, int alpha, int beta) {
		Decision node = tree[state];
		if (node.successors == null) {
			assert node.score != null;
			return node.score;
		}
		int a = alpha;
		int b = beta;
		updateAlpha(node, a, false);
		updateBeta(node, b, false);
		node.pruning.addAll(Arrays.asList(node.successors));
		for (Integer id : node.successors) {
			assert id != null;
			b = FastMath.min(b, maxValue(tree, id, a, b));
			node.pruning.remove(id);
			if (a >= b) {
				updateBeta(node, b, true);
				break;
			}
		}
		updateBeta(node, b, false);
		return b;
	}

	static void updateAlpha(Decision node, int alpha, boolean force) {
		assert node != null;
		if (force || (alpha > node.alpha && alpha <= node.beta)) {
			node.alpha = alpha;
		}
	}

	static void updateBeta(Decision node, int beta, boolean force) {
		assert node != null;
		if (force || (beta < node.beta && beta >= node.alpha)) {
			node.beta = beta;
		}
	}

}
