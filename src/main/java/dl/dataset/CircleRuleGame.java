package dl.dataset;

public class CircleRuleGame {

	static final double inputs[][] = { { 0, 3, 0, 0, 0, 0, 0, 1, 0, 0 }, { 1, 0, 0, 0, 0, 0, 0, 0, 2, 1 },
			{ 0, 1, 2, 0, 0, 0, 0, 1, 0, 0 }, { 0, 0, 0, 0, 0, 0, 4, 0, 0, 0 }, { 0, 4, 0, 0, 0, 0, 0, 0, 0, 0 },
			{ 0, 0, 4, 0, 0, 0, 0, 0, 0, 0 }, { 0, 0, 1, 0, 0, 0, 2, 1, 0, 0 }, { 0, 1, 0, 2, 0, 0, 0, 0, 0, 1 },
			{ 4, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 4, 0, 0, 0, 0 }, { 0, 1, 0, 1, 0, 0, 0, 0, 1, 1 },
			{ 1, 0, 0, 0, 0, 0, 1, 0, 1, 1 }, { 0, 0, 0, 1, 1, 0, 0, 0, 1, 1 }, { 0, 0, 0, 0, 1, 1, 0, 1, 0, 1 },
			{ 1, 0, 0, 1, 0, 0, 0, 0, 1, 1 }, { 0, 1, 0, 1, 1, 0, 0, 0, 1, 0 }, { 0, 0, 0, 0, 0, 0, 0, 0, 4, 0 },
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 4 }

	};

	static final double result[] = { 0, 6d, 0, 4d, 0d, 0d, 2d, 1d, 4d, 0, 3d, 5d, 3d, 1d, 4d, 2d, 8d, 4d };

	static final double question[][] = { { 1, 0, 0, 0, 0, 0, 0, 0, 2, 1 }, { 1, 2, 0, 0, 0, 0, 0, 0, 0, 1 } };

	public static int count() {
		return inputs.length;
	}

	public static double getLabel(int idx) {
		return result[idx];
	}

	public static double[] getData(int idx) {
		return inputs[idx];
	}

	public static double[] getQuestion(int idx) {
		return question[idx];
	}

}
