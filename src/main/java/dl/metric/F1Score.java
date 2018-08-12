package dl.metric;

public class F1Score {

	protected int tp = 0;
	protected int fp = 0;
	protected int tn = 0;
	protected int fn = 0;

	public int incrTP() {
		tp++;
		return tp;
	}

	public int incrFP() {
		fp++;
		return fp;
	}

	public int incrTN() {
		tn++;
		return tn;
	}

	public int incrFN() {
		fn++;
		return fn;
	}

	public double getPrecision() {
		return tp / (tp + fp);
	}

	public double getRecall() {
		return tp / (tp + fn);
	}

	public double getAccuracy() {
		return (tp + tn) / (tp + tn + fp + fn);
	}

}
