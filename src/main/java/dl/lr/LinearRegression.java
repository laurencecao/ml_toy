package dl.lr;

import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

import org.apache.commons.collections4.queue.CircularFifoQueue;
import org.apache.commons.lang3.ArrayUtils;

import dataset.CircleRuleGame;
import utils.DrawingUtils;

public class LinearRegression {

	final static double epi = 0.00000001d;
	final static double alpha = 0.01;

	public static void main(String[] args) throws IOException {
		double[] w = training();
		double r = 0d;
		r = predict(CircleRuleGame.getQuestion(0), w);
		System.out.println(Arrays.toString(CircleRuleGame.getQuestion(0)) + " ---> " + r);
		r = predict(CircleRuleGame.getQuestion(1), w);
		System.out.println(Arrays.toString(CircleRuleGame.getQuestion(1)) + " ---> " + r);
	}

	static double[] training() throws IOException {
		Random rnd = new Random();
		double[] w = rnd.doubles(CircleRuleGame.getData(0).length + 1).toArray();
		double err = 0d;
		err += 1000 * epi;
		CircularFifoQueue<Double> buf = new CircularFifoQueue<Double>(100000);
		while (err > epi) {
			err = 0;
			for (int i = 0; i < CircleRuleGame.count(); i++) {
				w = SGD(CircleRuleGame.getLabel(i), CircleRuleGame.getData(i), w, alpha);
			}
			for (int i = 0; i < CircleRuleGame.count(); i++) {
				err += MSE(CircleRuleGame.getLabel(i), CircleRuleGame.getData(i), w);
			}
			err /= CircleRuleGame.count();
			System.out.println("MSE ---> " + err);
			buf.add(err);
		}
		double[] error = ArrayUtils.toPrimitive(buf.toArray(new Double[buf.size()]));
		DrawingUtils.drawMSE(error, error.length, 1, "/tmp/lr_error.png");
		return w;
	}

	static double predict(double[] x, double[] w) {
		double ret = w[0];
		for (int i = 0; i < x.length; i++) {
			ret += w[i + 1] * x[i];
		}
		return ret;
	}

	static double MSE(double y, double[] x, double[] w) {
		return Math.pow(y - predict(x, w), 2);
	}

	static double[] SGD(double y, double[] x, double[] w, double alpha) {
		// w = w - alpha * (h - y) * x
		double h = predict(x, w);
		double[] theta = new double[x.length + 1];
		theta[0] = w[0] + alpha * 1 * (y - h);
		for (int i = 1; i < theta.length; i++) {
			theta[i] = w[i] + alpha * x[i - 1] * (y - h);
		}
		return theta;
	}

}
