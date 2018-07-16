package dl.kalman;

import org.apache.commons.math3.filter.KalmanFilter;
import org.apache.commons.math3.filter.MeasurementModel;
import org.apache.commons.math3.filter.ProcessModel;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.knowm.xchart.CategoryChart;
import org.knowm.xchart.CategoryChartBuilder;
import org.knowm.xchart.CategorySeries.CategorySeriesRenderStyle;
import org.knowm.xchart.SwingWrapper;

import dataset.Trajectory;

public class ToyKalmanFilter {

	static MyProcess proc = new MyProcess();
	static MyMeasurement measurement = new MyMeasurement();

	public static void main(String[] args) {
		double[][] data = Trajectory.getData();
		double[] trajectory = play(data);
		double[] trajectory2 = play2(data);
		draw(data, trajectory, trajectory2);
	}

	static double[] play(double[][] data) {
		KalmanFilter kf = new KalmanFilter(proc, measurement);
		double[] ret = new double[data[0].length];
		double[] delta = new double[] { data[0][1] - data[0][0] };
		for (int i = 0; i < ret.length; i++) {
			kf.predict(delta);
			kf.correct(new double[] { data[2][i] });
			ret[i] = kf.getStateEstimation()[0];
		}
		return ret;
	}

	static double[] play2(double[][] data) {
		SimpleKalmanFilter kf = new SimpleKalmanFilter(proc.init, MatrixUtils.createRealVector(new double[] { 0.01d }),
				MatrixUtils.createRealIdentityMatrix(1), MatrixUtils.createRealIdentityMatrix(1),
				MatrixUtils.createRealIdentityMatrix(1));
		double[] ret = new double[data[0].length];
		for (int i = 0; i < ret.length; i++) {
			kf.estimate(proc.noisy);
			kf.update(MatrixUtils.createRealVector(new double[] { data[2][i] }), measurement.noisy);
			ret[i] = kf.getState().getEntry(0);
		}
		return ret;
	}

	@SuppressWarnings({ "unchecked", "rawtypes" })
	static void draw(double[][] data, double[] kfdata, double[] kfdata2) {
		CategoryChart chart = new CategoryChartBuilder().width(800).height(600).title("trajectory").xAxisTitle("point")
				.yAxisTitle("trajectory").build();

		// Customize Chart
		chart.getStyler().setDefaultSeriesRenderStyle(CategorySeriesRenderStyle.Line);

		chart.addSeries("estimation", data[0], data[1]);
		chart.addSeries("measure", data[0], data[2]);
		chart.addSeries("real", data[0], data[3]);
		chart.addSeries("trajectory", data[0], kfdata);
		chart.addSeries("trajectory2", data[0], kfdata2);

		new SwingWrapper(chart).displayChart();
	}

}

class MyProcess implements ProcessModel {

	RealMatrix ctrl = MatrixUtils.createRealMatrix(new double[][] { { 0.01d } });
	RealMatrix identity = MatrixUtils.createRealMatrix(new double[][] { { 1d } });
	RealMatrix noisy = MatrixUtils.createRealMatrix(new double[][] { { 0.01d } });

	RealVector init = MatrixUtils.createRealVector(new double[] { Math.random() });
	RealMatrix covar = MatrixUtils.createRealMatrix(new double[][] { { 0.1d } });

	@Override
	public RealMatrix getStateTransitionMatrix() {
		return identity;
	}

	@Override
	public RealMatrix getControlMatrix() {
		return ctrl;
	}

	@Override
	public RealMatrix getProcessNoise() {
		return noisy;
	}

	@Override
	public RealVector getInitialStateEstimate() {
		return init;
	}

	@Override
	public RealMatrix getInitialErrorCovariance() {
		return covar;
	}

}

class MyMeasurement implements MeasurementModel {

	RealMatrix measure = MatrixUtils.createRealMatrix(new double[][] { { 1d } });
	RealMatrix noisy = MatrixUtils.createRealMatrix(new double[][] { { 0.3d } });

	@Override
	public RealMatrix getMeasurementMatrix() {
		return measure;
	}

	@Override
	public RealMatrix getMeasurementNoise() {
		return noisy;
	}

}