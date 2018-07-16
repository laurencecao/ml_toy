package dataset;

import java.util.function.Function;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.knowm.xchart.CategoryChart;
import org.knowm.xchart.CategoryChartBuilder;
import org.knowm.xchart.CategorySeries.CategorySeriesRenderStyle;
import org.knowm.xchart.SwingWrapper;

public class Trajectory {

	final static int SIZE = 100;

	final static double slope = 0.01d;

	final static double est_mean = 0d;
	final static double est_sd = 0.1d;
	final static NormalDistribution est = new NormalDistribution(est_mean, est_sd);

	final static double measure_mean = 0d;
	final static double measure_sd = 0.3d;
	final static NormalDistribution measure = new NormalDistribution(measure_mean, measure_sd);

	static Function<Double, double[]> getPoint = x -> {
		double xx = slope * x;
		return new double[] { xx + est.sample(), xx + measure.sample(), xx };
	};

	static double[][] getline(int x) {
		double[][] ret = new double[][] { new double[x], new double[x], new double[x], new double[x] };
		double step = 1.0d * x / SIZE;
		for (int i = 1; i < x + 1; i++) {
			double xx = step * i;
			ret[0][i - 1] = xx;
			double[] n = getPoint.apply(xx);
			ret[1][i - 1] = n[0]; // observe
			ret[2][i - 1] = n[1]; // measure
			ret[3][i - 1] = n[2]; // real
		}
		return ret;
	};

	public static double[][] getData() {
		return getline(SIZE);
	}

	@SuppressWarnings({ "unchecked", "rawtypes" })
	public static void main(String[] args) throws Exception {
		double[][] xy = getData();

		CategoryChart chart = new CategoryChartBuilder().width(800).height(600).title("trajectory").xAxisTitle("point")
				.yAxisTitle("trajectory").build();

		// Customize Chart
		chart.getStyler().setDefaultSeriesRenderStyle(CategorySeriesRenderStyle.Line);
		// chart.getStyler().setXAxisLabelRotation(270);
		// chart.getStyler().setLegendPosition(LegendPosition.OutsideE);
		// chart.getStyler().setAvailableSpaceFill(0);
		// chart.getStyler().setOverlapped(true);

		chart.addSeries("observe", xy[0], xy[1]);
		chart.addSeries("measure", xy[0], xy[2]);
		chart.addSeries("real", xy[0], xy[3]);

		new SwingWrapper(chart).displayChart();

		// BitmapEncoder.saveBitmap(chart, "./Sample_Chart", BitmapFormat.PNG);
		// BitmapEncoder.saveBitmapWithDPI(chart, "./Sample_Chart_300_DPI",
		// BitmapFormat.PNG, 300);
	}

}
