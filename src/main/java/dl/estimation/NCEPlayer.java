package dl.estimation;

import java.util.Arrays;
import java.util.List;

import org.apache.commons.math3.distribution.LogNormalDistribution;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartFrame;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.statistics.HistogramDataset;
import org.jfree.data.statistics.HistogramType;

public class NCEPlayer {
	
	final static int n = 100; // for x
	final static int m = 100; // for y

	public static void main(String[] args) {
		int sampleSize = 1000;
		display(Arrays.asList(distX(sampleSize), distY(sampleSize)), Arrays.asList("LogNormal", "Gaussian"));
	}

	static void estimation() {
		
	}
	
	static void loss(double[] x, double[] y) {
		
	}

	static double[] distY(int sampleSize) {
		// Gaussian Distribution
		NormalDistribution dist = new NormalDistribution(0.8d, 0.5d);
		return dist.sample(sampleSize);
	}

	static double[] distX(int sampleSize) {
		// LogNormal Distribution
		LogNormalDistribution dist = new LogNormalDistribution(0, 1d / 8d);
		return dist.sample(sampleSize);
	}

	static void display(List<double[]> dataset, List<String> title) {
		int bin = 20;
		HistogramDataset h = new HistogramDataset();
		h.setType(HistogramType.RELATIVE_FREQUENCY);
		for (int i = 0; i < dataset.size(); i++) {
			h.addSeries(title.get(i), dataset.get(i), bin);
		}
		JFreeChart chart = ChartFactory.createHistogram("distribution", "value", "percent", h, PlotOrientation.VERTICAL,
				true, true, false);
		try {
			ChartFrame cf = new ChartFrame("分布对比图", chart);
			cf.pack();
			cf.setSize(800, 600);
			cf.setVisible(true);
		} catch (Exception e) {
			System.err.println("Problem occurred creating chart.");
		}
	}

}
