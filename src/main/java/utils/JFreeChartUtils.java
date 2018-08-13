package utils;

import java.io.FileOutputStream;
import java.io.IOException;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtilities;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

public class JFreeChartUtils {

	static public void drawLineAndScatter(String out, String title, double[][] scatter, double[][] line) throws IOException {
		try (FileOutputStream of = new FileOutputStream(out);) {

			XYSeriesCollection seriesCollection = new XYSeriesCollection();

			// line plot
			XYSeries lineData = new XYSeries("Probability Density");
			for (int i = 0; i < line.length; i++) {
				lineData.add(line[i][0], line[i][1]);
			}
			seriesCollection.addSeries(lineData);

			// scatter plot
			XYSeries series = new XYSeries("Sample Points");
			for (int i = 0; i < scatter.length; i++) {
				series.add(scatter[i][0], scatter[i][1]);
			}
			seriesCollection.addSeries(series);

			JFreeChart chart = ChartFactory.createXYLineChart(title, "X", "P(X)", seriesCollection,
					PlotOrientation.VERTICAL, true, true, false);

			XYPlot plot = (XYPlot) chart.getPlot();
			XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();

			// "0" is the line plot
			renderer.setSeriesLinesVisible(0, true);
			renderer.setSeriesShapesVisible(0, false);

			// "1" is the scatter plot
			renderer.setSeriesLinesVisible(1, false);
			renderer.setSeriesShapesVisible(1, true);

			plot.setRenderer(renderer);
			ChartUtilities.writeChartAsPNG(of, chart, 800, 600);
		} catch (Exception e) {
			System.err.println(e.toString());
		}
	}

}
