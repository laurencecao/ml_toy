package utils;

import static guru.nidi.graphviz.model.Factory.graph;
import static guru.nidi.graphviz.model.Factory.node;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.lang3.ArrayUtils;
import org.knowm.xchart.BitmapEncoder;
import org.knowm.xchart.BitmapEncoder.BitmapFormat;
import org.knowm.xchart.CategoryChart;
import org.knowm.xchart.CategoryChartBuilder;
import org.knowm.xchart.Histogram;
import org.knowm.xchart.QuickChart;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.knowm.xchart.XYSeries;
import org.knowm.xchart.XYSeries.XYSeriesRenderStyle;
import org.knowm.xchart.style.Styler.LegendPosition;
import org.knowm.xchart.style.markers.Marker;
import org.knowm.xchart.style.markers.SeriesMarkers;

import dl.dt.DecisionNode;
//import guru.nidi.graphviz.attribute.Color;
//import guru.nidi.graphviz.attribute.Shape;
import guru.nidi.graphviz.attribute.Style;
import guru.nidi.graphviz.engine.Format;
import guru.nidi.graphviz.engine.Graphviz;
import guru.nidi.graphviz.model.Graph;
import guru.nidi.graphviz.model.Label;
import guru.nidi.graphviz.model.Link;
import guru.nidi.graphviz.model.Node;

public class DrawingUtils {

	public static void drawMSE(double[] err, int end, double step, String path) throws IOException {
		int sz = err.length;
		double[] x = new double[sz];
		double _x = end - sz * step;
		for (int i = 0; i < sz; i++) {
			x[i] = _x + step;
			_x += step;
		}

		drawMSE(err, x, path);
	}

	public static void drawMSE(double[] err, double last, int end, double step, String path) throws IOException {
		int sz = err.length;
		double[] x = new double[sz + 1];
		double _x = end - sz * step;
		for (int i = 0; i < sz; i++) {
			x[i] = _x + step;
			_x += step;
		}
		x[sz] = _x + step;
		double[] y = new double[sz + 1];
		System.arraycopy(err, 0, y, 0, sz);
		y[sz] = last;

		drawMSE(y, x, path);
	}

	public static void drawMSE(double[] err, double[] epoch, String path) throws IOException {
		XYChart chart = QuickChart.getChart("MSE", "Epoch", "Error", "mean square eror", epoch, err);
		BitmapEncoder.saveBitmapWithDPI(chart, path, BitmapFormat.PNG, 300);
	}

	public static void drawTree(dl.dt.DecisionNode root, String[] header, String path) throws IOException {
		final List<Node> allNode = new ArrayList<Node>();
		final Map<dl.dt.DecisionNode, Node> dict = new HashMap<>();
		dict.put(root, node(DecisionNode.dump(root, header)));
		dl.dt.DecisionNode.visitTree(root, x -> {
			// String msg = DecisionNode.dump(x);
			// Node n = node(msg);
			// dict.put(x, n);
			if (x.children != null) {
				Node[] cc = new Node[x.children.length];
				for (int i = 0; i < x.children.length; i++) {
					cc[i] = node(DecisionNode.dump(x.children[i], header));
					dict.put(x.children[i], cc[i]);
				}
				Node np = dict.get(x);
				Node lk = np.link(cc);
				Link.to(lk).with(Style.BOLD, Label.of(String.valueOf(x.searchVal[0])));
				allNode.add(lk);
			}
			// allNode.add(n);
			System.out.println(x.toString());
			return x;
		});

		// for (Node n : allNode) {
		// System.out.println(n);
		// }
		System.out.println(allNode.size());

		Graph g = graph("C4.5 Decision Tree").directed().with(allNode.toArray(new Node[allNode.size()]));
		Graphviz.fromGraph(g).width(3000).render(Format.PNG).toFile(new File(path));
	}

	public static void drawClusterXY(List<String> title, List<double[][]> data, String path) throws IOException {
		XYChart chart = new XYChartBuilder().title("Cluster").xAxisTitle("X").yAxisTitle("Y").build();
		chart.getStyler().setDefaultSeriesRenderStyle(XYSeriesRenderStyle.Scatter);
		chart.getStyler().setChartTitleVisible(false);
		chart.getStyler().setLegendPosition(LegendPosition.InsideSW);
		chart.getStyler().setMarkerSize(16);
		Marker[] cap = new Marker[] { SeriesMarkers.CIRCLE, SeriesMarkers.DIAMOND, SeriesMarkers.SQUARE,
				SeriesMarkers.TRIANGLE_DOWN, SeriesMarkers.TRIANGLE_UP };
		for (int i = 0; i < title.size(); i++) {
			XYSeries series = null;
			series = chart.addSeries(title.get(i), data.get(i)[0], data.get(i)[1]);
			series.setMarker(cap[i % cap.length]);
		}
		BitmapEncoder.saveBitmapWithDPI(chart, path, BitmapFormat.PNG, 300);
	}

	public static void drawMultiSeries(List<String> title, List<double[][]> data, String path) throws IOException {
		final XYChart chart = new XYChartBuilder().title("probabilities").xAxisTitle("epoch").yAxisTitle("Probability")
				.build();
		chart.getStyler().setMarkerSize(1);
		// Customize Chart
		// chart.getStyler().setLegendPosition(LegendPosition.InsideNE);
		// chart.getStyler().setDefaultSeriesRenderStyle(XYSeriesRenderStyle.Area);
		for (int i = 0; i < data.size(); i++) {
			chart.addSeries("Slot Machine " + i, data.get(i)[0], data.get(i)[1]);
		}
		BitmapEncoder.saveBitmapWithDPI(chart, path, BitmapFormat.PNG, 300);
	}

	public static void drawSampling(double[] err, double[] epoch, String path, String[] title) throws IOException {
		XYChart chart = QuickChart.getChart(title[0], "Epoch", title[1], title[2], epoch, err);
		BitmapEncoder.saveBitmapWithDPI(chart, path, BitmapFormat.PNG, 300);
	}

	public static void drawHistogram(List<String> title, List<double[]> err, double[] range, String path)
			throws IOException {
		CategoryChart chart = new CategoryChartBuilder().title("Histogram").xAxisTitle("X").yAxisTitle("Y").build();
		chart.getStyler().setLegendPosition(LegendPosition.InsideNW);
		chart.getStyler().setAvailableSpaceFill(.96);
		chart.getStyler().setOverlapped(true);
		for (int i = 0; i < err.size(); i++) {
			List<Double> v = Arrays.asList(ArrayUtils.toObject(err.get(i)));
			Histogram histogram1 = new Histogram(v, Double.valueOf(range[0]).intValue(), range[1], range[2]);
			chart.addSeries(title.get(i), histogram1.getxAxisData(), histogram1.getyAxisData());
		}
		BitmapEncoder.saveBitmapWithDPI(chart, path, BitmapFormat.PNG, 300);
	}

}
