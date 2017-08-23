package utils;

import java.io.IOException;

import org.knowm.xchart.BitmapEncoder;
import org.knowm.xchart.BitmapEncoder.BitmapFormat;
import org.knowm.xchart.QuickChart;
import org.knowm.xchart.XYChart;

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

}
