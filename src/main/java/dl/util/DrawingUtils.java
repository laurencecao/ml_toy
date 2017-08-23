package dl.util;

import java.io.IOException;

import org.knowm.xchart.BitmapEncoder;
import org.knowm.xchart.BitmapEncoder.BitmapFormat;
import org.knowm.xchart.QuickChart;
import org.knowm.xchart.XYChart;

public class DrawingUtils {

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

		// Create Chart
		XYChart chart = QuickChart.getChart("MSE", "Epoch", "Error", "mean square eror", x, y);

		// or save it in high-res
		BitmapEncoder.saveBitmapWithDPI(chart, path, BitmapFormat.PNG, 300);
	}

}