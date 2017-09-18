package dl.util;

import java.awt.AWTException;
import java.awt.Dimension;
import java.awt.GridLayout;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;
import javax.swing.JFrame;
import javax.swing.JPanel;

import smile.math.Math;
import smile.plot.Histogram3D;
import smile.plot.Palette;
import smile.plot.PlotCanvas;

/**
 *
 * @author Haifeng Li
 */
@SuppressWarnings("serial")
public class Histogram3Demo extends JPanel {
	public Histogram3Demo() throws AWTException, IOException {
		super(new GridLayout(1, 2));

		double[][] data = new double[10000][2];
		for (int j = 0; j < data.length; j++) {
			double x, y, r;
			do {
				x = 2 * (Math.random() - 0.5);
				y = 2 * (Math.random() - 0.5);
				r = x * x + y * y;
			} while (r >= 1.0);

			double z = Math.sqrt(-2.0 * Math.log(r) / r);
			data[j][0] = new Double(x * z);
			data[j][1] = new Double(y * z);
		}

		PlotCanvas canvas = Histogram3D.plot(data, 20);
		canvas.setTitle("Histogram 3D");
		add(canvas);

		canvas = Histogram3D.plot(data, 20, Palette.jet(16));
		canvas.setTitle("Histogram 3D with Colormap");
		add(canvas);
	}

	@Override
	public String toString() {
		return "Histogram 3D";
	}

	public static void main(String[] args) throws AWTException, IOException {
		Histogram3Demo demo = new Histogram3Demo();
		// demo.setSize(new Dimension(1200, 1080));
		JFrame frame = new JFrame();
		// frame.setBackground(Color.WHITE);
		// frame.setUndecorated(true);
		frame.getContentPane().add(demo);
		// frame.setSize(new Dimension(1200, 1080));
		frame.setPreferredSize(new Dimension(1200, 1080));
		frame.pack();
//		frame.setVisible(true);

		saveImage(demo);
		frame.dispose();
		// java.awt.Robot().createScreenCapture(screenRect)

		// BufferedImage bi = new BufferedImage(demo.getWidth(),
		// demo.getHeight(), BufferedImage.TYPE_INT_ARGB);
		// Graphics2D graphics = bi.createGraphics();
		// frame.print(graphics);
		// graphics.dispose();
		// frame.dispose();
		// try {
		// ImageIO.write(bi, "png", new File("/tmp/a.png"));
		// } catch (Exception e) {
		// e.printStackTrace();
		// }
	}

	static void saveImage(JPanel panel) {
		BufferedImage img = new BufferedImage(panel.getWidth(), panel.getHeight(), BufferedImage.TYPE_INT_RGB);
		panel.paint(img.getGraphics());
		try {
			ImageIO.write(img, "png", new File("/tmp/a.png"));
			System.out.println("panel saved as image");

		} catch (Exception e) {
			System.out.println("panel not saved" + e.getMessage());
		}
	}
}
