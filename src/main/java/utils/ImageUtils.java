package utils;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import javax.imageio.ImageIO;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.util.FastMath;

public class ImageUtils {

	public final static double TITTLE = 0.0001d;
	final static int PIXEL_END = 255;

	static double[] normalize(double[] pixel) {
		List<Double> lst = Arrays.asList(ArrayUtils.toObject(pixel));
		// minimal >= TITTLE
		return lst.stream().mapToDouble(x -> {
			x = x <= 0 ? 0 : x;
			return x + TITTLE;
		}).toArray();

	}

	public static double[] BMP2Vector(String imgpath) throws IOException {
		BufferedImage img = ImageIO.read(new File(imgpath));
		// int a = 0;
		// a = 100/a;
		int x = img.getWidth();
		int y = img.getHeight();
		int pos = 0;
		double[] ret = new double[x * y];
		double max = Double.MIN_VALUE;
		double min = Double.MAX_VALUE;
		for (int i = 0; i < y; i++) {
			for (int j = 0; j < x; j++) {
				int p = img.getRGB(j, i);
				// int r = (p >> 16) & 0xff;
				// int g = (p >> 8) & 0xff;
				int b = p & 0xff;
				// int avg = (299 * r + 587 * g + 114 * b) / 1000;
				ret[pos] = b;
				if (ret[pos] > max) {
					max = ret[pos];
				}
				if (ret[pos] < min) {
					min = ret[pos];
				}
				pos++;
			}
		}
		return normalize(ret);
	}

	public static void Vector2BMP(double[] val, String imgpath) throws IOException {
		double[] imgV = val;
		int R = Double.valueOf(FastMath.sqrt(imgV.length)).intValue();
		BufferedImage img = new BufferedImage(R, R, BufferedImage.TYPE_BYTE_GRAY);

		int x = img.getWidth();
		int y = img.getHeight();
		int pos = 0;
		for (int i = 0; i < y; i++) {
			for (int j = 0; j < x; j++) {
				int a = 0;
				int avg = Double.valueOf(imgV[pos]).intValue();
				int p = (a << 24) | (avg << 16) | (avg << 8) | avg;
				img.setRGB(j, i, p);
				pos++;
			}
		}
		try (FileOutputStream out = new FileOutputStream(imgpath)) {
			ImageIO.write(img, "bmp", out);
		} catch (Exception ex) {
			ex.printStackTrace();
		}
	}

	static void generateTinyPic(String outbase) throws Exception {
		double[][] pics = new double[][] {
				{ 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
				{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
				{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255 } };

		for (int i = 0; i < pics.length; i++) {
			double[] pixel = normalize(pics[i]);
			Vector2BMP(pixel, outbase + "t" + i + ".bmp");
		}
	}

	public static String drawMultinomialLines(double[][] lines) {
		StringBuilder sb = new StringBuilder();
		int deno = 100;
		for (int i = 0; i < lines.length; i++) {
			List<Double> data = Arrays.asList(ArrayUtils.toObject(lines[i]));
			Double total = data.stream().reduce(0d, (x, y) -> x + y);
			for (int j = 0; j < lines[i].length; j++) {
				int p = Double.valueOf(lines[i][j] / total * deno).intValue();
				char sym = (j % 2 == 0) ? '+' : '-';
				for (int pos = 0; pos < p; pos++) {
					sb.append(sym);
				}
				if (p < 1) {
					sb.append(".");
				}
			}
			sb.append("\n");
		}
		return sb.toString();
	}

	public static void main(String[] args) throws Exception {
		String outbase = "/Users/caowenjiong/Movies/tmp/";
		generateTinyPic(outbase);

		// String base = "src/main/resources/constellation/";
		// double[] v = BMP2Vector(base + "5.bmp");
		// System.out.println(Arrays.toString(v));
		// Vector2BMP(v, outbase + "9.png");
	}

}
