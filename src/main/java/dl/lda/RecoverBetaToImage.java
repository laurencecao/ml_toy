package dl.lda;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.util.FastMath;

import utils.ImageUtils;

public class RecoverBetaToImage {

	final static int D = 2000;
	final static int d = 100;
	final static int V = 25;

	public static void main(String[] args) throws IOException {
		String path = "/Users/caowenjiong/working/opensource/lda-c/lda-c-dist/aabbcc/final.beta";
		// fromImage(path, "/Users/caowenjiong/Movies/tmp/out");
		fromImage2(path, "/Users/caowenjiong/Movies/tmp/out");
	}

	static void fromImage(String path, String outbase) throws IOException {
		try (FileInputStream fis = new FileInputStream(path)) {
			BufferedReader br = new BufferedReader(new InputStreamReader(fis));
			String line;
			int n = 0;
			while ((line = br.readLine()) != null) {
				line = line.trim();
				String[] items = line.split(" ");
				double[] pixel = new double[items.length];
				for (int i = 0; i < items.length; i++) {
					pixel[i] = FastMath.exp(Double.valueOf(items[i]));
					pixel[i] = pixel[i] * D;
					pixel[i] = pixel[i] > 255 ? 255 : pixel[i];
					pixel[i] = pixel[i] < 0 ? 0 : pixel[i];
				}
				ImageUtils.Vector2BMP(pixel, outbase + "/out" + n + ".bmp");
				n++;
			}
			br.close();
		}
	}

	static void fromImage2(String path, String outbase) throws IOException {
		List<double[]> topics = new ArrayList<double[]>();
		try (FileInputStream fis = new FileInputStream(path)) {
			BufferedReader br = new BufferedReader(new InputStreamReader(fis));
			String line;
			while ((line = br.readLine()) != null) {
				line = line.trim();
				String[] items = line.split(" ");
				double[] pixel = new double[items.length];
				for (int i = 0; i < items.length; i++) {
					pixel[i] = FastMath.exp(Double.valueOf(items[i]));
					pixel[i] = pixel[i] * D;
				}
				topics.add(pixel);
			}
			br.close();
		}
		ToyLDAModel model = new ToyLDAModel(1, 1, topics.size(), V, D, 1, null);
		RealMatrix m = model.wordTopicCount;
		for (int i = 0; i < m.getColumnDimension(); i++) {
			m.setColumn(i, topics.get(i));
		}
		model.toImage(outbase);
	}

}
