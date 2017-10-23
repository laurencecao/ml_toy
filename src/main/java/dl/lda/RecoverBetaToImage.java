package dl.lda;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;

import org.apache.commons.math3.util.FastMath;

import utils.ImageUtils;

public class RecoverBetaToImage {

	public static void main(String[] args) throws IOException {
		String path = "/Users/caowenjiong/working/opensource/lda-c/lda-c-dist/aabbcc/final.beta";
		fromImage(path, "tmp");
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
					pixel[i] = pixel[i] * 1000;
					pixel[i] = pixel[i] > 255 ? 255 : pixel[i];
					pixel[i] = pixel[i] < 0 ? 0 : pixel[i];
				}
				ImageUtils.Vector2BMP(pixel, outbase + "/out" + n + ".bmp");
				n++;
			}
			br.close();
		}
	}

}
