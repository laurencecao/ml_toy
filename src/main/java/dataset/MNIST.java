package dataset;

import static java.lang.String.format;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.zip.GZIPInputStream;

import org.apache.commons.io.IOUtils;

import utils.SwingDisplayer;

public class MNIST {

	final static String path = "src/main/resources/";

	public static final int LABEL_FILE_MAGIC_NUMBER = 2049;
	public static final int IMAGE_FILE_MAGIC_NUMBER = 2051;

	protected int label; // label for image
	protected int[][] image; // matrix for image

	public int getLabel() {
		return label;
	}

	public int[][] getImage() {
		return image;
	}

	public static List<MNIST> getTraining() {
		return getObjects(path + "train-images-idx3-ubyte.gz", path + "train-labels-idx1-ubyte.gz");
	}

	public static List<MNIST> getTesting() {
		return getObjects(path + "t10k-images-idx3-ubyte.gz", path + "t10k-labels-idx1-ubyte.gz");
	}

	static List<MNIST> getObjects(String imgPath, String labelPath) {
		List<MNIST> ret = new ArrayList<>();
		List<int[][]> data = getImages(imgPath);
		int[] lb = getLabels(labelPath);
		for (int i = 0; i < data.size(); i++) {
			MNIST m = new MNIST();
			m.label = lb[i];
			m.image = data.get(i);
			ret.add(m);
		}
		return ret;
	}

	static int[] getLabels(String infile) {
		ByteBuffer bb = loadFileToByteBuffer(infile);
		assertMagicNumber(LABEL_FILE_MAGIC_NUMBER, bb.getInt());

		int numLabels = bb.getInt();
		int[] labels = new int[numLabels];

		for (int i = 0; i < numLabels; ++i)
			labels[i] = bb.get() & 0xFF; // To unsigned

		return labels;
	}

	static List<int[][]> getImages(String infile) {
		ByteBuffer bb = loadFileToByteBuffer(infile);

		assertMagicNumber(IMAGE_FILE_MAGIC_NUMBER, bb.getInt());

		int numImages = bb.getInt();
		int numRows = bb.getInt();
		int numColumns = bb.getInt();
		List<int[][]> images = new ArrayList<>();

		for (int i = 0; i < numImages; i++)
			images.add(readImage(numRows, numColumns, bb));

		return images;
	}

	private static int[][] readImage(int numRows, int numCols, ByteBuffer bb) {
		int[][] image = new int[numRows][];
		for (int row = 0; row < numRows; row++)
			image[row] = readRow(numCols, bb);
		return image;
	}

	private static int[] readRow(int numCols, ByteBuffer bb) {
		int[] row = new int[numCols];
		for (int col = 0; col < numCols; ++col)
			row[col] = bb.get() & 0xFF; // To unsigned
		return row;
	}

	public static void assertMagicNumber(int expectedMagicNumber, int magicNumber) {
		if (expectedMagicNumber != magicNumber) {
			switch (expectedMagicNumber) {
			case LABEL_FILE_MAGIC_NUMBER:
				throw new RuntimeException("This is not a label file.");
			case IMAGE_FILE_MAGIC_NUMBER:
				throw new RuntimeException("This is not an image file.");
			default:
				throw new RuntimeException(
						format("Expected magic number %d, found %d", expectedMagicNumber, magicNumber));
			}
		}
	}

	/*******
	 * Just very ugly utilities below here. Best not to subject yourself to them.
	 * ;-)
	 ******/

	public static ByteBuffer loadFileToByteBuffer(String infile) {
		return ByteBuffer.wrap(loadFile(infile));
	}

	public static byte[] loadFile(String infile) {
		try (RandomAccessFile f = new RandomAccessFile(infile, "r");) {
			FileChannel chan = f.getChannel();
			long fileSize = chan.size();
			ByteBuffer bb = ByteBuffer.allocate((int) fileSize);
			chan.read(bb);
			bb.flip();
			return flat(bb);
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	static byte[] flat(ByteBuffer data) throws IOException {
		byte[] compressed = new byte[data.remaining()];
		data.get(compressed);
		ByteArrayInputStream bis = new ByteArrayInputStream(compressed);
		GZIPInputStream gis = new GZIPInputStream(bis);
		return IOUtils.toByteArray(gis);
	}

	public static String renderImage(int[][] image) {
		StringBuffer sb = new StringBuffer();

		for (int row = 0; row < image.length; row++) {
			sb.append("|");
			for (int col = 0; col < image[row].length; col++) {
				int pixelVal = image[row][col];
				if (pixelVal == 0)
					sb.append(" ");
				else if (pixelVal < 256 / 3)
					sb.append(".");
				else if (pixelVal < 2 * (256 / 3))
					sb.append("x");
				else
					sb.append("X");
			}
			sb.append("|\n");
		}

		return sb.toString();
	}

	public static String repeat(String s, int n) {
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < n; i++)
			sb.append(s);
		return sb.toString();
	}

	public static void main(String[] args) throws InterruptedException {
		List<MNIST> tr = MNIST.getTraining();
		System.out.println(tr.size());
		List<MNIST> te = MNIST.getTesting();
		System.out.println(te.size());

		int idx;
		List<MNIST> t;
		List<int[][]> data = new ArrayList<>();
		t = tr;
		for (int i = 0; i < 8; i++) {
			idx = ThreadLocalRandom.current().nextInt(t.size());
			data.add(t.get(idx).image);
		}
		t = te;
		for (int i = 0; i < 8; i++) {
			idx = ThreadLocalRandom.current().nextInt(t.size());
			data.add(t.get(idx).image);
		}

		SwingDisplayer.displayImages2(data, 4, 32);
		System.exit(0);
	}

}
