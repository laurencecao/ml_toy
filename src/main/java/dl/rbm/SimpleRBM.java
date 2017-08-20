package dl.rbm;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.Random;

import org.apache.commons.io.FileUtils;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealMatrixChangingVisitor;
import org.apache.commons.math3.random.UncorrelatedRandomVectorGenerator;

public class SimpleRBM implements Serializable {

	private static final long serialVersionUID = 1L;

	static boolean SAVEIT = false;

	final static Random rng = new Random();
	transient UncorrelatedRandomVectorGenerator rnd = null;

	int x;
	int hidden;
	RealMatrix weights;
	int K = 1;

	public static SimpleRBM load(String path) {
		File f = new File(path);
		if (!f.exists()) {
			return null;
		}
		try (FileInputStream fis = new FileInputStream(f)) {
			ObjectInputStream ois = new ObjectInputStream(fis);
			SimpleRBM ret = (SimpleRBM) ois.readObject();
			return ret;
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	public static void save(SimpleRBM rbm, String path) throws IOException {
		if (!SAVEIT) {
			return;
		}
		File f = new File(path);
		if (f.exists()) {
			FileUtils.copyFile(f, new File(f.getAbsolutePath() + ".bak"));
		}
		try (FileOutputStream fis = new FileOutputStream(f)) {
			ObjectOutputStream ois = new ObjectOutputStream(fis);
			ois.writeObject(rbm);
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	public SimpleRBM(int x, int hidden) {
		this.x = x;
		this.hidden = hidden;
		this.weights = MatrixUtils.createRealMatrix(hidden + 1, x + 1);
		this.weights.walkInOptimizedOrder(initRnd);
	}

	public RealMatrix getWeights() {
		return this.weights;
	}

	final static RealMatrixChangingVisitor initRnd = new RealMatrixChangingVisitor() {

		@Override
		public double end() {
			return 0;
		}

		@Override
		public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {
		}

		@Override
		public double visit(int row, int column, double value) {
			return rng.nextDouble();
		}
	};
}
