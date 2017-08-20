package dl.rbm;

import java.util.Random;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealMatrixChangingVisitor;
import org.apache.commons.math3.random.UncorrelatedRandomVectorGenerator;

public class SimpleRBM {

	final static Random rng = new Random();

	int x;
	int hidden;
	RealMatrix weights;
	UncorrelatedRandomVectorGenerator rnd;

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
