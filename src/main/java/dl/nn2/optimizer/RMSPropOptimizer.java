package dl.nn2.optimizer;

import java.util.HashMap;
import java.util.Map;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealMatrixChangingVisitor;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.util.FastMath;
import org.apache.log4j.Level;
import org.apache.log4j.LogManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import dl.nn2.graph.MatrixDataEdge;

/**
 * a naive version for RMSProp Optimizer
 * 
 * @see <a href=
 *      "http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture7.pdf">Lecture
 *      7: Training Neural Networks, Part 2</a>
 * 
 * @author Laurence Cao
 * @date 2018年8月4日
 *
 */
public class RMSPropOptimizer {

	protected Logger logger = LoggerFactory.getLogger(RMSPropOptimizer.class);
	static protected boolean enabled = false;

	final static double VERY_SMALL = 1e-8;

	protected double decayRate;
	protected double learningRate;

	protected Map<String, MatrixDataEdge> gradSquared;
	protected GradSquaredA opSquaredA;
	protected GradSquaredB opSquaredB;
	protected GradSquaredC opC;

	public RMSPropOptimizer(double lr, double dr) {
		this.learningRate = lr;
		this.decayRate = dr;
		this.gradSquared = new HashMap<String, MatrixDataEdge>();
		this.opSquaredA = new GradSquaredA(dr);
		this.opSquaredB = new GradSquaredB(dr);
		this.opC = new GradSquaredC();
	}

	protected void trace(MatrixDataEdge input, MatrixDataEdge output, MatrixDataEdge squared, String memo) {
		if (enabled && (logger.isTraceEnabled() || logger.isDebugEnabled())) {
			String name = RMSPropOptimizer.class.getSimpleName() + " ==> " + memo;
			if (logger.isTraceEnabled()) {
				String m = name + " : \ndx: \b" + pretty(input) + "\nsqaured: \n" + pretty(squared) + "\noutput: \n"
						+ pretty(output);
				logger.trace(m);
			} else if (logger.isDebugEnabled()) {
				String m = name + " : \ngradient output: " + pretty(output);
				logger.debug(m);
			}
		}
	}

	public static String pretty(MatrixDataEdge m0) {
		if (m0 == null) {
			return "\n";
		}
		RealMatrix m = m0.asMat(0);
		if (m == null) {
			return "\n";
		}
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < m.getRowDimension(); i++) {
			double[] items = m.getRow(i);
			for (int j = 0; j < items.length; j++) {
				sb.append(String.format("%10.3f", items[j])).append("  ");
			}
			sb.append('\n');
		}
		return sb.toString();
	}

	public MatrixDataEdge eval(MatrixDataEdge deltaX, String memo) {
		MatrixDataEdge[] ret = null;
		try {
			ret = eval0(deltaX);
			return ret[0];
		} catch (Exception e) {
			throw e;
		} finally {
			trace(deltaX, ret[0], ret[1], memo);
		}
	}

	protected MatrixDataEdge[] eval0(MatrixDataEdge deltaX) {
		String id = deltaX.getId();
		MatrixDataEdge grad = null;
		if (!gradSquared.containsKey(id)) {
			RealMatrix m = MatrixUtils.createRealMatrix(deltaX.asMat(0).getRowDimension(),
					deltaX.asMat(0).getColumnDimension());
			grad = new MatrixDataEdge(id, "squared", m);
			gradSquared.put(id, grad);
		}
		grad = gradSquared.get(id);

		RealMatrix dx_1 = grad.asMat(0);
		dx_1.walkInOptimizedOrder(opSquaredA);

		RealMatrix dx_2 = deltaX.asMat(0).copy();
		dx_2.walkInOptimizedOrder(opSquaredB);

		grad.update(dx_1.add(dx_2));

		RealMatrix ret = deltaX.asMat(0).scalarMultiply(-1d * learningRate);
		RealMatrix sq = grad.asMat(0).copy();
		sq.walkInOptimizedOrder(opC);
		for (int i = 0; i < ret.getColumnDimension(); i++) {
			RealVector vEle = ret.getColumnVector(i);
			RealVector vDeno = sq.getColumnVector(i);
			ret.setColumnVector(i, vEle.ebeDivide(vDeno));
		}
		MatrixDataEdge r = new MatrixDataEdge("gradW", ret);
		return new MatrixDataEdge[] { r, grad };
	}

	static class GradSquaredA implements RealMatrixChangingVisitor {

		protected double dr;

		GradSquaredA(double decayRate) {
			dr = decayRate;
		}

		@Override
		public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {
		}

		@Override
		public double visit(int row, int column, double value) {
			return dr * value;
		}

		@Override
		public double end() {
			return 0;
		}

	};

	static class GradSquaredB implements RealMatrixChangingVisitor {

		protected double dr;

		GradSquaredB(double decayRate) {
			dr = decayRate;
		}

		@Override
		public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {
		}

		@Override
		public double visit(int row, int column, double value) {
			return (1 - dr) * value * value;
		}

		@Override
		public double end() {
			return 0;
		}

	};

	static class GradSquaredC implements RealMatrixChangingVisitor {

		@Override
		public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {
		}

		@Override
		public double visit(int row, int column, double value) {
			return FastMath.sqrt(value) + VERY_SMALL;
		}

		@Override
		public double end() {
			return 0;
		}

	};

	public static void debugLevel(int level) {
		switch (level) {
		case 1:
			enabled = true;
			LogManager.getLogger(RMSPropOptimizer.class).setLevel(Level.DEBUG);
			break;
		case 2:
			enabled = true;
			LogManager.getLogger(RMSPropOptimizer.class).setLevel(Level.TRACE);
			break;
		default:
			enabled = false;
		}
	}

	public static void main(String[] args) {
		RMSPropOptimizer opt = new RMSPropOptimizer(1e-4, 0.95d);
		MatrixDataEdge deltaX = new MatrixDataEdge("", MatrixUtils.createRealMatrix(new double[][] { { 1 } }));
		for (int i = 0; i < 10; i++) {
			MatrixDataEdge ret = opt.eval(deltaX, "");
			String msg = String.format("%10.8f", ret.asDouble(0));
			System.out.println(msg);
		}
		/**
		 * <pre>
		# 0 -0.000447213550779
		# 1 -0.000320256291187
		# 2 -0.000264790351464
		# 3 -0.000232185635364
		# 4 -0.000210249703143
		# 5 -0.00019429085759
		# 6 -0.000182070331555
		# 7 -0.000172367792128
		# 8 -0.000164454422172
		# 9 -0.000157864836382
		 * </pre>
		 */
	}
}
