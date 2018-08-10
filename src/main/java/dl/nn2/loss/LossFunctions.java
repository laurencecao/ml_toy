package dl.nn2.loss;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealMatrixChangingVisitor;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.RealVectorChangingVisitor;
import org.apache.commons.math3.util.FastMath;

/**
 * in practice these all should be interface and instances
 * 
 * @author Laurence Cao
 * @date 2018年8月2日
 *
 */
public class LossFunctions {

	public static class Sigmoid {
		public static RealMatrix activation(RealMatrix z) {
			RealMatrix ret = z.copy();
			ret.walkInOptimizedOrder(new RealMatrixChangingVisitor() {

				@Override
				public double visit(int row, int column, double value) {
					double a = FastMath.exp(-1d * value);
					return 1d / (1d + a);
				}

				@Override
				public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {
				}

				@Override
				public double end() {
					return 0;
				}
			});
			return ret;
		}

		public static RealMatrix derivation(RealMatrix x) {
			RealMatrix ret = x.copy();
			ret.walkInOptimizedOrder(new RealMatrixChangingVisitor() {

				@Override
				public double visit(int row, int column, double value) {
					double a = FastMath.exp(-1d * value);
					double b = 1d / (1d + a);
					return b * (1 - b);
				}

				@Override
				public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {
				}

				@Override
				public double end() {
					return 0;
				}
			});
			return ret;
		}
	}

	public static class Tanh {

		public static RealMatrix activation(RealMatrix z) {
			RealMatrix ret = z.copy();
			ret.walkInOptimizedOrder(new RealMatrixChangingVisitor() {

				@Override
				public double visit(int row, int column, double value) {
					double a = FastMath.exp(value);
					double b = FastMath.exp(-1d * value);
					return (a - b) / (a + b);
				}

				@Override
				public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {
				}

				@Override
				public double end() {
					return 0;
				}
			});
			return ret;
		}

		public static RealMatrix derivation(RealMatrix x) {
			RealMatrix ret = x.copy();
			ret.walkInOptimizedOrder(new RealMatrixChangingVisitor() {

				@Override
				public double visit(int row, int column, double value) {
					double a = FastMath.exp(value);
					double b = FastMath.exp(-1d * value);
					double tanh = (a - b) / (a + b);
					return 1 - tanh * tanh;
				}

				@Override
				public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {
				}

				@Override
				public double end() {
					return 0;
				}
			});
			return ret;
		}

	}

	public static class Softmax {

		public static RealMatrix activation(RealMatrix z) {
			RealMatrix ret = MatrixUtils.createRealMatrix(z.getRowDimension(), z.getColumnDimension());
			for (int pos = 0; pos < z.getColumnDimension(); pos++) {
				RealVector v = z.getColumnVector(pos);
				double m = v.getMaxValue();
				double deno = 0d;
				double ele = 0d;
				for (int i = 0; i < v.getDimension(); i++) {
					ele = FastMath.exp(v.getEntry(i) - m);
					v.setEntry(i, ele);
					deno += ele;
				}
				ret.setColumnVector(pos, v.mapDivide(deno));
			}
			return ret;
		}

	}

	public static class CrossEntropy {

		/**
		 * 
		 * @param a
		 *            after feedforward and activation
		 * @param label
		 * @return
		 */
		static public RealVector getDLDz(RealVector a, RealVector label) {
			return a.subtract(label);
		}

		static public RealMatrix getDLDz(RealMatrix a, RealMatrix label) {
			return a.subtract(label);
		}

		static public double[] getLogLoss(RealMatrix y, RealMatrix t) {
			double[] ret = new double[y.getColumnDimension()];
			for (int i = 0; i < ret.length; i++) {
				RealVector vY = y.getColumnVector(i);
				RealVector vT = t.getColumnVector(i);
				vY.walkInOptimizedOrder(new RealVectorChangingVisitor() {

					@Override
					public double visit(int index, double value) {
						return FastMath.log(value);
					}

					@Override
					public void start(int dimension, int start, int end) {
					}

					@Override
					public double end() {
						return 0;
					}
				});
				ret[i] = -1d * vT.dotProduct(vY);
			}
			return ret;
		}

	}

	public static class L2 {
		static public RealMatrix getDLDy(RealMatrix y, RealMatrix label) {
			return y.subtract(label);
		}
	}

}
