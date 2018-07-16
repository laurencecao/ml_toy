package dl.kalman;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealMatrixChangingVisitor;
import org.apache.commons.math3.linear.RealVector;

/**
 * @see <a href=
 *      "https://www.cl.cam.ac.uk/~rmf25/papers/Understanding%20the%20Basis%20of%20the%20Kalman%20Filter.pdf">
 *      Understanding the Basis of the Kalman Filter Via a Simple and Intuitive
 *      Derivation</a>
 * @author Laurence Cao
 * @date 2018年7月16日
 *
 */
public class SimpleKalmanFilter {

	// estimation stage
	protected RealVector x; // state vector
	protected RealVector u; // control vector

	protected RealMatrix F; // state transition matrix
	protected RealMatrix B; // control apply matrix

	// update stage
	protected RealMatrix H; // transformation matrix

	// covariance matrix for kalman gain
	protected RealMatrix P; // current process covariance matrix

	public void estimate(RealMatrix Q) {
		x = (F.operate(x)).add(B.operate(u));
		P = F.multiply(P).multiply(F.transpose()).add(Q);
	}

	public void update(RealVector z, RealMatrix R) {
		RealMatrix K = P.multiply(H.transpose()).multiply((H.multiply(P).multiply(H.transpose()).add(R)).transpose());
		x = x.add(K.operate(z.subtract(H.operate(x))));
		P = P.subtract(K.multiply(H).multiply(P));
	}

	public SimpleKalmanFilter(RealVector x, RealVector u, RealMatrix F, RealMatrix B, RealMatrix H) {
		this.x = x; // initial input
		this.u = u; // control input
		this.F = F; // state transition
		this.B = B; // control pattern
		this.H = H; // state to observe transformation matrix
		int d = x.getDimension();
		this.P = MatrixUtils.createRealMatrix(d, d);
		this.P.walkInOptimizedOrder(new RealMatrixChangingVisitor() {

			@Override
			public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {
			}

			@Override
			public double visit(int row, int column, double value) {
				return Math.random();
			}

			@Override
			public double end() {
				return 0;
			}

		});
	}

	public RealVector getState() {
		return this.x;
	}

}
