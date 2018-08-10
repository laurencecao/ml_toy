package dl.nn2.optimizer;

import org.apache.commons.math3.linear.RealMatrix;

import dl.nn2.graph.MatrixDataEdge;

public class SimpleGradientDescend extends RMSPropOptimizer {

	public SimpleGradientDescend(double lr, double dr) {
		super(lr, dr);
	}

	protected MatrixDataEdge[] eval0(MatrixDataEdge deltaX) {
		RealMatrix m = deltaX.asMat(0).scalarMultiply(-1d * learningRate);
		MatrixDataEdge r = new MatrixDataEdge("gradW", m);
		return new MatrixDataEdge[] { r, null };
	}

}
