package dl.nn2.graph;

import org.apache.commons.math3.linear.RealMatrix;

public class ScalarOp extends TracedComputation {

	protected double scalar;
	protected char op;
	protected String memo;

	public ScalarOp(double scalar, char op, String memo) {
		this.scalar = scalar;
		this.op = op;
		this.memo = memo;
	}

	@Override
	public String name() {
		return "{" + op + scalar + "}" + "#" + memo + "#";
	}

	@Override
	public String type() {
		return super.type() + name();
	}

	@Override
	public int[] inShape() {
		return new int[] { -1, -1 };
	}

	@Override
	public int[] outShape() {
		return new int[] { -1, -1 };
	}

	@Override
	protected MatrixDataEdge eval0(MatrixDataEdge data) {
		RealMatrix d = data.asMat(0);
		RealMatrix ret = null;
		switch (op) {
		case '+':
			ret = d.scalarAdd(scalar);
			break;
		case '-':
			ret = d.scalarAdd(-1d * scalar);
			break;
		case '*':
			ret = d.scalarMultiply(scalar);
			break;
		case '/':
			ret = d.scalarMultiply(1d / scalar);
			break;
		default:
			break;
		}
		return new MatrixDataEdge("scalar", ret);
	}

}
