package dl.nn2.graph;

import org.apache.commons.math3.linear.RealMatrix;

import dl.nn2.activation.GateFunction;

public class GateOp extends TracedComputation {

	protected GateFunction gate;
	protected boolean forward;
	protected String memo;

	public GateOp(GateFunction g, boolean forward, String memo) {
		this.gate = g;
		this.forward = forward;
		this.memo = memo;
	}

	@Override
	protected MatrixDataEdge eval0(MatrixDataEdge data) {
		RealMatrix r;
		if (forward) {
			r = gate.forward(data.asMat(0));
		} else {
			r = gate.backward(data.asMat(0));
		}
		return new MatrixDataEdge(type() + "_" + name() + "_" + id(), r);
	}

	@Override
	public String type() {
		return super.type() + "{" + name() + (forward ? "" : "'") + "} " + "#" + memo + "#";
	}

	@Override
	public String name() {
		return gate.name();
	}

	@Override
	public int[] inShape() {
		return new int[] { -1, -1 };
	}

	@Override
	public int[] outShape() {
		return new int[] { -1, -1 };
	}

}
