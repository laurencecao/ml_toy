package dl.nn2.graph;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.linear.RealMatrix;

import dl.nn2.activation.GateFunction;

public class GateOp extends TracedComputation {

	protected GateFunction gate;
	protected boolean forward;
	protected String memo;
	protected boolean multiChannel;

	public GateOp(GateFunction g, boolean forward, String memo) {
		this(g, forward, false, memo);
	}

	public GateOp(GateFunction g, boolean forward, boolean multiChannel, String memo) {
		this.gate = g;
		this.forward = forward;
		this.memo = memo;
		this.multiChannel = multiChannel;
	}

	@Override
	protected MatrixDataEdge eval0(MatrixDataEdge data) {
		List<RealMatrix> source;
		if (multiChannel) {
			source = data.asMatList();
		} else {
			source = new ArrayList<>();
			source.add(data.asMat(0));
		}

		MatrixDataEdge ret = new MatrixDataEdge(type() + "_" + name() + "_" + id());
		RealMatrix r;
		for (RealMatrix m : source) {
			if (forward) {
				r = gate.forward(m);
			} else {
				r = gate.backward(m);
			}
			ret.addToMatList(r);
		}
		return ret;
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
