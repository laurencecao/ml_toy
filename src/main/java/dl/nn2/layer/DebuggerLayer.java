package dl.nn2.layer;

import java.util.List;
import java.util.UUID;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.util.FastMath;

import dl.nn2.activation.GateFunction;
import dl.nn2.activation.Tanh;
import dl.nn2.graph.Computation;
import dl.nn2.graph.GroupComputation;
import dl.nn2.graph.MatrixDataEdge;
import dl.nn2.graph.MulOp;
import dl.nn2.graph.Reshape;
import dl.nn2.graph.TracedComputation;

public class DebuggerLayer extends AbstractCompGraphLayer {

	protected MatrixDataEdge output;
	protected MatrixDataEdge loss;
	protected Reshape reshape;
	protected Pair<GroupComputation, GroupComputation> comp;
	protected ConstantOp in;
	protected ConstantOp out;
	protected MatrixDataEdge weight;

	/**
	 * @param inShape WHC
	 * @param output
	 * @param loss
	 */
	public DebuggerLayer(int[] inShape, List<RealMatrix> output, List<RealMatrix> loss, Reshape reshape) {
		this.output = new MatrixDataEdge(UUID.randomUUID().toString(), "output", output);
		this.loss = new MatrixDataEdge(UUID.randomUUID().toString(), "loss", loss);
		this.reshape = reshape;
		this.comp = init();
	}

	Pair<GroupComputation, GroupComputation> init() {
		in = new ConstantOp(output);
		GroupComputation ff = new GroupComputation(name + "_FF", in);
		ff.setAttach(this);

		out = new ConstantOp(loss);
		MatrixDataEdge weight = new MatrixDataEdge("dummy weight");
		for (RealMatrix m : loss.asMatList()) {
			int sz = Double.valueOf(FastMath.min(m.getRowDimension(), m.getColumnDimension())).intValue();
			RealMatrix w = MatrixUtils.createRealMatrix(sz, sz);
			for (int i = 0; i < sz; i++) {
				w.setEntry(i, i, 1);
			}
			weight.addToMatList(w);
		}
		MulOp mul = new MulOp(weight, false, false, false, "");
		GroupComputation bp = new GroupComputation(name, mul, reshape);
		bp.setAttach(this);
		return Pair.of(ff, bp);
	}

	@Override
	protected String typeName() {
		return DebuggerLayer.class.getName();
	}

	@Override
	protected GateFunction getActivationFunction() {
		return new Tanh(); // not important
	}

	@Override
	public Pair<GroupComputation, GroupComputation> build() {
		return comp;
	}

	@Override
	protected Computation getErrorBackWeight() {
		return new ConstantOp(weight);
	}

}

class ConstantOp extends TracedComputation {

	protected MatrixDataEdge data;

	public ConstantOp(MatrixDataEdge data) {
		this.data = data;
	}

	@Override
	public String name() {
		return ConstantOp.class.getName();
	}

	@Override
	public int[] inShape() {
		RealMatrix d = data.asMat(0);
		return new int[] { d.getRowDimension(), d.getColumnDimension() };
	}

	@Override
	public int[] outShape() {
		return inShape();
	}

	@Override
	protected MatrixDataEdge eval0(MatrixDataEdge data) {
		return data;
	}

}
