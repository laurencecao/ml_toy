package dl.nn2.layer;

import org.apache.commons.lang3.tuple.Pair;

import dl.nn2.activation.GateFunction;
import dl.nn2.activation.MaxPooling;
import dl.nn2.graph.Computation;
import dl.nn2.graph.GateOp;
import dl.nn2.graph.GroupComputation;
import dl.nn2.graph.MulVarOp;
import dl.nn2.graph.Reshape;
import dl.nn2.graph.VarGateOp;
import dl.nn2.graph.VarOp;

public class PoolingLayer extends AbstractCompGraphLayer {

	protected int kernel;
	protected MaxPooling pool;
	protected int[] reshape;

	public PoolingLayer(int kernel, int[] reshape, String name) {
		this.kernel = kernel;
		this.pool = new MaxPooling(kernel);
		this.reshape = reshape;
		this.name = name;
	}

	@Override
	protected String typeName() {
		return PoolingLayer.class.getName();
	}

	@Override
	protected GateFunction getActivationFunction() {
		return gate;
	}

	@Override
	protected Computation getErrorBackWeight() {
		return super.getErrorBackWeight();
	}

	@Override
	public Pair<GroupComputation, GroupComputation> build() {
		VarOp var = new VarOp(name + "_Z", name + "_ff_saveZ");
		GroupComputation ff = new GroupComputation(name + "_FF", var, new GateOp(pool, true, name),
				new Reshape(reshape));
		ff.setAttach(this);

		AbstractCompGraphLayer nl = this.getNextLayer();
		Computation backW = nl.getErrorBackWeight();
		MulVarOp dGate = new MulVarOp(new VarGateOp(getActivationFunction(), false, "ff_bp", var.getVar()), false,
				"dLdz_mul_dzdy");
		GroupComputation bp = new GroupComputation(name, backW, new Reshape(reshape, true), dGate);
		bp.setAttach(this);
		return Pair.of(ff, bp);
	}

}
