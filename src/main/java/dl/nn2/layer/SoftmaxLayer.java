package dl.nn2.layer;

import org.apache.commons.lang3.tuple.Pair;

import dl.nn2.activation.GateFunction;
import dl.nn2.activation.Softmax;
import dl.nn2.graph.BiasedOp;
import dl.nn2.graph.GateOp;
import dl.nn2.graph.GroupComputation;
import dl.nn2.graph.MulOp;
import dl.nn2.graph.MulVarOp;
import dl.nn2.graph.VarGateOp;
import dl.nn2.graph.VarOp;

public class SoftmaxLayer extends AbstractCompGraphLayer {

	protected GateFunction getActivationFunction() {
		return new Softmax();
	}

	public SoftmaxLayer(int in, int out, String name) {
		super(in, out, name);
	}

	@Override
	protected String typeName() {
		return SoftmaxLayer.class.getSimpleName();
	}

	public Pair<GroupComputation, GroupComputation> build() {
		// default dense layer
		VarOp var = new VarOp(name + "_Z", name + "_ff_saveZ");
		GroupComputation ff = new GroupComputation(name + "_FF", new BiasedOp(true, name),
				new MulOp(w, false, false, false, name + "_ff"), new GateOp(getActivationFunction(), true, name), var);
		ff.setAttach(this);

		MulVarOp dLdz_mul_dzdy = new MulVarOp(new VarGateOp(getActivationFunction(), false, "ff_bp", var.getVar()),
				false, "dLdz_mul_dzdy");
		// dL/dy * dy/dz
		GroupComputation bp = new GroupComputation(name + "_BP", dLdz_mul_dzdy);
		bp.setAttach(this);
		return Pair.of(ff, bp);
	}

}
