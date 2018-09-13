package dl.nn2.layer;

import java.util.Arrays;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.math3.linear.RealMatrix;

import dl.nn2.activation.GateFunction;
import dl.nn2.graph.BiasedOp;
import dl.nn2.graph.GateOp;
import dl.nn2.graph.GroupComputation;
import dl.nn2.graph.MatrixDataEdge;
import dl.nn2.graph.MulOp;
import dl.nn2.graph.MulVarOp;
import dl.nn2.graph.VarGateOp;
import dl.nn2.graph.VarOp;
import dl.nn2.init.Xavier;

public abstract class AbstractCompGraphLayer {

	protected MatrixDataEdge w;
	protected GateFunction gate;

	protected Xavier xinit = new Xavier();
	protected AbstractCompGraphLayer next;
	protected AbstractCompGraphLayer pre;
	protected int inSize;
	protected int outSize;
	protected String name;
	protected double biased = -1d;

	abstract protected String typeName();

	abstract protected GateFunction getActivationFunction();

	protected void init(int in, int out, String name) {
		this.inSize = in;
		this.outSize = out;
		this.name = name;
		RealMatrix w0 = xinit.initWeights(inSize + 1, outSize);
		w = new MatrixDataEdge(name + "_weights", name + "_weights_" + (inSize + 1) + "x" + outSize, w0);
	}

	public int getIn() {
		return inSize;
	}

	public int getOut() {
		return outSize;
	}

	public void setBiased(double biased) {
		this.biased = biased;
	}

	public String getName() {
		return name + "@" + typeName();
	}

	public Pair<GroupComputation, GroupComputation> build() {
		// default dense layer
		VarOp var = new VarOp(name + "_Z", name + "_ff_saveZ");
		GroupComputation ff = new GroupComputation(name + "_FF", new BiasedOp(true, name),
				new MulOp(w, false, false, false, name + "_ff"), var, new GateOp(getActivationFunction(), true, name));
		ff.setAttach(this);

		MulVarOp dLdz_mul_dzdy = null;
		GroupComputation bp = null;
		AbstractCompGraphLayer nl = this.getNextLayer();
		if (nl == null) {
			dLdz_mul_dzdy = new MulVarOp(new VarGateOp(getActivationFunction(), false, "ff_bp", var.getVar()), false,
					"dLdz_mul_dzdy");
			// dL/dy * dy/dz
			bp = new GroupComputation(name + "_BP", dLdz_mul_dzdy);
		} else {
			// dL/dZ * dZ/dy * dy/dz {layer(Z) == layer(y) + 1}
			bp = new GroupComputation(name, new MulOp(nl.w, true, false, false, name + "_bp"),
					new BiasedOp(false, name),
					new MulVarOp(new VarGateOp(getActivationFunction(), false, "ff_bp", var.getVar()), false,
							"dLdz_mul_dzdy"));
		}
		bp.setAttach(this);
		return Pair.of(ff, bp);
	}

	public void updateWeights(MatrixDataEdge nW) {
		if (biased >= 0) {
			RealMatrix dw = nW.asMat(0).copy();
			double[] dd = new double[dw.getColumnVector(0).getDimension()];
			Arrays.fill(dd, biased);
			dw.setColumn(0, dd);
			nW = new MatrixDataEdge("updateWeights", dw);
		}
		w.update(nW);
	}

	public MatrixDataEdge getWeights() {
		return w;
	}

	public AbstractCompGraphLayer getNextLayer() {
		return next;
	}

	public AbstractCompGraphLayer getPreLayer() {
		return pre;
	}

	public void setNextLayer(AbstractCompGraphLayer next) {
		this.next = next;
		if (this.next.getPreLayer() != this) {
			next.setPreLayer(this);
		}
	}

	public void setPreLayer(AbstractCompGraphLayer pre) {
		this.pre = pre;
		if (this.pre.getNextLayer() != this) {
			pre.setNextLayer(this);
		}
	}

	public String debugInfo() {
		String a = "------------" + name + "--------------\n";
		String typeName = typeName();
		String preName = this.getPreLayer() == null ? "" : this.getPreLayer().getClass().getSimpleName();
		String nextName = this.getNextLayer() == null ? "" : this.getNextLayer().getClass().getSimpleName();
		String inOut = "";
		for (int i = 0; i < w.asMat(0).getRowDimension(); i++) {
			inOut += Arrays.toString(w.asMat(0).getData()[i]) + "\n";
		}

		String ret = a;
		ret += typeName + " <== " + preName + "\n";
		ret += typeName + " ==> " + nextName + "\n";
		ret += inOut;
		ret += a;
		return ret;
	}

}
