package dl.nn2.layer;

import java.util.function.Function;

import org.apache.commons.lang3.tuple.Pair;

import dl.nn2.activation.GateFunction;
import dl.nn2.activation.Tanh;
import dl.nn2.graph.Computation;
import dl.nn2.graph.ConvOp;
import dl.nn2.graph.GateOp;
import dl.nn2.graph.GroupComputation;
import dl.nn2.graph.MulVarOp;
import dl.nn2.graph.Reshape;
import dl.nn2.graph.VarGateOp;
import dl.nn2.graph.VarOp;
import dl.nn2.init.Xavier;

/**
 * @see <a href="http://cs231n.github.io/convolutional-networks/"> CS231n
 *      Convolutional Neural Networks for Visual Recognition</a>
 * 
 * @author Laurence Cao
 * @date 2018年8月31日
 * 
 */
public class ConvolutionalLayer extends AbstractCompGraphLayer {

	static Function<Integer[], Integer> calcOutSize = x -> {
		Integer WIDTH = x[0];
		Integer KERNEL = x[1];
		Integer PADDING = x[2];
		Integer STRIDE = x[3];
		return (WIDTH - KERNEL + 2 * PADDING) / STRIDE + 1;
	};

	final protected Xavier initiator = new Xavier();

	protected int[] in;
	protected int[] out;

	protected int kernel; // kernel width and height
	protected int channel; // input matrices dimension
	protected int filterGroup; // output matrices count
	protected int stride; // stride step
	protected int padding; // padding size, for simplicy using valid
	protected int pooling; // max pooling size

	protected Reshape reshape;
	protected ConvOp conv;

	public ConvolutionalLayer(int[] inSize, int kernel, int channel, int filters, int stride, int padding,
			int[] reshape, String name) {
		this.in = inSize;
		this.out = new int[] { calcOutSize.apply(new Integer[] { inSize[0], kernel, padding, stride }),
				calcOutSize.apply(new Integer[] { inSize[1], kernel, padding, stride }) };
		this.kernel = kernel;
		this.channel = channel;
		this.filterGroup = filters;
		this.stride = stride;
		this.padding = padding;
		this.conv = new ConvOp(kernel, channel, filters, inSize, out, false);
		this.reshape = reshape == null ? new Reshape() : new Reshape(reshape);
		init(inSize[0] * inSize[1], out[0] * out[1], name);
	}

	@Override
	protected String typeName() {
		return ConvolutionalLayer.class.getName();
	}

	@Override
	protected GateFunction getActivationFunction() {
		return new Tanh();
	}

	@Override
	protected Computation getErrorBackWeight() {
		return this.conv.rotate();
	}

	@Override
	public Pair<GroupComputation, GroupComputation> build() {
		// default dense layer
		VarOp var = new VarOp(name + "_Z", name + "_ff_saveZ");
		GateOp tanh = new GateOp(getActivationFunction(), true, name);
		GroupComputation ff = new GroupComputation(name + "_FF", conv, var, tanh);
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
			Computation backW = nl.getErrorBackWeight();
			MulVarOp d_tanh = new MulVarOp(new VarGateOp(getActivationFunction(), false, "ff_bp", var.getVar()), false,
					"dLdz_mul_dzdy");
			bp = new GroupComputation(name, backW, d_tanh);
		}
		bp.setAttach(this);
		return Pair.of(ff, bp);
	}

}
