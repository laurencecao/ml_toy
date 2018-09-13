package dl.nn2.layer;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

import org.apache.commons.math3.util.FastMath;

import dl.nn2.activation.GateFunction;
import dl.nn2.graph.MatrixDataEdge;
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

//	protected Filter filter;

	public ConvolutionalLayer(int[] inSize, int kernel, int channel, int filters, int stride, int padding,
			String name) {
		this.in = inSize;
		this.out = new int[] { calcOutSize.apply(new Integer[] { inSize[0], kernel, padding, stride }),
				calcOutSize.apply(new Integer[] { inSize[1], kernel, padding, stride }) };
		this.kernel = kernel;
		this.channel = channel;
		this.filterGroup = filters;
		this.stride = stride;
		this.padding = padding;
//		this.filter = new Filter(this.kernel, this.channel, this.filterGroup, initiator);
		init(inSize[0] * inSize[1], out[0] * out[1], name);
	}

	@Override
	protected String typeName() {
		return ConvolutionalLayer.class.getName();
	}

	@Override
	protected GateFunction getActivationFunction() {
		return null;
	}

	public static int calcPaddingSize(int inW, int outW, int kernel, int stride) {
		double r = ((outW - 1) * stride + kernel - inW) * 1.0d / 2;
		int ret = Double.valueOf(FastMath.ceil(r)).intValue();
		return FastMath.max(ret, 0);
	}

	public static void main(String[] args) {
		int r = calcPaddingSize(3, 4, 3, 1);
		System.out.println(r);
	}

}

