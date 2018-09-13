package dl.nn2.graph;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import dl.nn2.init.Xavier;

/**
 * only convolve 2D supported for simplicy, no stride or dilate supported
 * 
 * @author Laurence Cao
 * @date 2018年9月9日
 *
 */
public class ConvOp extends TracedComputation {

	protected Xavier initiator = new Xavier();

	protected int kernel;
	protected int channel;
	protected int group;
	protected int[] inShape;
	protected int[] outShape;
	protected boolean rotate;

	protected Filter filter;

	public ConvOp(int kernel, int channel, int group, int[] inShape, int[] outShape, boolean rotate) {
		this.rotate = rotate;
		this.outShape = outShape;
		this.inShape = inShape;
		this.kernel = kernel;
		this.channel = channel;
		this.group = group;
		this.filter = new Filter(this.kernel, this.channel, this.group, initiator);
	}

	public ConvOp rotate() {
		ConvOp ret = new ConvOp(this.kernel, this.channel, this.group, this.inShape, this.outShape, !rotate);
		ret.filter = this.filter;
		return ret;
	}

	@Override
	public String name() {
		return ConvOp.class.getName() + filter;
	}

	@Override
	public int[] inShape() {
		return inShape;
	}

	@Override
	public int[] outShape() {
		return outShape;
	}

	@Override
	protected MatrixDataEdge eval0(MatrixDataEdge data) {
		// 这里比较难看，初期没打算写CNN，所以未考虑multi channel，但是简便起见demo无所谓
		MatrixDataEdge ret = new MatrixDataEdge("convDD", null);
		int r, c;
		if (!rotate) {
			r = outShape[0];
			c = outShape[1];
		} else {
			r = inShape[0];
			c = inShape[1];
		}
		for (int i = 0; i < filter.group(); i++) {
			MatrixDataEdge rt = new MatrixDataEdge("convRet", MatrixUtils.createRealMatrix(r, c));
			Kernel[] ks = filter.getKernelGroup(i);
			for (int j = 0; j < ks.length; j++) {
				if (!rotate) {
					ks[j].conv(1, data, rt);
				} else {
					ks[j].rotateConv(1, data, rt);
				}
			}
			ret.addToMatList(rt.asMat(0));
		}
		return ret;
	}

}

class Kernel {

	final static Function<Integer[], Integer> calcOutSize = x -> {
		Integer WIDTH = x[0];
		Integer KERNEL = x[1];
		Integer PADDING = x[2];
		Integer STRIDE = x[3];
		return (WIDTH - KERNEL + 2 * PADDING) / STRIDE + 1;
	};

	final static Function<Integer[], Integer> calcFromOutSize = x -> {
		Integer WIDTH = x[0];
		Integer KERNEL = x[1];
		Integer STRIDE = x[2];
		Integer O_WIDTH = x[3];
		// note: total paddings
		return ((O_WIDTH - 1) * STRIDE + KERNEL - WIDTH);
	};

	final String name;
	final int F;
	final MatrixDataEdge w;

	Kernel(int F, Xavier initial, String name) {
		this.name = name;
		this.F = F;
		this.w = new MatrixDataEdge(name, initial.initWeights(F, F));
	}

	public MatrixDataEdge getKernel() {
		return this.w;
	}

	public void updateKernel(MatrixDataEdge wt) {
		this.w.update(wt);
	}

	void conv(RealMatrix W, int stride, MatrixDataEdge data, MatrixDataEdge output) {
		int[] pad = calcFromOutput(stride, data.shape(), output.shape());
		RealMatrix d0 = data.asMat(0);
		if (pad[0] > 0) { // row
			int a = pad[0] / 2;
			RealMatrix d = MatrixUtils.createRealMatrix(pad[0] + d0.getRowDimension(), d0.getColumnDimension());
			d.setSubMatrix(d0.getData(), a, 0);
			d0 = d;
		}
		if (pad[1] > 0) { // col
			int a = pad[1] / 2;
			RealMatrix d = MatrixUtils.createRealMatrix(d0.getRowDimension(), pad[1] + d0.getColumnDimension());
			d.setSubMatrix(d0.getData(), 0, a);
			d0 = d;
		}

		RealMatrix ret = doConv2d(d0, W);
		ret = output.asMat(0).add(ret);
		output.update(ret);
	}

	void reverse(double[] array) {
		if (array == null) {
			return;
		}
		int i = 0;
		int j = array.length - 1;
		double tmp;
		while (j > i) {
			tmp = array[j];
			array[j] = array[i];
			array[i] = tmp;
			j--;
			i++;
		}
	}

	void rotate180(RealMatrix w) {
		for (int i = 0; i < w.getRowDimension(); i++) {
			double[] dd = w.getRow(i);
			reverse(dd);
			w.setRow(i, dd);
		}
		for (int i = 0; i < w.getColumnDimension(); i++) {
			double[] dd = w.getColumn(i);
			reverse(dd);
			w.setColumn(i, dd);
		}
	}

	public void rotateConv(int stride, MatrixDataEdge data, MatrixDataEdge output) {
		RealMatrix w = this.w.asMat(0).copy();
		rotate180(w);
		conv(w, stride, data, output);
	}

	public void conv(int stride, MatrixDataEdge data, MatrixDataEdge output) {
		conv(this.w.asMat(0), stride, data, output);
	}

	int[] calcFromOutput(int stride, int[] inShape, int[] outShape) {
		int[] ret = new int[] { calcFromOutSize.apply(new Integer[] { inShape[0], F, stride, outShape[0] }),
				calcFromOutSize.apply(new Integer[] { inShape[1], F, stride, outShape[0] }), };
		return ret;
	}

	RealMatrix doConv2d(RealMatrix data, RealMatrix knl) {
		int dW = data.getColumnDimension();
		int dH = data.getRowDimension();
		int kW = knl.getColumnDimension();
		int kH = knl.getRowDimension();
		Integer col = calcOutSize.apply(new Integer[] { dW, kW, 0, 1 });
		Integer row = calcOutSize.apply(new Integer[] { dH, kH, 0, 1 });
		RealMatrix ret = MatrixUtils.createRealMatrix(row, col);
		for (int i = 0; i < dH - kH + 1; i++) {
			for (int j = 0; j < dW - kW + 1; j++) {
				RealMatrix d0 = data.getSubMatrix(i, i + kH - 1, j, j + kW - 1);
				double r = 0d;
				for (int x = 0; x < d0.getRowDimension(); x++) {
					r += d0.getRowVector(x).dotProduct(knl.getRowVector(x));
				}
				ret.setEntry(i, j, r);
			}
		}
		return ret;
	}
}

class Filter {

	final int KERNEL;
	final int CHANNEL;
	final int K;
	final List<Kernel[]> filters;

	Filter(int kernel, int Channel, int K, Xavier initial) {
		this.KERNEL = kernel;
		this.CHANNEL = Channel;
		this.K = K;
		this.filters = new ArrayList<>();
		for (int i = 0; i < this.K; i++) {
			Kernel[] k = new Kernel[this.CHANNEL];
			for (int j = 0; j < k.length; j++) {
				k[j] = new Kernel(this.KERNEL, initial, "Kernel[" + i + "," + j + "]");
			}
			filters.add(k);
		}
	}

	public Kernel[] getKernelGroup(int idx) {
		return filters.get(idx);
	}

	public int group() {
		return this.filters.size();
	}

	public String toString() {
		return Filter.class.getName() + "_" + CHANNEL + ">> _" + KERNEL + "X" + KERNEL + "_ >>" + K;
	}

}
