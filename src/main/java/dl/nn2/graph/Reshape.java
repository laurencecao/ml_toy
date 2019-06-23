package dl.nn2.graph;

import java.util.stream.DoubleStream;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

public class Reshape extends TracedComputation {

	protected int[] outShape;
	protected boolean dummy = false;
	protected boolean merge = false;
	protected boolean reverse = false;

	public Reshape() {
		// dummy reshape
		this(new int[] { -1, -1 }, false);
	}

	public Reshape(int[] out, boolean merge) {
		this(out, merge, false);
	}

	public Reshape(int[] out, boolean merge, boolean reverse) {
		if ((out == null) || (out[0] == out[1] && out[0] == -1)) {
			this.dummy = true;
		}
		this.outShape = out;
		this.reverse = reverse;
		this.merge = merge;
	}

	@Override
	public String name() {
		return Reshape.class.getName();
	}

	@Override
	public int[] inShape() {
		return new int[] { -1, -1 };
	}

	@Override
	public int[] outShape() {
		return outShape;
	}

	@Override
	protected MatrixDataEdge eval0(MatrixDataEdge data) {
		if (dummy) {
			return new MatrixDataEdge("no_reshape", data);
		}
		MatrixDataEdge ret = new MatrixDataEdge("reshape");
		MatrixDataEdge r;
		if (merge) {
			int rr = data.asMatList().size();
			int cc = data.asMat(0).getRowDimension() * data.asMat(0).getColumnDimension();
			RealMatrix dd = MatrixUtils.createRealMatrix(rr, cc);
			for (int z = 0; z < data.asMatList().size(); z++) {
				RealMatrix mm = data.asMatList().get(z);
				DoubleStream all = null;
				for (int i = 0; i < mm.getRowDimension(); i++) {
					DoubleStream tmp = DoubleStream.of(mm.getRow(i));
					all = all == null ? tmp : DoubleStream.concat(all, tmp);
				}
				dd.setRow(z, all.toArray());
			}
			data = new MatrixDataEdge("reshape_0", dd);
		}
		if (!reverse) {
			for (RealMatrix mm : data.asMatList()) {
				r = doCalc(data, new int[] { mm.getRowDimension(), mm.getColumnDimension() },
						new int[] { outShape[0], outShape[1] });
				ret.addToMatList(r.asMat(0));
			}
		} else {
			for (RealMatrix mm : data.asMatList()) {
				r = doCalc(data, new int[] { outShape[0], outShape[1] },
						new int[] { mm.getRowDimension(), mm.getColumnDimension() });
				ret.addToMatList(r.asMat(0));
			}
		}
		return ret;
	}

	protected MatrixDataEdge doCalc(MatrixDataEdge data, int[] in, int[] out) {
		RealMatrix m = data.asMat(0);
		int r = in[0];
		int c = in[1];
		int R = out[0];
		int C = out[1];
		if (R < 0) {
			double d = 1.0d * c * r / C;
			R = Double.valueOf(d).intValue();
		}
		if (C < 0) {
			double d = 1.0d * c * r / R;
			C = Double.valueOf(d).intValue();
		}

		if (C * R != c * r) {
			throw new IllegalArgumentException();
		}
		RealMatrix ret = MatrixUtils.createRealMatrix(R, C);
		int x = 0, y = 0;
		for (int i = 0; i < m.getRowDimension(); i++) {
			double[] d = m.getRow(i);
			for (int z = 0; z < d.length; z++) {
				ret.setEntry(x, y, d[z]);
				y++;
				if (y >= C) {
					x++;
					y = 0;
				}
			}
		}
		return new MatrixDataEdge("", ret);
	}

}
