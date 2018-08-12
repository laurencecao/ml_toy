package dl.nn2.activation;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.util.FastMath;

import dl.nn2.graph.MatrixDataEdge;

public class Softmax implements GateFunction {

	@Override
	public RealMatrix forward(RealMatrix m0) {
		RealMatrix ret = m0.copy();
		for (int pos = 0; pos < ret.getColumnDimension(); pos++) {
			RealVector v = ret.getColumnVector(pos);
			double m = v.getMaxValue();
			double deno = 0d;
			double ele = 0d;
			for (int i = 0; i < v.getDimension(); i++) {
				ele = FastMath.exp(v.getEntry(i) - m);
				v.setEntry(i, ele);
				deno += ele;
			}
			ret.setColumnVector(pos, v.mapDivide(deno));
		}
		return ret;
	}

	@Override
	public RealMatrix backward(RealMatrix m) {
		RealMatrix ret = MatrixUtils.createRealMatrix(m.getRowDimension(), m.getRowDimension());
		double sz = m.getColumnDimension();
		for (int i = 0; i < sz; i++) {
			RealVector s = m.getColumnVector(i);
			RealMatrix iden = MatrixUtils.createRealIdentityMatrix(s.getDimension());
			for (int j = 0; j < iden.getRowDimension(); j++) {
				RealVector v = iden.getRowVector(j).subtract(s);
				double _v = s.getEntry(j);
				v = v.mapMultiply(_v);
				iden.setRowVector(j, v);
			}
			ret = ret.add(iden);
		}
		return ret;
	}

	@Override
	public String name() {
		return Softmax.class.getSimpleName();
	}

	public static void main(String[] args) {
		RealMatrix d = MatrixUtils.createRealMatrix(new double[][] { { 0.09003057d, 0.24472847d, 0.66524096d } })
				.transpose();
		d = MatrixUtils.createRealMatrix(new double[][] { { 1, 2, 3 } }).transpose();
		Softmax h = new Softmax();
		RealMatrix y = h.forward(d);
		System.out.println(y);
		RealMatrix dx = null;
		dx = h.backward(y);
		System.out.println(MatrixDataEdge.pretty0(dx));

		y = MatrixUtils.createRealMatrix(new double[][] { { 0.2d, 0.8d } }).transpose();
		System.out.println(y);
		dx = h.backward(y);
		System.out.println(MatrixDataEdge.pretty0(dx));

	}

}
