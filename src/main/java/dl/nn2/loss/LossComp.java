package dl.nn2.loss;

import org.apache.commons.lang3.StringUtils;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import dl.nn2.graph.MatrixDataEdge;
import dl.nn2.graph.TracedComputation;

public abstract class LossComp extends TracedComputation {

	protected LossFunction loss;
	protected MatrixDataEdge target;
	protected boolean isDif;

	public LossComp(MatrixDataEdge target, boolean dif) {
		this.target = target;
		this.isDif = dif;
		loss = loss(target);
	}

	abstract protected LossFunction loss(MatrixDataEdge target);

	@Override
	public MatrixDataEdge eval0(MatrixDataEdge data) {
		RealMatrix t = target.asMat(0);
		RealMatrix y = data.asMat(0);
		if (!isDif) {
			double ret = loss.eval(t, y);
			return new MatrixDataEdge("loss", MatrixUtils.createRealMatrix(new double[][] { { ret } }));
		}
		RealMatrix ret = loss.dif(t, y);
		return new MatrixDataEdge("loss", ret);
	}

	@Override
	public String name() {
		return loss.name();
	}

	@Override
	public String id() {
		return toString();
	}

	@Override
	public int[] inShape() {
		return null;
	}

	@Override
	public int[] outShape() {
		return null;
	}

	public static LossComp create(String name, MatrixDataEdge t, boolean isDif) {
		if (StringUtils.compareIgnoreCase("crossentropy", name) == 0) {
			return new CrossEntropy(t, isDif);
		}
		if (StringUtils.compareIgnoreCase("mse", name) == 0) {
			return new MSEComp(t, isDif);
		}
		throw new IllegalArgumentException();
	}

}

class CrossEntropy extends LossComp {

	public CrossEntropy(MatrixDataEdge target, boolean dif) {
		super(target, dif);
	}

	@Override
	protected LossFunction loss(MatrixDataEdge target) {
		return new CategoricalCrossEntropy();
	}

}

class MSEComp extends LossComp {

	public MSEComp(MatrixDataEdge target, boolean dif) {
		super(target, dif);
	}

	@Override
	protected LossFunction loss(MatrixDataEdge target) {
		return new MSE();
	}

}
