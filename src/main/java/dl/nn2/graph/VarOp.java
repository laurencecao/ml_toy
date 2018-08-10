package dl.nn2.graph;

public class VarOp extends TracedComputation {

	protected String varName;
	protected String memo;

	protected Refresher observer;

	public VarOp(String varName, String memo) {
		this.varName = varName;
		this.memo = memo;
		this.observer = new Refresher();
	}

	@Override
	public String type() {
		return super.type() + "#" + memo + "#";
	}

	@Override
	public String name() {
		return varName;
	}

	@Override
	public int[] inShape() {
		return new int[] { -1, -1 };
	}

	@Override
	public int[] outShape() {
		return new int[] { -1, -1 };
	}

	@Override
	protected MatrixDataEdge eval0(MatrixDataEdge data) {
		observer.writeVar(data);
		return data;
	}

	public MatrixDataEdge getVar() {
		MatrixDataEdge ret = new MatrixDataEdge("bridge", null);
		ret.setUpdater(observer);
		return ret;
	}

}
