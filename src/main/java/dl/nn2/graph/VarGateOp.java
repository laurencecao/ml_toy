package dl.nn2.graph;

import dl.nn2.activation.GateFunction;

public class VarGateOp extends GateOp {

	protected MatrixDataEdge var;

	public VarGateOp(GateFunction g, boolean forward, String memo) {
		this(g, forward, false, memo, null);
	}

	public VarGateOp(GateFunction g, boolean forward, String memo, MatrixDataEdge data) {
		this(g, forward, false, memo, data);
	}

	public VarGateOp(GateFunction g, boolean forward, boolean multiChannel, String memo, MatrixDataEdge data) {
		super(g, forward, multiChannel, memo);
		this.var = data;
	}

	public MatrixDataEdge eval(MatrixDataEdge data, String rtMsg) {
		MatrixDataEdge ret = null;
		try {
			ret = eval0(var);
		} catch (Exception e) {
			throw e;
		} finally {
			trace(rtMsg, var, ret);
		}
		return ret;
	}

}
