package dl.nn2.graph;

import java.util.ArrayList;
import java.util.List;

public class GroupComputation extends TracedComputation {

	protected Computation[] comps;
	protected MatrixDataEdge[] output;
	protected String memo;

	protected Object attach;
	protected boolean reversed;

	public Object getAttach() {
		return attach;
	}

	public void setAttach(Object attach) {
		this.attach = attach;
	}

	public void setReversed(boolean reversed) {
		this.reversed = reversed;
	}

	public GroupComputation(Computation... data) {
		this("", data);
	}

	public GroupComputation(String memo, Computation... data) {
		this.memo = memo;
		this.comps = data;
		this.output = new MatrixDataEdge[this.comps.length];
	}

	@Override
	protected MatrixDataEdge eval0(MatrixDataEdge data) {
		MatrixDataEdge d0 = data;
		if (!reversed) {
			for (int i = 0; i < comps.length; i++) {
				d0 = comps[i].eval(d0);
				output[i] = d0;
			}
		} else {
			for (int i = comps.length - 1; i >= 0; i--) {
				d0 = comps[i].eval(d0);
				output[i] = d0;
			}
		}
		return d0;
	}

	public MatrixDataEdge eval(MatrixDataEdge data) {
		return eval(data, "");
	}

	public MatrixDataEdge eval(MatrixDataEdge data, String rtMsg) {
		MatrixDataEdge ret = eval0(data);
		List<String> names = new ArrayList<String>();
		for (Computation c : comps) {
			if (c instanceof GroupComputation) {
				names.add(((GroupComputation) c).memo);
			} else {
				names.add(c.type());
			}
		}
		trace(rtMsg, null, data, ret, name() + " -----> " + names.toString(), 1);
		return ret;
	}

	@Override
	public String name() {
		return memo + " [componets=" + String.valueOf(comps.length) + "]";
	}

	@Override
	public int[] inShape() {
		return comps[0].inShape();
	}

	@Override
	public int[] outShape() {
		return comps[comps.length - 1].outShape();
	}

	public int size() {
		return comps.length;
	}

	public Computation getComputation(int idx) {
		return comps[idx];
	}

	public MatrixDataEdge getComputationOutput(int idx) {
		return output[idx];
	}

	public MatrixDataEdge getOutput() {
		if (!reversed) {
			return output[output.length - 1];
		} else {
			return output[0];
		}
	}

}
