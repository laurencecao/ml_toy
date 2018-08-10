package dl.nn2.model;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import dl.nn2.graph.GroupComputation;
import dl.nn2.graph.MatrixDataEdge;
import dl.nn2.graph.TracedComputation;
import dl.nn2.layer.AbstractCompGraphLayer;

public class NNDebuger {

	protected Logger logger = LoggerFactory.getLogger(NNDebuger.class);

	protected List<NNInfo> history = new ArrayList<>();
	protected List<Integer> nodes = new ArrayList<>();
	protected List<String> names = new ArrayList<>();

	public NNDebuger(List<AbstractCompGraphLayer> layers) {
		for (AbstractCompGraphLayer layer : layers) {
			nodes.add(layer.getIn());
			names.add(layer.getName());
		}
		nodes.add(layers.get(layers.size() - 1).getOut());
		names.add(0, "Input Layer");
	}

	public void snapshot(NNModel model) {
		NNInfo info = new NNInfo();
		for (Pair<GroupComputation, GroupComputation> c : model.cgs) {
			GroupComputation ff_comps = c.getLeft();
			AbstractCompGraphLayer layer = (AbstractCompGraphLayer) ff_comps.getAttach();
			RealMatrix w = layer.getWeights().asMat(0);
			info.weights.add(w.getSubMatrix(0, w.getRowDimension() - 1, 1, w.getColumnDimension() - 1));
			info.biased.add(w.getColumnVector(0));
			MatrixDataEdge r = ff_comps.getComputationOutput(ff_comps.size() - 1);
			info.ff.add(r.asMat(0).getColumnVector(0));
			GroupComputation bp_comps = c.getRight();
			r = bp_comps.getComputationOutput(bp_comps.size() - 1);
			info.bp.add(r.asMat(0).getColumnVector(0));
		}
		history.add(info);
	}

	public void printHistory(int base, int sz) {
		for (int i = base; i < history.size(); i++) {
			String s = printInfo(history.get(i), i);
			if (logger.isDebugEnabled()) {
				logger.debug(s);
			}
		}
	}
	
	public int getHistorySize() {
		return this.history.size();
	}

	public String printInfo(NNInfo info, int sn) {
		StringBuilder sb = new StringBuilder();
		String border = "-------------------------\n";
		String layer_title = "--------- Layer_%d %s ---------\n";
		sb.append("=========================" + sn + "===========================\n");
		for (int i = 0; i < info.weights.size(); i++) {
			String title = String.format(layer_title, i, names.get(i));
			sb.append(title).append("Weights").append(border);
			RealMatrix wt = info.weights.get(i);
			sb.append(TracedComputation.pretty(wt));
			sb.append(border);
			sb.append("Biased").append(border);
			sb.append(TracedComputation.pretty(info.biased.get(i)));
			sb.append(border);
			sb.append("FeedForward").append(border);
			sb.append(TracedComputation.pretty(info.ff.get(i)));
			sb.append(border);
			sb.append("BackForward").append(border);
			sb.append(TracedComputation.pretty(info.bp.get(i)));
			sb.append(border);
			sb.append(title);
		}
		sb.append("=========================" + sn + "===========================\n");
		return sb.toString();
	}

}

class NNInfo {

	protected List<RealMatrix> weights = new ArrayList<>();
	protected List<RealVector> biased = new ArrayList<>();
	protected List<RealVector> ff = new ArrayList<>();
	protected List<RealVector> bp = new ArrayList<>();

}
