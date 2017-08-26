package dl.dt;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.util.FastMath;

public class DecisionNode implements Comparable<DecisionNode> {

	final static AtomicInteger gid = new AtomicInteger(-1);

	public Integer layerId; // layer id
	public Integer siblingId; // sibling index id
	public Integer uid; // global id

	public Double entropy; // total information gain
	public Integer attrIdx; // attribute for decision
	public Double[] searchVal; // boundary values search for

	public DecisionNode parent; // for convenient
	public DecisionNode[] children; // children for more decision
	public Integer[] decisionTypes; // output class type if children is null

	public transient RealVector label;
	public transient RealMatrix example;
	public transient Set<Integer> labelNames;
	public transient Set<Integer> leads;
	public transient Set<Integer> usedAttrIdx;

	public DecisionNode(Integer layerId, Integer siblingId, RealVector label, RealMatrix example,
			Set<Integer> lbNames) {
		this.uid = gid.incrementAndGet();
		this.layerId = layerId;
		this.siblingId = siblingId;
		this.label = label;
		this.example = example;
		this.labelNames = lbNames;
		this.leads = new HashSet<Integer>();
		this.usedAttrIdx = new HashSet<Integer>();
	}

	public void build() {
		// calculate total entropy this node
		List<Integer> le = new ArrayList<Integer>(leads);
		Stream<Integer> clz = le.stream().map(x -> Double.valueOf(label.getEntry(x)).intValue());

		Map<Integer, Long> counting = clz.collect(Collectors.groupingBy(x -> x, Collectors.counting()));
		int[] v = counting.values().stream().mapToInt(x -> x.intValue()).toArray();
		this.entropy = Control.ENTROPY.apply(new int[][] { v });

		// build left , right
		int attr_idx_sz = example.getRowDimension();
		if (usedAttrIdx.size() == attr_idx_sz || v.length == 1) {
			buildLeaf();
		} else {
			List<Control> cand_attr_split = new ArrayList<Control>();
			for (int i = 0; i < attr_idx_sz; i++) {
				if (usedAttrIdx.contains(Integer.valueOf(i))) {
					continue;
				}
				cand_attr_split.add(calcAttrOptEn(i));
			}
			Collections.sort(cand_attr_split);
			Control control = cand_attr_split.get(cand_attr_split.size() - 1);
			attrIdx = control.attrId;
			searchVal = control.searchValues;
			decisionTypes = control.clzTypes;

			if (control.left != null) {

				DecisionNode left = new DecisionNode(layerId + 1, 0, label, example, labelNames);
				left.parent = this;
				for (Attr a : control.left[0]) {
					left.leads.add(a.idx);
				}
				left.usedAttrIdx.addAll(usedAttrIdx);
				left.usedAttrIdx.add(attrIdx);

				DecisionNode right = new DecisionNode(layerId + 1, 1, label, example, labelNames);
				right.parent = this;
				for (Attr a : control.left[1]) {
					right.leads.add(a.idx);
				}
				right.usedAttrIdx.addAll(usedAttrIdx);
				right.usedAttrIdx.add(attrIdx);

				this.children = new DecisionNode[] { left, right };

			}
		}

	}

	@SuppressWarnings("unchecked")
	public Control calcAttrOptEn(Integer available) {
		double[] d = this.example.getRow(available);
		List<Attr> options = new ArrayList<Attr>();
		for (int i = 0; i < d.length; i++) {
			if (this.leads.contains(Integer.valueOf(i))) {
				options.add(new Attr(i, d[i], Double.valueOf(label.getEntry(i)).intValue()));
			}
		}
		Map<Integer, Long> clzCount = options.stream()
				.collect(Collectors.groupingBy(Attr::getClzIdx, Collectors.counting()));

		Collections.sort(options);

		List<Control> posInfo = new ArrayList<Control>();
		// must 2 cutpoint
		int[] posCount = new int[labelNames.size()];
		int[] oppCount = new int[labelNames.size()];
		for (Entry<Integer, Long> en : clzCount.entrySet()) {
			oppCount[en.getKey().intValue()] = en.getValue().intValue();
		}
		for (int i = 0; i <= options.size(); i++) {

			if ((i > 0 && i < options.size()) && (options.get(i - 1).attrVal.equals(options.get(i).attrVal))) {
				Integer clzIdx = options.get(i).classIdx;
				posCount[clzIdx.intValue()]++;
				oppCount[clzIdx.intValue()]--;
				continue;
			}

			double cut = 0d;
			if (i == 0) {
				cut = options.get(0).attrVal;
			} else if (i == options.size()) {
				cut = options.get(options.size() - 1).attrVal;
			} else {
				cut = 1.0d * (options.get(i - 1).attrVal + options.get(i).attrVal) / 2;
			}
			int[][] x = new int[][] { posCount, oppCount };
			Control ret = new Control();
			ret.attrId = available;
			ret.clzTypes = null;
			ret.entropy = this.entropy;
			ret.score = ret.INFOGAIN.apply(x);
			ret.left = null;
			ret.position = i;
			ret.searchValues = new Double[] { cut };
			posInfo.add(ret);

			if (i < options.size()) {
				Integer clzIdx = options.get(i).classIdx;
				posCount[clzIdx.intValue()]++;
				oppCount[clzIdx.intValue()]--;
			}

		}

		Collections.sort(posInfo);
		Control ret = posInfo.get(posInfo.size() - 1);

		if (ret.position < 1 || ret.position >= options.size()) {
			long m = 0;
			ret.clzTypes = new Integer[] { -1 };
			for (Entry<Integer, Long> en : clzCount.entrySet()) {
				if (en.getValue() > m) {
					m = en.getValue();
					ret.clzTypes[0] = en.getKey();
				}
			}
			ret.searchValues = null;
			ret.left = null;
			ret.score = this.entropy;
			// must output 1 class type
			// force output node
			// System.out.println("leaf node: " + clzType);
			return ret;
		}

		List<Attr> left = new ArrayList<Attr>();
		List<Attr> right = new ArrayList<Attr>();
		left.addAll(options.subList(0, ret.position));
		right.addAll(options.subList(ret.position, options.size()));
		ret.left = (List<Attr>[]) Array.newInstance(left.getClass(), 2);
		ret.left[0] = left;
		ret.left[1] = right;

		return ret;

	}

	public void buildLeaf() {
		long m = 0;
		this.decisionTypes = new Integer[] { -1 };
		Stream<Integer> clz = leads.stream().map(x -> Double.valueOf(label.getEntry(x)).intValue());
		Map<Integer, Long> clzCount = clz.collect(Collectors.groupingBy(x -> x, Collectors.counting()));
		for (Entry<Integer, Long> en : clzCount.entrySet()) {
			if (en.getValue() > m) {
				m = en.getValue();
				this.decisionTypes[0] = en.getKey();
			}
		}
		this.searchVal = null;
		this.children = null;
		this.entropy = null;
		// must output 1 class type
		// force output node
		// System.out.println("leaf node: " + clzType);
	}

	public static void buildSubTree(DecisionNode parent) {
		if (parent.children == null) {
			return;
		}
		for (int i = 0; i < parent.children.length; i++) {
			DecisionNode ch = parent.children[i];
			if (ch != null) {
				ch.build();
			}
			buildSubTree(ch);
		}
	}

	public static void visitTree(DecisionNode parent, Function<DecisionNode, DecisionNode> fn) {
		if (parent != null) {
			fn.apply(parent);
		}
		if (parent.children == null) {
			return;
		}
		for (int i = 0; i < parent.children.length; i++) {
			visitTree(parent.children[i], fn);
		}
	}

	public static String dump(DecisionNode node, String[] header) {
		String val = null;
		if (node.searchVal == null) {
			val = "-->" + node.decisionTypes[0];
		} else {
			val = "<=" + node.searchVal[0];
		}
		String at = null;
		if (node.attrIdx == null) {
			at = "";
		} else {
			at = header[node.attrIdx];
		}
		String n = at + val;
		// String n = node.uid + "_" + node.layerId + "." + node.siblingId + "@"
		// + header[node.attrIdx];
		return n;
	}

	@Override
	public int compareTo(DecisionNode o) {
		if (this.layerId.compareTo(o.layerId) == 0) {
			return this.siblingId.compareTo(o.siblingId);
		}
		return this.layerId.compareTo(o.layerId);
	}

	public String toString() {
		String n = this.layerId + "." + this.siblingId + " @ " + this.attrIdx + " = " + this.entropy;
		if (children != null) {
			n += " -> {";
			for (int i = 0; i < children.length; i++) {
				String s = children[i].layerId + "." + children[i].siblingId + " @ " + children[i].attrIdx + " = "
						+ children[i].entropy;
				n += s + " , ";
			}
			n += " }";
		}
		return n;
	}

}

class Attr implements Comparable<Attr> {
	final Integer idx;
	final Double attrVal;
	final Integer classIdx;

	Integer getClzIdx() {
		return classIdx;
	}

	// example_id, attribute_value, class_id
	Attr(int idx, double val, int clz) {
		this.idx = idx;
		this.attrVal = val;
		this.classIdx = clz;
	}

	@Override
	public int compareTo(Attr o) {
		return this.attrVal.compareTo(o.attrVal);
	}
}

class Control implements Comparable<Control> {

	double entropy; // H(DS)

	// now infogain used, can be changed to gini
	double score;
	Double[] searchValues; // cutpoint search values
	Integer[] clzTypes; // output class type

	List<Attr>[] left; // sub dataset

	Integer position; // splitting position
	Integer attrId;// which attrbute using

	@Override
	public int compareTo(Control o) {
		return Double.valueOf(score).compareTo(Double.valueOf(o.score));
	}

	public Function<int[][], Double> INFOGAIN = x -> {
		return this.entropy - ENTROPY.apply(x);
	};

	public static Function<int[][], Double> ENTROPY = x -> {
		double[] ret = new double[x.length];
		int[] part = new int[x.length];
		int all = 0;
		// [0] = let; [1] = gt
		for (int i = 0; i < ret.length; i++) {
			ret[i] = 0d;
			part[i] = 0;
			for (int j = 0; j < x[i].length; j++) {
				part[i] += x[i][j];
			}
			if (part[i] == 0) {
				ret[i] = 0;
				continue;
			}
			for (int j = 0; j < x[i].length; j++) {
				double d = 1.0d * x[i][j] / part[i];
				if (d != 0) {
					ret[i] += -1d * FastMath.log(2, d) * d;
				}
			}
			all += part[i];
		}

		double r = 0d;
		for (int i = 0; i < ret.length; i++) {
			r += 1.0d * ret[i] * part[i] / all;
		}
		return r;
	};

}