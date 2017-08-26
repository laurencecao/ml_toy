package dl.dt;

import java.io.IOException;
import java.util.HashSet;
import java.util.Set;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import dataset.NNDataset;
import utils.DrawingUtils;

public class SimpleC45 {

	public static void main(String[] args) throws IOException {
		playC45();
	}

	/**
	 * sigh! bullshit, maybe GA for tree constructor is better!
	 * 
	 * @throws IOException
	 */
	static void playC45() throws IOException {
		RealVector[] data = NNDataset.getData(NNDataset.HOME);
		RealVector[] label = NNDataset.getLabel(NNDataset.HOME);
		String[] header = NNDataset.getHeader(NNDataset.HOME);
		RealMatrix d = MatrixUtils.createRealMatrix(data.length, data[0].getDimension());
		for (int i = 0; i < data.length; i++) {
			d.setRowVector(i, data[i]);
		}
		d = d.transpose(); // data = column vector

		RealVector l = MatrixUtils.createRealVector(new double[label.length]);
		for (int i = 0; i < l.getDimension(); i++) {
			l.setEntry(i, label[i].getEntry(0));
		}

		DecisionNode root = training(d, l);
		// printTree(root, header);

		Integer lb = predict(root, data[0]);
		System.out.println(lb);

		double ok = MSE(root, data, label);
		System.out.println("total hit: " + ok);
	}

	static Set<Integer> labelNames(RealVector lb) {
		Set<Integer> ret = new HashSet<Integer>();
		for (int i = 0; i < lb.getDimension(); i++) {
			ret.add(Double.valueOf(lb.getEntry(i)).intValue());
		}
		return ret;
	}

	static DecisionNode training(RealMatrix examples, RealVector labels) throws IOException {
		DecisionNode root = new DecisionNode(0, 0, labels, examples, labelNames(labels));
		int sz = examples.getColumnDimension();
		for (int i = 0; i < sz; i++) {
			root.leads.add(i);
		}
		root.build();
		DecisionNode.buildSubTree(root);

		// initialization finished
		return root;
	}

	static void printTree(DecisionNode root, String[] header) throws IOException {
		DrawingUtils.drawTree(root, header);
	}

	static Integer predict(DecisionNode root, RealVector data) {
		DecisionNode node = root;
		while (true) {
			if (node.decisionTypes != null) {
				return node.decisionTypes[0];
			}
			int idx = node.attrIdx;
			double val = data.getEntry(idx);
			Double cutpoint = node.searchVal[0];
			if (val <= cutpoint) {
				node = node.children[0];
			} else {
				node = node.children[1];
			}
		}
	}

	static double MSE(DecisionNode root, RealVector[] data, RealVector[] label) {
		int right = 0, wrong = 0;
		for (int i = 0; i < data.length; i++) {
			Integer lb = predict(root, data[i]);
			if (label[i].getEntry(0) == lb) {
				right++;
			} else {
				wrong++;
			}
		}
		return 1.0d * right / (right + wrong);
	}

}
