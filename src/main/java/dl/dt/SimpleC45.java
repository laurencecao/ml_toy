package dl.dt;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import dataset.NNDataset;

public class SimpleC45 {

	public static void main(String[] args) {
		playC45();
	}

	/**
	 * sigh! bullshit, maybe GA for tree constructor is better!
	 */
	static void playC45() {
		RealVector[] data = NNDataset.getData(NNDataset.HOME);
		RealVector[] label = NNDataset.getLabel(NNDataset.HOME);
		RealMatrix d = MatrixUtils.createRealMatrix(data.length, data[0].getDimension() + 1);
		for (int i = 0; i < data.length; i++) {
			d.setEntry(i, 0, label[i].getEntry(0));
			d.setSubMatrix(new double[][] { data[i].toArray() }, i, 1);
		}
		d = d.transpose(); // data = column vector

	}

	static void pickAttr(int i) {

	}

}
