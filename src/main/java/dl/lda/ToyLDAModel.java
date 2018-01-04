package dl.lda;

import java.io.IOException;
import java.io.Serializable;
import java.util.List;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import utils.ImageUtils;

public class ToyLDAModel implements Serializable {

	private static final long serialVersionUID = 1L;

	// parameter of the Dirichlet prior on the per-document topic distributions
	final double alpha;
	// parameter of the Dirichlet prior on the per-topic word distribution
	final double beta;
	// topic number
	final int K;
	// training data
	final List<Integer[]> data;

	RealMatrix z; // every word's corresponding topic
	RealMatrix docTopicCount;
	RealMatrix wordTopicCount;
	RealVector zCount;
	final int V; // vocabulary uniqe count
	final int D; // document count
	final int N; // document length

	final RealMatrix theta;
	final RealMatrix phi;

	ToyLDAModel(double alpha, double beta, int K, int V, int D, int N, List<Integer[]> data) {
		this.alpha = alpha;
		this.beta = beta;
		this.K = K;
		this.V = V;
		this.D = D;
		this.N = N;
		this.data = data;

		this.z = MatrixUtils.createRealMatrix(N, D);
		this.docTopicCount = MatrixUtils.createRealMatrix(D, K);
		this.wordTopicCount = MatrixUtils.createRealMatrix(V, K);
		this.zCount = MatrixUtils.createRealVector(new double[K]);

		this.theta = MatrixUtils.createRealMatrix(K, D);
		this.phi = MatrixUtils.createRealMatrix(V, K);
	}

	public void toImage(String outbase) throws IOException {
		for (int i = 0; i < wordTopicCount.getColumnDimension(); i++) {
			String outname = outbase + "/out" + i + ".bmp";
			RealVector wordCount = wordTopicCount.getColumnVector(i);
			double[] pixel = new double[wordCount.getDimension()];
			for (int p = 0; p < pixel.length; p++) {
				pixel[p] = wordCount.getEntry(p);
				pixel[p] = pixel[p] > 255 ? 255 : pixel[p];
				pixel[p] = pixel[p] < 0 ? 0 : pixel[p];
			}
			ImageUtils.Vector2BMP(pixel, outname);
		}
	}

}
