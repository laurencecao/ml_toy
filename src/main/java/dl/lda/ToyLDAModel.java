package dl.lda;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import umontreal.ssj.randvarmulti.DirichletGen;
import umontreal.ssj.rng.WELL607;
import utils.ImageUtils;

public class ToyLDAModel implements Serializable {

	private static final long serialVersionUID = 1L;

	transient DirichletGen dirTheta;
	transient DirichletGen dirPhi;

	// parameter of the Dirichlet prior on the per-document topic distributions
	final double alpha;
	// parameter of the Dirichlet prior on the per-topic word distribution
	final double beta;
	// topic number
	final int K;
	// training data
	final List<Integer[]> data;

	List<Integer[]> z; // every word's corresponding topic
	RealMatrix docTopicCount;
	RealMatrix wordTopicCount;
	RealVector zCount;
	int W;

	RealMatrix theta;
	RealMatrix phi;

	ToyLDAModel(double alpha, double beta, int K, List<Integer[]> data) {
		this.alpha = alpha;
		this.beta = beta;
		this.K = K;
		this.data = data;
		this.docTopicCount = MatrixUtils.createRealMatrix(data.size(), K);
		this.zCount = MatrixUtils.createRealVector(new double[K]);
		Set<Integer> dict = new HashSet<Integer>();
		for (int i = 0; i < data.size(); i++) {
			Integer[] w = data.get(i);
			for (int j = 0; j < w.length; j++) {
				dict.add(w[j]);
			}
		}
		ArrayList<Integer> lst = new ArrayList<Integer>(dict);
		Integer max = Collections.max(lst);
		this.W = max + 1;
		this.wordTopicCount = MatrixUtils.createRealMatrix(this.W, this.K);

		this.theta = MatrixUtils.createRealMatrix(data.size(), this.K);
		this.phi = MatrixUtils.createRealMatrix(this.W, this.K);

		double[] alphas = new double[K];
		Arrays.fill(alphas, alpha);
		double[] betas = new double[wordTopicCount.getRowDimension()];
		Arrays.fill(betas, beta);
		this.dirTheta = new DirichletGen(new WELL607(), alphas);
		this.dirPhi = new DirichletGen(new WELL607(), betas);
	}

	public static void saveModel(ToyLDAModel model, int turn, String path) throws IOException {
		File fos = new File(path + "." + turn);
		try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(fos))) {
			oos.writeObject(model);
		}
	}

	public static ToyLDAModel loadModel(String path) {
		try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(path))) {
			ToyLDAModel model = (ToyLDAModel) ois.readObject();
			double[] alphas = new double[model.K];
			Arrays.fill(alphas, model.alpha);
			double[] betas = new double[model.wordTopicCount.getRowDimension()];
			Arrays.fill(betas, model.beta);
			model.dirTheta = new DirichletGen(new WELL607(), alphas);
			model.dirPhi = new DirichletGen(new WELL607(), betas);
			return model;
		} catch (Exception e) {
			// e.printStackTrace();
		}
		return null;
	}

	public static void dumpW2TImage(ToyLDAModel model, String basedir) throws IOException {
		RealMatrix m = model.wordTopicCount;
		RealVector iden = MatrixUtils.createRealVector(new double[m.getRowDimension()]);
		iden.set(1);
		for (int i = 0; i < m.getColumnDimension(); i++) {
			RealVector w2t = m.getColumnVector(i);
			double deno = w2t.dotProduct(iden);
			RealVector v = w2t.mapDivideToSelf(deno);
			ImageUtils.Vector2BMP(v.toArray(), basedir + "/word2topic_" + (i + 1) + ".bmp");
		}
	}

}
