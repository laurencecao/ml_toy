package dl.lda;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.distribution.EnumeratedDistribution;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.util.Pair;

import dataset.NewsCorpus;

public class GibbsLDAToy {

	final static String CORPUS = "src/main/resources/constellation/corpus.dat";
	final static String MODEL_PATH = "tmp/toyldamodel.dat";
	final static String MODEL_DIR = "tmp";
	final static int SNAPTIME = 100;

	final static double alpha = 1d;
	final static double beta = 1d;
	final static int TOPIC_K = 12;

	public static void main(String[] args) throws IOException {
		List<NewsCorpus> toy = null;
		toy = buildToy();
		toy = buildFromCorpus();
		ToyLDAModel model = training(toy);
		System.out.println(model.theta);
		System.out.println(model.phi);
	}

	static List<NewsCorpus> buildToy() {
		List<NewsCorpus> ret = new ArrayList<NewsCorpus>();
		String[] words;
		NewsCorpus d = null;
		d = new NewsCorpus();
		words = "1, 4, 3, 2, 3, 1, 4, 3, 2, 3, 1, 4, 3, 2, 3, 6".split(",");
		d.words = ArrayUtils.toObject(Arrays.asList(words).stream().mapToInt(x -> Integer.valueOf(x.trim())).toArray());
		ret.add(d);
		d = new NewsCorpus();
		words = "2, 2, 4, 2, 4, 2, 2, 2, 2, 4, 2, 2".split(", ");
		d.words = ArrayUtils.toObject(Arrays.asList(words).stream().mapToInt(x -> Integer.valueOf(x.trim())).toArray());
		ret.add(d);
		d = new NewsCorpus();
		words = "1, 6, 5, 6, 0, 1, 6, 5, 6, 0, 1, 6, 5, 6, 0, 0".split(", ");
		d.words = ArrayUtils.toObject(Arrays.asList(words).stream().mapToInt(x -> Integer.valueOf(x.trim())).toArray());
		ret.add(d);
		d = new NewsCorpus();
		words = "5, 6, 6, 2, 3, 3, 6, 5, 6, 2, 2, 6, 5, 6, 6, 6, 0".split(", ");
		d.words = ArrayUtils.toObject(Arrays.asList(words).stream().mapToInt(x -> Integer.valueOf(x.trim())).toArray());
		ret.add(d);
		d = new NewsCorpus();
		words = "2, 2, 4, 4, 4, 4, 1, 5, 5, 5, 5, 5, 5, 1, 1, 1, 1, 0".split(", ");
		d.words = ArrayUtils.toObject(Arrays.asList(words).stream().mapToInt(x -> Integer.valueOf(x.trim())).toArray());
		ret.add(d);
		d = new NewsCorpus();
		words = "5, 4, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2".split(", ");
		d.words = ArrayUtils.toObject(Arrays.asList(words).stream().mapToInt(x -> Integer.valueOf(x.trim())).toArray());
		ret.add(d);
		return ret;
	}

	static List<NewsCorpus> buildFromCorpus() throws IOException {
		List<NewsCorpus> ret = new ArrayList<NewsCorpus>();
		try (FileInputStream fis = new FileInputStream(CORPUS)) {
			BufferedReader br = new BufferedReader(new InputStreamReader(fis));
			String line;
			String[] words;
			while ((line = br.readLine()) != null) {
				words = line.trim().split("\t");
				NewsCorpus cor = new NewsCorpus();
				cor.words = ArrayUtils
						.toObject(Arrays.asList(words).stream().mapToInt(x -> Integer.valueOf(x.trim())).toArray());
				ret.add(cor);
			}
		}
		return ret;
	}

	static ToyLDAModel training(List<NewsCorpus> data) throws IOException {
		ToyLDAModel ret = null;
		ret = ToyLDAModel.loadModel(MODEL_PATH);
		if (ret == null) {
			ret = initRandomized(data);
		}
		for (int i = 0; i < 100000; i++) {
			long ts = System.currentTimeMillis();
			for (int m = 0; m < ret.data.size(); m++) {
				Integer[] words = ret.data.get(m);
				for (int w = 0; w < words.length; w++) {
					gibbsSample(ret, m, w);
				}
			}
			estimate(ret);
			ts = System.currentTimeMillis() - ts;
			if (i % SNAPTIME == 0) {
				System.out.println("checkpoint at [" + i + "]");
				ToyLDAModel.saveModel(ret, i, MODEL_PATH);
				ToyLDAModel.dumpW2TImage(ret, MODEL_DIR);
			}
			System.out.println("turn " + i + " --> " + ts + "ms");
		}
		return ret;
	}

	static List<Integer[]> convertDoc(List<NewsCorpus> data) {
		List<Integer[]> ret = new ArrayList<Integer[]>();
		for (int i = 0; i < data.size(); i++) {
			NewsCorpus c = data.get(i);
			ret.add(c.words);
		}
		return ret;
	}

	static ToyLDAModel initRandomized(List<NewsCorpus> data) {
		ToyLDAModel ret = new ToyLDAModel(alpha, beta, TOPIC_K, convertDoc(data));
		ret.z = new ArrayList<Integer[]>();
		for (int i = 0; i < ret.data.size(); i++) {
			// i -> docId
			Integer[] c = ret.data.get(i);
			Integer[] theZ = new Integer[c.length];
			for (int j = 0; j < c.length; j++) {
				// j -> word index
				theZ[j] = ThreadLocalRandom.current().nextInt(TOPIC_K);
				double cc = ret.zCount.getEntry(theZ[j]) + 1;
				ret.zCount.setEntry(theZ[j], cc);
				cc = ret.docTopicCount.getEntry(i, theZ[j]) + 1;
				ret.docTopicCount.setEntry(i, theZ[j], cc);
				cc = ret.wordTopicCount.getEntry(c[j], theZ[j]) + 1;
				ret.wordTopicCount.setEntry(c[j], theZ[j], cc);
			}
			ret.z.add(theZ);
		}
		return ret;
	}

	/**
	 * <pre>
	 * 𝜶 → |
	 * 	   | 𝜽(𝒅) → |                   |
	 * 				|  𝑍(𝑛,𝑑) → 𝑊(𝑛,𝑑)  |
	 * 				|					|	← 𝝍(𝒌) |
	 * 												|← 𝜷
	 * 
	 * 1. 𝜽(𝑑) ∼ 𝐷𝑖𝑟𝑖𝑐h𝑙𝑒𝑡(𝜶)
	 * 2. 𝒁(𝑛,𝑑)|𝜽(𝒅) ∼ 𝑀𝑢𝑙𝑡𝑖𝑛𝑜𝑚𝑖𝑎𝑙(𝜽(𝑑))
	 * 3. 𝝍(𝒌) ∼ 𝐷𝑖𝑟𝑖𝑐h𝑙𝑒𝑡(𝜷)
	 * 4. 𝑊(𝑛,𝑑)|𝒁(𝑛,𝑑) = 𝑘, 𝝍(𝑘) ∼ 𝑀𝑢𝑙𝑡𝑖𝑛𝑜𝑚𝑖𝑎𝑙(𝝍(𝑘))
	 * 
	 * for GIBBS SAMPLING estimation:
	 * 
	 * 𝑃(𝒁(𝑛,𝑑)=𝑘|𝒁(−𝑛,𝑑),𝜶) 
	 *  			= (𝐶𝑘(−𝑛,𝑑) + 𝛼) / (𝐶(−𝑛,𝑑) + 𝐾𝛼)
	 *  
	 * 𝑃(𝑊(𝑛,𝑑)=𝑤|𝒁(𝑛,𝑑)=𝑘,𝒁(−𝑛,𝑑),𝑊(−𝑛,𝑑))
	 * 			= (𝐶(−𝑛,*,𝑤,𝑘) + 𝛽) / (𝐶(−𝑛,*,𝑘) + 𝑊*𝛽)
	 * 
	 * </pre>
	 */
	static int gibbsSample(ToyLDAModel model, int docId, int wordIdx) {
		int z = model.z.get(docId)[wordIdx];
		int word = model.data.get(docId)[wordIdx];
		double count = 0;
		count = model.docTopicCount.getEntry(docId, z) - 1;
		if (count < 0) {
			System.out.println("aaaa");
		}
		model.docTopicCount.setEntry(docId, z, count);
		count = model.wordTopicCount.getEntry(word, z) - 1;
		if (count < 0) {
			System.out.println("bbbb");
		}
		model.wordTopicCount.setEntry(word, z, count);
		count = model.zCount.getEntry(z) - 1;
		if (count < 0) {
			System.out.println("cccc");
		}
		model.zCount.setEntry(z, count);

		List<Pair<Integer, Double>> items = new ArrayList<Pair<Integer, Double>>();
		// iterate at Z
		for (z = 0; z < model.K; z++) {
			RealVector iden = null;
			RealVector v = null;
			iden = MatrixUtils.createRealVector(new double[model.K]);
			iden.set(1);

			double d2t_d = model.docTopicCount.getEntry(docId, z);
			v = model.docTopicCount.getRowVector(docId);
			double d2t_all = iden.dotProduct(v);

			iden = MatrixUtils.createRealVector(new double[model.wordTopicCount.getRowDimension()]);
			iden.set(1);
			double w2t_w = model.wordTopicCount.getEntry(word, z);
			v = model.wordTopicCount.getColumnVector(z);
			double w2t_all = iden.dotProduct(v);

			double A = (d2t_d + model.alpha) / (d2t_all + model.K * model.alpha);
			double B = (w2t_w + model.beta) / (w2t_all + model.W * model.beta);

			if (A * B < 0) {
				System.out.println("ERROR: " + A * B);
			}
			items.add(new Pair<Integer, Double>(z, A * B));
		}

		EnumeratedDistribution<Integer> dist = new EnumeratedDistribution<Integer>(items);
		z = dist.sample();

		count = model.docTopicCount.getEntry(docId, z) + 1;
		model.docTopicCount.setEntry(docId, z, count);
		model.z.get(docId)[wordIdx] = z;
		count = model.wordTopicCount.getEntry(word, z) + 1;
		model.wordTopicCount.setEntry(word, z, count);
		count = model.zCount.getEntry(z) + 1;
		model.zCount.setEntry(z, count);

		RealVector id = MatrixUtils.createRealVector(new double[model.K]);
		id.set(1);
		assert 16 == id.dotProduct(model.docTopicCount.getRowVector(0));
		assert 12 == id.dotProduct(model.docTopicCount.getRowVector(1));
		assert 16 == id.dotProduct(model.docTopicCount.getRowVector(2));
		assert 17 == id.dotProduct(model.docTopicCount.getRowVector(3));
		assert 18 == id.dotProduct(model.docTopicCount.getRowVector(4));
		assert 12 == id.dotProduct(model.docTopicCount.getRowVector(5));

		id = MatrixUtils.createRealVector(new double[model.wordTopicCount.getRowDimension()]);
		id.set(1);
		double ccc = 0d;
		ccc += id.dotProduct(model.wordTopicCount.getColumnVector(0));
		ccc += id.dotProduct(model.wordTopicCount.getColumnVector(1));
		// ccc += id.dotProduct(model.wordTopicCount.getColumnVector(2));
		assert 91 == ccc;
		return z;
	}

	static void estimate(ToyLDAModel model) {
		for (int m = 0; m < model.docTopicCount.getRowDimension(); m++) {
			// update theta
			for (int z = 0; z < model.K; z++) {
				RealVector iden = null;
				RealVector v = null;
				iden = MatrixUtils.createRealVector(new double[model.K]);
				iden.set(1);

				double d2t_d = model.docTopicCount.getEntry(m, z);
				v = model.docTopicCount.getRowVector(m);
				double d2t_all = iden.dotProduct(v);
				double A = (d2t_d + model.alpha) / (d2t_all + model.K * model.alpha);
				model.theta.setEntry(m, z, A);
			}
		}

		for (int w = 1; w < model.wordTopicCount.getRowDimension(); w++) {
			// update phi
			for (int z = 0; z < model.K; z++) {
				RealVector iden = null;
				RealVector v = null;
				iden = MatrixUtils.createRealVector(new double[model.wordTopicCount.getRowDimension()]);
				iden.set(1);

				double w2t_w = model.wordTopicCount.getEntry(w, z);
				v = model.wordTopicCount.getColumnVector(z);
				double w2t_all = iden.dotProduct(v);

				double B = (w2t_w + model.beta) / (w2t_all + model.W * model.beta);
				model.phi.setEntry(w, z, B);
			}
		}
	}

}
