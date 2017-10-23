package dl.lda;

import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.distribution.EnumeratedDistribution;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.util.Pair;

import umontreal.ssj.randvarmulti.DirichletGen;
import umontreal.ssj.rng.WELL607;
import utils.ImageUtils;

/**
 * <pre>
 * notice: using LDA-C by David M. Blei
 * 
 * mysettings.txt:
 * var max iter 20000
 * var convergence 1e-6
 * em max iter 1000
 * em convergence 1e-4
 * alpha fixed
 * 
 * command line: 
 * lda est 1 3 mysettings.txt corpus.lda random mydir
 * 
 * </pre>
 * 
 * @author caowenjiong
 *
 */
public class LDAGenerator {

	final static String IMG_BASE = "src/main/resources/constellation";
	final static double INIT_ALPHA = 1d;
	final static String CORPUS_NAME = "corpus.dat";
	final static String CORPUS_NAME2 = "corpus.lda";
	final static int VOCABULARY_COUNT = 5 * 5;
	final static double TITTLE = 0.0001d;

	public static void main(String[] args) throws IOException {
		RealMatrix w2t = initModel(0, 3, VOCABULARY_COUNT);
		int[][] corpus = generate2(w2t, 2000, 100, w2t.getColumnDimension());
		try (FileOutputStream fos = new FileOutputStream(IMG_BASE + "/" + CORPUS_NAME)) {
			for (int i = 0; i < corpus.length; i++) {
				StringBuilder sb = new StringBuilder();
				for (int j = 0; j < corpus[i].length; j++) {
					sb.append(corpus[i][j]).append(" ");
				}
				sb.replace(sb.length() - 1, sb.length(), "\n");
				fos.write(sb.toString().getBytes());
			}
		}
		try (FileOutputStream fos = new FileOutputStream(IMG_BASE + "/" + CORPUS_NAME2)) {
			fos.write(("" + corpus.length + "\n").getBytes());
			for (int i = 0; i < corpus.length; i++) {
				StringBuilder sb = new StringBuilder();
				List<Integer> lst = Arrays.asList(ArrayUtils.toObject(corpus[i]));
				Map<Integer, Integer> item = lst.stream()
						.collect(Collectors.groupingBy(x -> x, Collectors.summingInt(p -> 1)));

				sb.append(item.size());
				Iterator<Entry<Integer, Integer>> it = item.entrySet().iterator();
				while (it.hasNext()) {
					Entry<Integer, Integer> en = it.next();
					sb.append(" ").append(en.getKey()).append(":").append(en.getValue());
				}
				sb.append("\n");
				fos.write(sb.toString().getBytes());
			}
		}
	}

	static RealMatrix initModel(int baseId, int count, int V) throws IOException {
		RealMatrix ret = MatrixUtils.createRealMatrix(V, count);
		for (int i = 0; i < count; i++) {
			int id = baseId + i;
			String p = IMG_BASE + "/t" + id + ".bmp";
			double[] vec = ImageUtils.BMP2Vector(p);
			RealVector v = MatrixUtils.createRealVector(vec);
			ret.setColumnVector(i, v);
		}
		return ret;
	}

	static int[][] generate2(RealMatrix p, int M, int N, int K) {
		int[][] ret = new int[M][N];
		// get distribution for a word over a topic
		// word1@topic1, word2@topic1, ... , wordN@topic1
		// word1@topic2, word2@topic2, ... , wordN@topic2
		// word1@topicK, word2@topicK, ... , wordN@topicK
		List<EnumeratedDistribution<Integer>> diceB = getDiceB(p);
		for (int m = 0; m < M; m++) {
			// every document
			ret[m] = new int[N];
			// get distribution for a topic over a document
			// topic1@doc1, topic2@doc1, ... , topicK@doc1
			// topic1@doc2, topic2@doc2, ... , topicK@doc2
			// topic1@docM, topic2@docM, ... , topicK@docM
			DirichletGen genA = getDiceA(K);
			EnumeratedDistribution<Integer> diceA = getMultinomial(genA);
			for (int n = 0; n < N; n++) {
				// every word at this document
				Integer z = diceA.sample();
				EnumeratedDistribution<Integer> dice = diceB.get(z);
				ret[m][n] = dice.sample();
			}
		}
		return ret;
	}

	static int[][] generate(RealMatrix v2t, int docCount, int docSize) {
		int T = v2t.getColumnDimension();
		int[][] ret = new int[docCount][];

		List<EnumeratedDistribution<Integer>> wordByTopic = new ArrayList<>();
		for (int i = 0; i < v2t.getColumnDimension(); i++) {
			double[] w2t = v2t.getColumn(i);
			double[] wordDist = newDirichlet(w2t);
			wordByTopic.add(newMultinomial(wordDist));
		}

		for (int i = 0; i < docCount; i++) {
			System.out.println("Generating Doc[" + i + "] ......");
			ret[i] = new int[docSize];
			double[] theta = new double[T];
			Arrays.fill(theta, INIT_ALPHA);
			double[] topics = newDirichlet(theta);
			EnumeratedDistribution<Integer> tDist = newMultinomial(topics);
			// sample 𝜽(𝑑) ∼ 𝐷𝑖𝑟𝑖𝑐h𝑙𝑒𝑡(𝜶)
			for (int j = 0; j < docSize; j++) {
				Integer t = tDist.sample();
				EnumeratedDistribution<Integer> wDist = wordByTopic.get(t);
				Integer w = wDist.sample();
				ret[i][j] = w;
			}
		}

		return ret;
	}

	/**
	 * Dice of Topics_OF_Document: distribution for a topic over a document
	 */
	public static DirichletGen getDiceA(int K) {
		double[] alpha = new double[K];
		Arrays.fill(alpha, 1);
		return new DirichletGen(new WELL607(), alpha);
	}

	/**
	 * Dice of Words_OF_Topic: distribution for a word over a topic
	 */
	public static List<EnumeratedDistribution<Integer>> getDiceB(RealMatrix word2topic) {
		List<EnumeratedDistribution<Integer>> ret = new ArrayList<EnumeratedDistribution<Integer>>();
		for (int k = 0; k < word2topic.getColumnDimension(); k++) {
			RealVector alpha = word2topic.getColumnVector(k);
			// alpha = alpha.mapAdd(TITTLE);
			System.out.println(Arrays.toString(alpha.toArray()));
			DirichletGen gen = new DirichletGen(new WELL607(), alpha.toArray());
			ret.add(getMultinomial(gen));
		}
		return ret;
	}

	static EnumeratedDistribution<Integer> getMultinomial(DirichletGen gen) {
		double[] p = new double[gen.getDimension()];
		gen.nextPoint(p);
		return newMultinomial(p);
	}

	static double[] newDirichlet(double[] alpha) {
		int T = alpha.length;
		DirichletGen dirT = new DirichletGen(new WELL607(), alpha);
		double[] ret = new double[T];
		dirT.nextPoint(ret);
		return ret;
	}

	static EnumeratedDistribution<Integer> newMultinomial(double[] p) {
		List<Pair<Integer, Double>> t = new ArrayList<>();
		IntStream.rangeClosed(0, p.length - 1).forEach(x -> t.add(new Pair<Integer, Double>(x, p[x])));
		return new EnumeratedDistribution<>(t);
	}

}
