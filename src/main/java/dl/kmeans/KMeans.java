package dl.kmeans;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.Function;
import java.util.stream.Collectors;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealVector;

import dataset.NNDataset;

public class KMeans {

	final static double VERY_BIG = 2 ^ 40;
	final static int MAX_TURN = 30000;
	final static int debug = 10000;

	public static void main(String[] args) {
		training(5);
	}

	static void training(int K) {
		RealVector[] origin = NNDataset.getData(NNDataset.WHOLESALE);
		String[] header = NNDataset.getHeader(NNDataset.WHOLESALE);
		header = ArrayUtils.subarray(header, 2, header.length);

		Function<RealVector, SomeThing> conv = x -> {
			SomeThing ret = new SomeThing();
			int label = ThreadLocalRandom.current().nextInt(K);
			// cutoff discrete value
			ret.desc = x.getSubVector(2, x.getDimension() - 2);
			ret.centre = ret.desc;
			ret.idx = label;
			ret.dist = -1;
			ret.last = label;
			return ret;
		};

		// intialization: assume centre for cluster
		List<SomeThing> data = Arrays.asList(origin).stream().map(conv).collect(Collectors.toList());
		List<SomeThing> centre = new ArrayList<SomeThing>();
		for (int i = 0; i < K; i++) {
			SomeThing r = new SomeThing();
			r.dist = 0d;
			r.idx = i;
			r.last = i;
			r.label = "Cluster_" + i;
			centre.add(r);
		}

		int attribute_sz = data.get(0).desc.getDimension();

		int epoch = 0;
		boolean toBeContinue = true;
		while (toBeContinue) {

			// new centre assumed
			List<List<SomeThing>> grp = mapToGrpList(data, K);
			for (int i = 0; i < grp.size(); i++) {
				generate(attribute_sz, grp.get(i), centre.get(i));
			}

			// check which cluster assigned to
			for (SomeThing st : data) {
				recheckCluster(st, centre);
			}

			// recheck if convergenced
			epoch++;
			toBeContinue = !isConvergenced(centre, data);
			toBeContinue = toBeContinue || epoch < MAX_TURN;

			if (epoch % debug == 0) {
				int err = validClusting(data, centre, false);
				System.out.println("epoch[" + epoch + "] rate: " + 1.0d * err / data.size());
			}

		}

		System.out.println("KMeans clusting for " + K + " total epoch: " + epoch);

		double[] count = new double[K];
		int total = 0;
		for (int i = 0; i < K; i++) {
			SomeThing thing = centre.get(i);
			count[i] = countDataInCluster(data, i);
			printLabel(thing, header, Double.valueOf(count[i]).intValue());
			total += count[i];
		}
		for (int i = 0; i < K; i++) {
			count[i] /= 1.0 * total;
		}

		int err = validClusting(data, centre, true);
		System.out.println("not shortest node rate: " + 1.0d * err / data.size());
		System.out.println("Cluster percent: " + Arrays.toString(count));

	}

	static void printLabel(SomeThing st, String[] header, int ct) {
		StringBuilder sb = new StringBuilder();
		sb.append(st.label).append("=").append(ct).append(" --> ");
		for (int i = 0; i < st.desc.getDimension(); i++) {
			sb.append(header[i]).append("=").append(st.desc.getEntry(i)).append(" | ");
		}
		int sz = " | ".length();
		sb.delete(sb.length() - sz, sb.length());
		System.out.println(sb.toString());
	}

	static void generate(int sz, List<SomeThing> inCluster, SomeThing ret) {
		RealVector target = MatrixUtils.createRealVector(new double[sz]);
		for (int i = 0; i < inCluster.size(); i++) {
			target = inCluster.get(i).desc.add(target);
		}
		target.mapDivideToSelf(inCluster.size());
		ret.centre = ret.desc;
		ret.desc = target;
		if (ret.centre == null) {
			ret.centre = ret.desc; // first time
		}
	}

	static List<List<SomeThing>> mapToGrpList(List<SomeThing> data, int k) {
		List<List<SomeThing>> ret = new ArrayList<List<SomeThing>>();
		for (int i = 0; i < k; i++) {
			ret.add(new ArrayList<SomeThing>());
		}
		for (SomeThing st : data) {
			Integer idx = st.idx;
			ret.get(idx).add(st);
		}
		return ret;
	}

	// using Minkowski Distance now
	// maybe using Mahalanobis Distance?
	static double getDistance(SomeThing v, SomeThing centre) {
		return v.desc.getLInfDistance(centre.desc);
	}

	static void recheckCluster(SomeThing v, List<SomeThing> centre) {
		double min = -1;
		for (int i = 0; i < centre.size(); i++) {
			SomeThing c = centre.get(i);
			double d = getDistance(v, c);
			if (d <= min || min == -1) {
				min = d;
				v.dist = d;
				v.centre = c.desc;
				v.label = c.label;
				v.last = v.idx; // backup
				v.idx = c.idx;
			}
		}
	}

	static boolean isConvergenced(List<SomeThing> centre, List<SomeThing> data) {
		// every node is unique
		int k = centre.size();
		Set<String> checker = new HashSet<String>();
		for (int i = 0; i < centre.size(); i++) {
			SomeThing a = centre.get(i);
			checker.add(a.toString());
		}
		if (checker.size() != k) {
			return false;
		}

		for (int i = 0; i < centre.size(); i++) {
			SomeThing a = centre.get(i);
			if (!a.checkV()) {
				return false;
			}
		}

		return validClusting(data, centre, false) == 0;
	}

	static int validClusting(List<SomeThing> data, List<SomeThing> centre, boolean debugIt) {
		int ret = 0;
		boolean p = false;
		for (int i = 0; i < data.size(); i++) {
			SomeThing st = data.get(i);
			int idx = -1;
			double d = st.dist + 1;
			for (int j = 0; j < centre.size(); j++) {
				double dd = getDistance(st, centre.get(j));
				if (dd < d) {
					d = dd;
					idx = j;
				}
			}
			if (idx != st.idx) {
				p = true;
				ret++;
			}
			if (p && debugIt) {
				String s = "";
				for (int j = 0; j < centre.size(); j++) {
					s += "[" + j + "]=" + getDistance(st, centre.get(j)) + "; ";
				}
				System.out.println("data[" + i + "] => " + st.idx + " | " + st.last + " || " + s);
				p = false;
			}
		}

		return ret;
	}

	static int countDataInCluster(List<SomeThing> data, int idx) {
		int ret = 0;
		for (int i = 0; i < data.size(); i++) {
			SomeThing d = data.get(i);
			if (d.idx == idx) {
				ret++;
			}
		}
		return ret;
	}

}

class SomeThing {

	RealVector desc;
	RealVector centre; // null if centre
	String label;
	Integer idx;
	double dist;
	Integer last;

	SomeThing copy() {
		SomeThing ret = new SomeThing();
		ret.desc = this.desc.copy();
		ret.centre = this.centre.copy();
		ret.label = this.label;
		ret.idx = this.idx;
		ret.dist = this.dist;
		ret.last = this.last;
		return ret;
	}

	public String toString() {
		String s = Arrays.toString(desc.toArray()) + ";" + Arrays.toString(centre.toArray()) + ";" + label + ";" + idx;
		return s;
	}

	public boolean checkV() {
		return Arrays.toString(desc.toArray()).equals(Arrays.toString(centre.toArray()));
	}

}
