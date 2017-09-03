package dl.dbscan;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;
import java.util.stream.Collectors;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import dataset.NNDataset;
import utils.DrawingUtils;

public class TinyDBScan {

	public static void main(String[] args) throws IOException {
		RealVector[] origin = NNDataset.getData(NNDataset.WINGNUT);

		AtomicInteger seq = new AtomicInteger(0);
		Function<RealVector, VisualPoint> conv = x -> {
			int idx = seq.getAndIncrement();
			// cutoff discrete value
			RealVector v = x.getSubVector(1, x.getDimension() - 1);
			VisualPoint ret = new VisualPoint(idx, v);
			return ret;
		};

		// intialization: assume centre for cluster
		List<VisualPoint> data = Arrays.asList(origin).stream().map(conv).collect(Collectors.toList());
		Collections.sort(data);

		double eps = 0.25;
		int mPts = 7;

		training(data, eps, mPts);

		printVisualPoints(data);

		printDBScan(data);
	}

	static void training(List<VisualPoint> data, double eps, int mPts) {
		buildNeighbors(data, eps, mPts);
		AtomicInteger labelIdx = new AtomicInteger(0);

		boolean term = false;
		while (!term) {
			//
			buildCore(data, eps, mPts, labelIdx);
			term = checkDone(data);
		}
	}

	static void buildNeighbors(List<VisualPoint> data, double eps, int mPts) {
		for (int i = 0; i < data.size(); i++) {
			VisualPoint d1 = data.get(i);
			for (int j = i + 1; j < data.size(); j++) {
				VisualPoint d2 = data.get(j);
				// System.out.println(d1.v.getLInfDistance(d2.v));
				if (d1.v.getDistance(d2.v) <= eps) {
					d1.candidate.add(d2.idx);
					d2.candidate.add(d1.idx);
				}
			}
			if (d1.candidate.size() + 1 >= mPts) {
				d1.type = VisualPoint.core;
			}
		}
	}

	static void buildCore(List<VisualPoint> data, double eps, int mPts, AtomicInteger labelIdx) {
		Set<VisualPoint> cores = data.stream().filter(x -> x.type == VisualPoint.core).collect(Collectors.toSet());
		for (VisualPoint c : cores) {
			if (c.label.equals(VisualPoint.EMPTY)) {
				c.label = "Cluster_" + labelIdx.getAndIncrement();
			}
			Set<Integer> visited = new HashSet<Integer>();
			buildCore(c, data, visited);
		}
	}

	static void buildCore(VisualPoint core, List<VisualPoint> data, Set<Integer> visited) {
		for (Integer idx : core.candidate) {
			if (visited.contains(idx)) {
				continue;
			}
			visited.add(idx);
			VisualPoint d1 = data.get(idx);
			if (d1.label.equals(VisualPoint.EMPTY)) {
				d1.label = core.label;
			}
			if (d1.type == VisualPoint.noisy) {
				d1.type = VisualPoint.border;
			} else if (d1.type == VisualPoint.core) {
				buildCore(d1, data, visited);
			}
		}
	}

	static boolean checkDone(List<VisualPoint> data) {
		for (VisualPoint d : data) {
			if (d.type != VisualPoint.noisy) {
				continue;
			}
			for (Integer idx : d.candidate) {
				VisualPoint c = data.get(idx);
				if (c.type == VisualPoint.core) {
					return false;
				}
			}
		}
		return true;
	}

	static void printVisualPoints(List<VisualPoint> data) {
		Map<String, Long> count1 = data.stream().map(x -> x.label)
				.collect(Collectors.groupingBy(x -> x, Collectors.counting()));

		Map<Integer, Long> count2 = data.stream().map(x -> x.type)
				.collect(Collectors.groupingBy(x -> x, Collectors.counting()));

		StringBuilder sb = new StringBuilder();
		String bar = "-----------------------------------";
		sb.append(bar).append("Cluster").append(bar).append("\n");
		for (Entry<String, Long> en : count1.entrySet()) {
			sb.append(en.getKey()).append("=").append(en.getValue()).append("\n");
		}
		sb.append(bar).append("Type").append(bar).append("\n");
		for (Entry<Integer, Long> en : count2.entrySet()) {
			String msg = "Noisy";
			if (en.getKey().equals(1)) {
				msg = "BORDER";
			} else if (en.getKey().equals(2)) {
				msg = "CORE";
			}
			sb.append(msg).append("=").append(en.getValue()).append("\n");
		}

		System.out.println(sb.toString());
	}

	static void printDBScan(List<VisualPoint> data) throws IOException {
		Map<String, List<VisualPoint>> d = data.stream().collect(Collectors.groupingBy(VisualPoint::getLabel));
		List<String> title = new ArrayList<String>();
		List<double[][]> xy = new ArrayList<double[][]>();
		d.forEach((k, v) -> {
			title.add(k);
			double[][] yy = new double[v.size()][];
			for (int i = 0; i < yy.length; i++) {
				yy[i] = v.get(i).v.toArray();
			}
			RealMatrix m = MatrixUtils.createRealMatrix(yy);
			xy.add(m.transpose().getData());
		});
		DrawingUtils.drawClusterXY(title, xy, "tmp/dbscan.png");
	}

}

class VisualPoint implements Comparable<VisualPoint> {

	final static String EMPTY = "${NONE}";
	final static int noisy = 0;
	final static int border = 1;
	final static int core = 2;

	// global unique index
	final int idx;

	// density type
	int type = noisy;

	// vector space
	final RealVector v;

	// directly reachable
	Set<Integer> reachable0 = new HashSet<Integer>();

	// density reachable
	Set<Integer> reachable1 = new HashSet<Integer>();

	// density connected
	Set<Integer> connected = new HashSet<Integer>();

	// cluster name or label
	String label = EMPTY;

	// if any changed made
	transient boolean changed;

	// staring set
	transient List<Integer> candidate = new ArrayList<Integer>();

	VisualPoint(int idx, RealVector v) {
		this.idx = idx;
		this.v = v;
	}

	@Override
	public int compareTo(VisualPoint o) {
		return Integer.valueOf(idx).compareTo(Integer.valueOf(o.idx));
	}

	public String getLabel() {
		return label;
	}

}
