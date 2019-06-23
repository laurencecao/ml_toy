package dl.nn2.layer;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import dl.nn2.graph.GroupComputation;
import dl.nn2.graph.MatrixDataEdge;
import dl.nn2.graph.Reshape;

public class PoolingLayerTest {

	PoolingLayer layer;
	DebuggerLayer last;
	MatrixDataEdge data;
	MatrixDataEdge lost;

	@Before
	public void setUp() {
		double[][] d0 = { { 1, 2, 3, 4 }, { 6, 4, 5, 6 }, { 7, 8, 9, 2 }, { 4, 7, 1, 2 } };
		double[][] d1 = { { 1, 1, 1, 1 }, { 1, 1, 1, 1 }, { 1, 1, 1, 1 }, { 1, 1, 1, 1 } };

		double[][] r0 = { { 60, 60 }, { 80, 90 } };
		double[][] r1 = { { 10, 10 }, { 10, 10 } };

		layer = new PoolingLayer(2, new int[] { -1, -1 }, "pooling");
		data = new MatrixDataEdge("foobar");
		data.addToMatList(MatrixUtils.createRealMatrix(d0));
		data.addToMatList(MatrixUtils.createRealMatrix(d1));

		List<RealMatrix> dd = new ArrayList<>();
		dd.add(MatrixUtils.createRealMatrix(r0));
		dd.add(MatrixUtils.createRealMatrix(r1));
		last = new DebuggerLayer(new int[] { 2, 2 }, dd, dd, new Reshape());
		layer.setNextLayer(last);

		lost = new MatrixDataEdge("lost");
		lost.addToMatList(dd.get(0));
		lost.addToMatList(dd.get(1));
	}

	@After
	public void tearDown() {

	}

	@Test
	public void test1() {
		for (RealMatrix d : data.asMatList()) {
			System.out.println("!");
			System.out.println(MatrixDataEdge.pretty0(d));
		}
		System.out.println("******");

		Pair<GroupComputation, GroupComputation> comps = layer.build();
		MatrixDataEdge r1 = comps.getLeft().eval(data);
		for (RealMatrix d : r1.asMatList()) {
			System.out.println("!");
			System.out.println(MatrixDataEdge.pretty0(d));
		}
		System.out.println("-----");
		MatrixDataEdge r2 = comps.getRight().eval(r1);
		for (RealMatrix d : r2.asMatList()) {
			System.out.println("!");
			System.out.println(MatrixDataEdge.pretty0(d));
		}
		System.out.println("=====");
		MatrixDataEdge r3 = comps.getRight().eval(lost);
		for (RealMatrix d : r3.asMatList()) {
			System.out.println("!");
			System.out.println(MatrixDataEdge.pretty0(d));
		}
	}

}
