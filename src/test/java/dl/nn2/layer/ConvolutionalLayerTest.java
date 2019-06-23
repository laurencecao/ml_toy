package dl.nn2.layer;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.junit.Before;
import org.junit.Test;

import dl.nn2.activation.NoOp;
import dl.nn2.graph.GroupComputation;
import dl.nn2.graph.MatrixDataEdge;
import dl.nn2.graph.Reshape;
import dl.nn2.init.Xavier;

public class ConvolutionalLayerTest {

	ConvolutionalLayer layer;
	DebuggerLayer last;
	MatrixDataEdge data;
	MatrixDataEdge lost;

	@Before
	public void setUp() {
		Xavier.debug = .1d;
		RealMatrix d0 = MatrixUtils.createRealMatrix(new double[][] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } });
//		RealMatrix in1 = MatrixUtils.createRealMatrix(new double[][] { { 1, 1, 1 }, { 1, 1, 1 }, { 1, 1, 1 } });
		RealMatrix d1 = MatrixUtils.createRealMatrix(new double[][] { { 1, 1, 1 }, { 1, 1, 1 }, { 1, 1, 1 } });

		RealMatrix r0 = MatrixUtils.createRealMatrix(new double[][] { { 0.2833, 0.3500 }, { 0.4833, 0.5500 } });
		RealMatrix r1 = MatrixUtils.createRealMatrix(new double[][] { { 0.2833, 0.3500 }, { 0.4833, 0.5500 } });
		RealMatrix r2 = MatrixUtils.createRealMatrix(new double[][] { { 0.2833, 0.3500 }, { 0.4833, 0.5500 } });

		data = new MatrixDataEdge("foobar");
		data.addToMatList(d0);
		data.addToMatList(d1);

		layer = new ConvolutionalLayer(new int[] { 3, 3 }, 2, 2, 3, 1, 0, new Reshape(new int[] { 1, -1 }, true),
				"conv1");
		layer.setBiased(.1d);
		layer.setActivationFunction(new NoOp());

		List<RealMatrix> dd = new ArrayList<>();
		dd.add(r0);
		dd.add(r1);
		dd.add(r2);
		last = new DebuggerLayer(new int[] { 2, 2 }, dd, dd, new Reshape(new int[] { 1, -1 }, true));
		layer.setNextLayer(last);

		lost = new MatrixDataEdge("lost");
		lost.addToMatList(r0);
		lost.addToMatList(r1);
		lost.addToMatList(r2);
	}

//	@Test
	public void testForward() {
		Pair<GroupComputation, GroupComputation> comps = layer.build();
		GroupComputation ff = comps.getLeft();
		MatrixDataEdge r = ff.eval(data);
		System.out.println("total data siz: " + r.asMatList().size());
		for (int i = 0; i < r.asMatList().size(); i++) {
			RealMatrix d = r.asMatList().get(i);
			System.out.println(MatrixDataEdge.pretty0(d));
		}
	}

	@Test
	public void testBackward() {
		Pair<GroupComputation, GroupComputation> comps = layer.build();
		GroupComputation ff = comps.getLeft();
		MatrixDataEdge r0 = ff.eval(data);
		GroupComputation bp = comps.getRight();
		MatrixDataEdge r1 = bp.eval(lost);
		System.out.println("total data siz: " + r1.asMatList().size());
		for (int i = 0; i < r1.asMatList().size(); i++) {
			RealMatrix d = r1.asMatList().get(i);
			System.out.println(MatrixDataEdge.pretty0(d));
		}
	}

}
