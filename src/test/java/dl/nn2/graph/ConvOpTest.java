package dl.nn2.graph;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import dl.nn2.activation.Tanh;
import dl.nn2.init.Xavier;
import dl.nn2.model.NNModel;

public class ConvOpTest {

	@Before
	public void setUp() throws Exception {
	}

	@After
	public void tearDown() throws Exception {
	}

	// @Test
	public void test1() {
		Xavier.debug = 1d;
		Kernel k = new Kernel(2, new Xavier(), "test");
		RealMatrix data = MatrixUtils
				.createRealMatrix(new double[][] { { 1d, 1d, 1d }, { 1d, 1d, 1d }, { 1d, 1d, 1d }, });
		MatrixDataEdge r = new MatrixDataEdge("", MatrixUtils.createRealMatrix(2, 2));
		k.conv(1, new MatrixDataEdge("", data), r);
		String s = MatrixDataEdge.pretty0(r.asMat(0));
		System.out.println(s);

		r = new MatrixDataEdge("", MatrixUtils.createRealMatrix(3, 3));
		k.conv(1, new MatrixDataEdge("", data), r);
		s = MatrixDataEdge.pretty0(r.asMat(0));
		System.out.println(s);

		data = MatrixUtils.createRealMatrix(new double[][] { { 1d, 1d }, { 1d, 1d } });
		r = new MatrixDataEdge("", MatrixUtils.createRealMatrix(3, 3));
		k.conv(1, new MatrixDataEdge("", data), r);
		s = MatrixDataEdge.pretty0(r.asMat(0));
		System.out.println(s);

	}

	// @Test
	public void test2() {
		double[][] d = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } };
		RealMatrix m = MatrixUtils.createRealMatrix(d);
		Kernel k = new Kernel(3, new Xavier(), "test");
		k.w.update(m);
		System.out.println(MatrixDataEdge.pretty0(m));
		k.rotate180(m);
		k.w.update(m);
		System.out.println(MatrixDataEdge.pretty0(m));
	}

	// @Test
	public void test3() {
		Xavier.debug = 1d;
		ConvOp conv = new ConvOp(2, 1, 1, new int[] { 4, 4 }, new int[] { 3, 3 }, false);

		Kernel l1 = new Kernel(2, new Xavier(), "L1");
		GateOp tanh = new GateOp(new Tanh(), true, "L1_Tanh");
		Kernel l2 = new Kernel(2, new Xavier(), "L2");

		MatrixDataEdge d1 = new MatrixDataEdge("", MatrixUtils.createRealMatrix(
				new double[][] { { 0, 0, 0, -1 }, { -1, 0, -1, 0 }, { -1, 0, -1, -1 }, { 1, 0, -1, -1 }, }));

		MatrixDataEdge o1 = new MatrixDataEdge("", MatrixUtils.createRealMatrix(3, 3));
		l1.conv(1, d1, o1);
		System.out.println(MatrixDataEdge.pretty0(o1.asMat(0)));

		System.out.println("==== check ====");
		GroupComputation g1 = new GroupComputation(conv, tanh);
		o1 = g1.eval(d1);
		System.out.println(MatrixDataEdge.pretty0(o1.asMat(0)));

		System.out.println("==== check2 ====");
		ConvOp conv2 = new ConvOp(2, 1, 1, new int[] { 2, 2 }, new int[] { 2, 2 }, false);
		GroupComputation g2 = new GroupComputation(conv2, tanh);
		MatrixDataEdge o2 = g2.eval(o1);
		System.out.println(MatrixDataEdge.pretty0(o2.asMat(0)));

	}

	// @Test
	public void test4() {
		Xavier.debug = 1d;
		ConvOp conv;
		// back propagation start here
		RealMatrix dLda = MatrixUtils
				.createRealMatrix(new double[][] { { 0.0005879d, 0.0005879d }, { 0.0005879d, 0.0005879d }, });
		RealMatrix dadz = MatrixUtils
				.createRealMatrix(new double[][] { { 0.00401306d, 0.00251779d }, { 0.01222816d, 0.00156579d }, });
		RealMatrix dd = MatrixUtils.createRealMatrix(2, 2);
		for (int i = 0; i < dLda.getRowDimension(); i++) {
			RealVector v = dLda.getRowVector(i);
			v = dadz.getRowVector(i).ebeMultiply(v);
			dd.setRowVector(i, v);
		}
		System.out.println(dd);

		RealMatrix dLdz = MatrixUtils
				.createRealMatrix(new double[][] { { 1.38153198, 1.38170182 }, { 1.38171926, 1.38177662 }, });
		RealMatrix expect = MatrixUtils.createRealMatrix(new double[][] { { 0.0690766, 0.15197701, 0.08290211 },
				{ 0.1657932, 0.35923367, 0.19344274 }, { 0.09672035, 0.2072619, 0.11054213 }, });

		conv = new ConvOp(2, 1, 1, new int[] { 2, 2 }, new int[] { 3, 3 }, true);
		conv.filter.getKernelGroup(0)[0].w
				.update(MatrixUtils.createRealMatrix(new double[][] { { .05, .06 }, { .07, .08 }, }));

		MatrixDataEdge d1 = conv.eval(new MatrixDataEdge("t", dLdz));
		System.out.println(MatrixDataEdge.pretty0(d1.asMat(0)));
	}

	// @Test
	public void test5() {
		Reshape shape = new Reshape(new int[] { 2, 2 });
		Reshape shape2 = new Reshape(new int[] { 1, 4 });
		Reshape shape3 = new Reshape(new int[] { 4, 1 });
		MatrixDataEdge d = new MatrixDataEdge("", MatrixUtils.createRealMatrix(new double[][] { { 1, 2, 3, 4 } }));
		MatrixDataEdge r = shape.eval(d);
		System.out.println(MatrixDataEdge.pretty0(r.asMat(0)));
		r = shape2.eval(r);
		System.out.println(MatrixDataEdge.pretty0(r.asMat(0)));
		r = shape3.eval(r);
		System.out.println(MatrixDataEdge.pretty0(r.asMat(0)));

		Reshape shape4 = new Reshape(new int[] { -1, 4 });
		r = shape4.eval(r);
		System.out.println(MatrixDataEdge.pretty0(r.asMat(0)));
		Reshape shape5 = new Reshape(new int[] { 4, -1 });
		r = shape5.eval(r);
		System.out.println(MatrixDataEdge.pretty0(r.asMat(0)));
	}

//	@Test
	public void test6() {
		NNModel.resetLogging();
		NNModel.setLoggingDebugMode(true);
		TracedComputation.debugLevel(2);
		// RMSPropOptimizer.debugLevel(2);
		VarOp var1 = new VarOp("L1_Z", "L1_ff_saveZ");
		VarOp var2 = new VarOp("L2_Z", "L2_ff_saveZ");
		VarOp var3 = new VarOp("L3_Z", "L3_ff_saveZ");
		MatrixDataEdge data1 = new MatrixDataEdge("data1", MatrixUtils.createRealMatrix(
				new double[][] { { 0, 0, 0, -1 }, { -1, 0, -1, 0 }, { -1, 0, -1, -1 }, { 1, 0, -1, -1 }, }));
		ConvOp l1 = new ConvOp(2, 1, 1, new int[] { 4, 4 }, new int[] { 3, 3 }, false);
		l1.filter.getKernelGroup(0)[0].w
				.update(MatrixUtils.createRealMatrix(new double[][] { { .01, .02 }, { .03, .04 }, }));
		GateOp tanh1 = new GateOp(new Tanh(), true, "L1");
		ConvOp l2 = new ConvOp(2, 1, 1, new int[] { 3, 3 }, new int[] { 2, 2 }, false);
		l2.filter.getKernelGroup(0)[0].w
				.update(MatrixUtils.createRealMatrix(new double[][] { { .05, .06 }, { .07, .08 }, }));
		GateOp tanh2 = new GateOp(new Tanh(), true, "L2");
		Reshape shape = new Reshape(new int[] { -1, 1 });
		// BiasedOp biased = new BiasedOp(true, "L2");
		MatrixDataEdge w = new MatrixDataEdge("", MatrixUtils.createRealMatrix(new double[][] { { 1, 1, 1, 1 }, }));
		MulOp mul = new MulOp(w, false, false, false, "L3");
		GateOp tanh3 = new GateOp(new Tanh(), true, "L3");
		GroupComputation g1 = new GroupComputation(l1, var1, tanh1, l2, var2, tanh2, shape, mul, var3, tanh3);

		MatrixDataEdge ret1 = g1.eval(data1);
		System.out.println("feed forward value: \n" + MatrixDataEdge.pretty0(ret1.asMat(0)));

		System.out.println(" back propagation begin ...... ");

		MulVarOp d_tanh1 = new MulVarOp(new VarGateOp(new Tanh(), false, "ff_bp", var1.getVar()), false,
				"dLdz_mul_dzdy");

		MulVarOp d_tanh2 = new MulVarOp(new VarGateOp(new Tanh(), false, "ff_bp", var2.getVar()), false,
				"dLdz_mul_dzdy");
		Reshape d_shape = new Reshape(new int[] { 2, 2 });
		// BiasedOp biased = new BiasedOp(true, "L2");
		MulOp d_mul = new MulOp(w, true, false, false, "L3");
		MulVarOp d_tanh3 = new MulVarOp(new VarGateOp(new Tanh(), false, "ff_bp", var3.getVar()), false,
				"dLdz_mul_dzdy");
		GroupComputation d_g = new GroupComputation(d_tanh3, d_mul, d_shape, d_tanh2, l2.rotate(), d_tanh1,
				l1.rotate());
		MatrixDataEdge dL = new MatrixDataEdge("", MatrixUtils.createRealMatrix(new double[][] { { 1.37294995 } }));
		ret1 = d_g.eval(dL);
		System.out.println(MatrixDataEdge.pretty0(ret1.asMat(0)));
	}

	@Test
	public void testMultiChannelMultiOut() {
		Xavier.debug = 0.1d;
		ConvOp conv = new ConvOp(2, 2, 3, new int[] { 3, 3 }, new int[] { 2, 2 }, false);
		RealMatrix in1 = MatrixUtils.createRealMatrix(new double[][] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } });
		RealMatrix in2 = MatrixUtils.createRealMatrix(new double[][] { { 1, 1, 1 }, { 1, 1, 1 }, { 1, 1, 1 } });
		MatrixDataEdge img = new MatrixDataEdge("image");
		img.addToMatList(in1);
		img.addToMatList(in2);
		MatrixDataEdge r = conv.eval(img);
		for (RealMatrix dd : r.asMatList()) {
			System.out.println(MatrixDataEdge.pretty0(dd));
		}

		ConvOp conv2 = conv.rotate();
		MatrixDataEdge r2 = conv2.eval(r);
		for (RealMatrix dd : r2.asMatList()) {
			System.out.println(MatrixDataEdge.pretty0(dd));
		}
	}

}
