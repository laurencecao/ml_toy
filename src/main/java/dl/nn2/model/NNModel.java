package dl.nn2.model;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.log4j.ConsoleAppender;
import org.apache.log4j.Level;
import org.apache.log4j.LogManager;
import org.apache.log4j.PatternLayout;
import org.math.plot.utils.FastMath;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import dl.nn2.graph.Computation;
import dl.nn2.graph.GroupComputation;
import dl.nn2.graph.MatrixDataEdge;
import dl.nn2.graph.Mul2Op;
import dl.nn2.graph.ScalarOp;
import dl.nn2.layer.AbstractCompGraphLayer;
import dl.nn2.loss.LossComp;
import dl.nn2.optimizer.RMSPropOptimizer;

public class NNModel {

	public static void resetLogging() {
		ConsoleAppender ca = new ConsoleAppender(new PatternLayout(PatternLayout.TTCC_CONVERSION_PATTERN),
				"System.out");
		ca.setName("console");
		LogManager.resetConfiguration();
		LogManager.getRootLogger().addAppender(ca);
		LogManager.getRootLogger().setLevel(Level.ERROR);
		LogManager.getLogger(NNModel.class).setLevel(Level.INFO);
	}

	final static Logger logger = LoggerFactory.getLogger(NNModel.class);

	protected List<AbstractCompGraphLayer> layers;

	protected List<Pair<GroupComputation, GroupComputation>> cgs;

	protected RMSPropOptimizer optimizer;
	protected String lossName;

	protected double learningRate;
	protected double decayRate;

	protected double error;
	protected int epoch;

	public NNModel(double lr, double dr) {
		layers = new ArrayList<>();
		// cgs = new ArrayList<>();
		learningRate = lr;
		decayRate = dr;
		// optimizer = new SimpleGradientDescend(lr, dr);
		optimizer = new RMSPropOptimizer(lr, dr);
	}

	void valid() {
		if (cgs == null) {
			throw new RuntimeException("Please compile model first!");
		}
	}

	public void setLossName(String name) {
		this.lossName = name;
	}

	public void addLayer(AbstractCompGraphLayer layer) {
		if (!layers.isEmpty()) {
			AbstractCompGraphLayer last = layers.get(layers.size() - 1);
			last.setNextLayer(layer);
		}
		layers.add(layer);
	}

	public void epochIncr(int i) {
		epoch += i;
	}

	public void compile() {
		cgs = new ArrayList<>();
		cgs.clear();
		for (int i = 0; i < layers.size(); i++) {
			AbstractCompGraphLayer l = layers.get(i);
			cgs.add(l.build());
		}
	}

	public double learning(double epsilon, List<RealMatrix> input, List<RealMatrix> target, int info, int iter) {
		compile();
		valid();
		NNDebuger debugger = new NNDebuger(layers);
		if (logger.isDebugEnabled()) {
			logger.debug(debugInfo());
		}
		double loss = 0d;
		double cur = 0d;
		int epoch = 0;
		while (true) {
			epoch++;
			for (int i = 0; i < input.size(); i++) {
				// int i = ThreadLocalRandom.current().nextInt(input.size());
				RealMatrix x = input.get(i);
				RealMatrix t = target.get(i);
				opt(x, t);
				if (logger.isDebugEnabled()) {
					debugger.snapshot(this);
				}
			}
			cur = estimateLoss(input, target);
			if (info > 0 && epoch % info == 0) {
				String border = "**************************************";
				System.out.println(border + "Learning epoch checkpoint: " + epoch + border);
			}
			if (FastMath.abs(cur - loss) < epsilon) {
				System.out.println("{" + epoch + "} epsilon ..... " + FastMath.abs(cur - loss));
				break;
			}
			loss = cur;
			if (iter > 0 && epoch >= iter) {
				break;
			}
		}
		debugger.printHistory(0, debugger.getHistorySize());
		loss = cur;
		return loss;
	}

	public void opt(RealMatrix data, RealMatrix dest) {
		if (logger.isDebugEnabled()) {
			logger.debug("Optimization Beging ...... ");
			logger.debug("1. Building FF,BP,GRAD Chain ...... ");
		}
		GroupComputation[] chain = buildChain(cgs);

		GroupComputation ff = chain[0];
		GroupComputation bp = chain[1];

		MatrixDataEdge input = new MatrixDataEdge("input", data);
		MatrixDataEdge target = new MatrixDataEdge("output", dest);

		if (logger.isDebugEnabled()) {
			logger.debug("2. Feed Forward Stage ...... ");
		}
		// feed forward stage
		MatrixDataEdge output = ff.eval(input);

		if (logger.isDebugEnabled()) {
			logger.debug("3. Loss computation Stage ...... ");
		}
		// diff loss
		LossComp loss = LossComp.create(lossName, target, true);

		if (logger.isDebugEnabled()) {
			logger.debug("4. Last layer dL/dy computation Stage ...... ");
		}
		// last differential dL/dy
		MatrixDataEdge dLdy = loss.eval(output);

		if (logger.isDebugEnabled()) {
			logger.debug("5. Back Propagation Stage ...... ");
		}
		// back propagation stage
		bp.eval(dLdy);

		if (logger.isDebugEnabled()) {
			logger.debug("6. Weights gradient updating Stage ...... ");
		}
		// update weights by gradient stage
		updateWeights(chain, input);
		if (logger.isDebugEnabled()) {
			logger.debug("Optimization Done ...... ");
			logger.debug(debugInfo());
		}
	}

	GroupComputation[] buildChain(List<Pair<GroupComputation, GroupComputation>> comps) {
		valid();
		List<Computation> ff = new ArrayList<Computation>();
		List<Computation> bp = new ArrayList<Computation>();
		for (Pair<GroupComputation, GroupComputation> c : comps) {
			ff.add(c.getLeft());
			bp.add(c.getRight());
		}

		Computation[] ops = ff.toArray(new Computation[ff.size()]);
		GroupComputation r0 = new GroupComputation("FeedForward", ops);

		ops = bp.toArray(new Computation[bp.size()]);
		GroupComputation r1 = new GroupComputation("BackForward", ops);
		r1.setReversed(true);

		return new GroupComputation[] { r0, r1 };
	}

	public void updateWeights(GroupComputation[] chain, MatrixDataEdge input) {
		GroupComputation ff = chain[0];
		GroupComputation bp = chain[1];
		MatrixDataEdge dLdy;
		int sz = input.asMat(0).getColumnDimension();

		if (logger.isDebugEnabled()) {
			logger.debug("6.1 Building Gradient computations Stage ...... ");
		}

		List<Computation> ops = new ArrayList<>();
		for (int i = ff.size() - 1; i >= 0; i--) {
			// (dL/dw) = (dL/dy) * (dy/dz) * (dz/dw)
			// (dz/dw) = [Σ(w * x)]'
			// notice: Layer(y) == Layer(z) == {Layer(x) + 1}
			String title = bp.getComputation(i).name();
			dLdy = bp.getComputationOutput(i);
			MatrixDataEdge x = null;
			if (i - 1 < 0) {
				x = input;
			} else {
				x = ff.getComputationOutput(i - 1);
			}
			Mul2Op mul = new Mul2Op(true, true, dLdy, x, "Mul2Op_#" + title + "#");
			ScalarOp scalar = new ScalarOp(sz, '/', "AvgOp_#" + title + "#");
			GroupComputation g = new GroupComputation("dw_#" + title + "#", mul, scalar);
			g.setAttach(((GroupComputation) bp.getComputation(i)).getAttach());
			ops.add(g);
		}

		GroupComputation grad = new GroupComputation(ops.toArray(new Computation[ops.size()]));
		grad.eval(null);

		if (logger.isDebugEnabled()) {
			logger.debug("6.2 Delta weights evaluation by Optimizer Stage ...... ");
		}
		for (int i = 0; i < grad.size(); i++) {
			GroupComputation cg = (GroupComputation) grad.getComputation(i);
			AbstractCompGraphLayer l = (AbstractCompGraphLayer) cg.getAttach();
			MatrixDataEdge g = grad.getComputationOutput(i);
			g.setId(l.getWeights().getId());
			MatrixDataEdge dG = optimizer.eval(g, g.getName());

			MatrixDataEdge.isSameShape(l.getWeights().asMat(0), dG.asMat(0), g.getName() + "@Layer_" + i);

			RealMatrix _w = l.getWeights().asMat(0).add(dG.asMat(0));

			if (logger.isDebugEnabled()) {
				logger.debug("6.3 Weights updating operation Stage ...... #" + i + "#");
			}
			l.updateWeights(new MatrixDataEdge("", _w));
		}
	}

	public double estimateLoss(List<RealMatrix> input, List<RealMatrix> target) {
		compile();
		valid();
		GroupComputation[] chain = buildChain(cgs);
		GroupComputation ff = chain[0];

		double ret = 0d;
		for (int i = 0; i < input.size(); i++) {
			RealMatrix in = input.get(i);
			RealMatrix t = target.get(i);
			RealVector one = MatrixUtils.createRealVector(new double[in.getColumnDimension()]);
			one.set(1);
			MatrixDataEdge output = ff.eval(new MatrixDataEdge("input", in));
			LossComp loss = LossComp.create(lossName, new MatrixDataEdge("target", t), false);
			MatrixDataEdge theloss = loss.eval(output);
			ret += theloss.asDouble(0);
		}
		return ret;
	}

	public List<RealMatrix> predict(List<RealMatrix> input) {
		compile();
		valid();
		GroupComputation[] chain = buildChain(cgs);
		GroupComputation ff = chain[0];

		List<RealMatrix> ret = new ArrayList<RealMatrix>();
		for (int i = 0; i < input.size(); i++) {
			RealMatrix in = input.get(i);
			MatrixDataEdge output = ff.eval(new MatrixDataEdge("input", in));
			ret.add(output.asMat(0));
		}
		return ret;
	}

	public static void setLoggingDebugMode(boolean enable) {
		if (enable) {
			LogManager.getLogger(NNModel.class).setLevel(Level.DEBUG);
			LogManager.getLogger(NNDebuger.class).setLevel(Level.DEBUG);
		} else {
			LogManager.getLogger(NNModel.class).setLevel(Level.INFO);
			LogManager.getLogger(NNDebuger.class).setLevel(Level.INFO);
		}
	}

	public String debugInfo() {
		StringBuilder sb = new StringBuilder();
		sb.append('\n');
		for (AbstractCompGraphLayer l : layers) {
			sb.append(l.debugInfo()).append("\n");
		}
		return sb.toString();
	}

}
