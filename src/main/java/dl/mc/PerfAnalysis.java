package dl.mc;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.math.plot.utils.FastMath;

import net.sf.cglib.proxy.Enhancer;
import net.sf.cglib.proxy.MethodInterceptor;
import net.sf.cglib.proxy.MethodProxy;

public class PerfAnalysis {

	public static void main(String[] args) {
		Map<String, Double[]> factor = new HashMap<>();
		factor.put("A", new Double[] { 20d, 5d });
		factor.put("B", new Double[] { 30d, 10d });
		factor.put("C", new Double[] { 25d, 3d });
		factor.put("D", new Double[] { 50d, 30d });
		factor.put("E", new Double[] { 10d, 5d });
		factor.put("F", new Double[] { 40d, 20d });

		// 用scale过的时长来模拟sleep的时间，会与真实情况有误差，如果有时间等待，可将scale设为1
		ReqBase app = createApp(factor, 0.1d);
		List<Resp> result = runner(app, 100);
		printResult(result);

		tuning(factor, 0.1d);
	}

	static SummaryStatistics printResult(List<Resp> re) {
		SummaryStatistics stats = new SummaryStatistics();
		for (int i = 0; i < re.size(); i++) {
			Resp r = re.get(i);
			stats.addValue(r.elapsed);
			// System.out.println(r.info + " --> " + r.elapsed + "ms");
		}
		System.out.println(stats);
		return stats;
	}

	static List<Resp> runner(ReqBase app, int loop) {
		long ts = System.currentTimeMillis();
		ArrayList<Resp> ret = new ArrayList<>();
		for (int i = 0; i < loop; i++) {
			Resp r = app.request("");
			ret.add(r);
		}
		ts = System.currentTimeMillis() - ts;
		System.out.println("Total loop[" + loop + "] time elapsed: " + ts + "ms");
		return ret;
	}

	static SummaryStatistics printResultSimu(List<Map<String, Double>> re, String name, String memo) {
		SummaryStatistics stats = new SummaryStatistics();
		for (int i = 0; i < re.size(); i++) {
			Map<String, Double> r = re.get(i);
			stats.addValue(r.get(name));
			// System.out.println(r.info + " --> " + r.elapsed + "ms");
		}
		System.out.println("................ " + memo + " begin ................");
		System.out.println(stats);
		System.out.println("................ " + memo + " end ................");
		return stats;
	}

	static List<Map<String, Double>> simuRun(SimulationComputerGraph g, int loop) {
		Map<String, Double> reg = g.getRegister();
		List<Map<String, Double>> ret = new ArrayList<>();
		long ts = System.currentTimeMillis();
		for (int i = 0; i < loop; i++) {
			reg.clear();
			g.compute();
			Map<String, Double> m = new HashMap<>();
			m.putAll(reg);
			ret.add(m);
		}
		ts = System.currentTimeMillis() - ts;
		System.out.println("Simulation total loop[" + loop + "] time elapsed: " + ts + "ms");
		return ret;
	}

	static ReqBase createApp(Map<String, Double[]> factor, double scale) {
		ReqBase a = ReqBase.create("A", factor.get("A")[0] * scale, factor.get("A")[1] * scale);
		ReqBase b = ReqBase.create("B", factor.get("B")[0] * scale, factor.get("B")[1] * scale);
		ReqBase c = ReqBase.create("C", factor.get("C")[0] * scale, factor.get("C")[1] * scale);
		ReqBase d = ReqBase.create("D", factor.get("D")[0] * scale, factor.get("D")[1] * scale);
		ReqBase e = ReqBase.create("E", factor.get("E")[0] * scale, factor.get("E")[1] * scale);
		ReqBase f = ReqBase.create("F", factor.get("F")[0] * scale, factor.get("F")[1] * scale);

		// a -> ((b -> (c, e)), (d -> ((e -> f), c)))
		a.addChildren(b);
		a.addChildren(d);
		b.addChildren(c);
		b.addChildren(e);
		d.addChildren(e);
		d.addChildren(c);
		e.addChildren(f);

		return a;
	}

	static SimulationComputerGraph createSimulationApp(Map<String, Double> reg,
			Map<String, SimulationComputerGraph> inst, Map<String, Double[]> factor) {
		SimulationComputerGraph a = new SimulationComputerGraph("A", factor.get("A")[0], factor.get("A")[1], reg);
		SimulationComputerGraph b = new SimulationComputerGraph("B", factor.get("B")[0], factor.get("B")[1], reg);
		SimulationComputerGraph c = new SimulationComputerGraph("C", factor.get("C")[0], factor.get("C")[1], reg);
		SimulationComputerGraph d = new SimulationComputerGraph("D", factor.get("D")[0], factor.get("D")[1], reg);
		SimulationComputerGraph e = new SimulationComputerGraph("E", factor.get("E")[0], factor.get("E")[1], reg);
		SimulationComputerGraph f = new SimulationComputerGraph("F", factor.get("F")[0], factor.get("F")[1], reg);

		inst.put("A", a);
		inst.put("B", b);
		inst.put("C", c);
		inst.put("D", d);
		inst.put("E", e);
		inst.put("F", f);

		// a -> ((b -> (c, e)), (d -> ((e -> f), c)))
		a.addChildren(b);
		a.addChildren(d);
		b.addChildren(c);
		b.addChildren(e);
		d.addChildren(e);
		d.addChildren(c);
		e.addChildren(f);

		return a;
	}

	static void tuning(Map<String, Double[]> factor, double scale) {
		String[] K = { "A", "B", "C", "D", "E", "F" };
		Map<String, Double> reg = new HashMap<>();
		Map<String, SimulationComputerGraph> inst = new HashMap<>();
		SimulationComputerGraph app1 = createSimulationApp(reg, inst, factor);
		List<Map<String, Double>> res = simuRun(app1, 100000);
		SummaryStatistics origin = printResultSimu(res, "A", "origin");

		Map<String, Double[]> changed = new HashMap<String, Double[]>();
		SummaryStatistics[] result = new SummaryStatistics[K.length];
		for (int i = 0; i < K.length; i++) {
			String k = K[i];
			reg = new HashMap<>();
			inst = new HashMap<>();
			Map<String, Double[]> f = autoTune(factor, k, scale);
			changed.put(k, f.get(k));
			app1 = createSimulationApp(reg, inst, f);
			res = simuRun(app1, 1000);
			SummaryStatistics r = printResultSimu(res, "A", "dL/d" + k);
			result[i] = r;
		}

		double m0 = 0d;
		double m1 = Double.MAX_VALUE;
		String mk0 = "", mk1 = "";
		for (int i = 0; i < K.length; i++) {
			double a = origin.getMean();
			double b = result[i].getMean();
			m0 = FastMath.max(FastMath.abs(a - b), m0);
			mk0 = FastMath.abs(a - b) == m0 ? K[i] : mk0;
			m1 = FastMath.min(FastMath.abs(a - b), m1);
			mk1 = FastMath.abs(a - b) == m1 ? K[i] : mk1;
			Double[] ch = changed.get(K[i]);
			Double[] _mean = factor.get(K[i]);
			String msg = "mean: " + _mean[0] + " ==> " + ch[0] + " with sd: " + _mean[1] + " ==> " + ch[1];
			System.out.println("tuning '" + K[i] + "' " + msg + ";  " + a + "  ==> " + b + " ; reduced to: " + (b / a));
		}
		System.out.println("Max Reduced [" + mk0 + "]: " + m0 + "ms");
		System.out.println("Min Reduced [" + mk1 + "]: " + m1 + "ms");
	}

	static Map<String, Double[]> autoTune(Map<String, Double[]> factor, String K, double scale) {
		Map<String, Double[]> ret = new HashMap<>();
		ret.putAll(factor);
		Double[] f = factor.get(K);
		Double[] f1 = new Double[] { f[0] * (1 - scale), f[1] * (1 - scale) };
		if (f1[0] <= 0) {
			f1[0] = 1e7;
		}
		if (f1[1] <= 0) {
			f1[1] = 1e7;
		}
		ret.put(K, f1);
		return ret;
	}

}

class Resp {

	String info;
	long elapsed;

}

abstract class ReqBase {

	final static Enhancer eh = new Enhancer();
	static {
		eh.setSuperclass(ReqBase.class);
		eh.setCallback(new PassHandler());
	}

	protected NormalDistribution dist;
	protected double mean;
	protected double sd;
	protected List<ReqBase> children = new ArrayList<>();
	protected String name;

	public void init() {
		dist = new NormalDistribution(mean, sd);
	}

	public void setMean(double mean) {
		this.mean = mean;
	}

	public void setSd(double sd) {
		this.sd = sd;
	}

	public void setName(String name) {
		this.name = name;
	}

	public void addChildren(ReqBase node) {
		this.children.add(node);
	}

	protected String doWork(String msg) {
		try {
			double el = Math.abs(dist.sample());
			Thread.sleep(Double.valueOf(el).longValue());
		} catch (InterruptedException e) {
		}
		return msg + ";" + name;
	}

	public Resp request(String msg) {
		long ts = System.currentTimeMillis();
		Resp ret = new Resp();
		ret.info = doWork(msg);
		for (ReqBase node : this.children) {
			node.request(ret.info);
		}
		ts = System.currentTimeMillis() - ts;
		ret.elapsed = ts;
		return ret;
	}

	public static ReqBase create(String name, double mean, double sd) {
		ReqBase ret = (ReqBase) eh.create();
		ret.setMean(mean);
		ret.setSd(sd);
		ret.setName(name);
		ret.init();
		return ret;
	}

	static class PassHandler implements MethodInterceptor {

		@Override
		public Object intercept(Object obj, Method method, Object[] args, MethodProxy proxy) throws Throwable {
			return proxy.invokeSuper(obj, args);
		}

	}

}

class SimulationComputerGraph {

	NormalDistribution dist;
	double mean;
	double sd;
	String name;
	List<SimulationComputerGraph> chain = new ArrayList<>();
	Map<String, Double> register;

	public Map<String, Double> getRegister() {
		return register;
	}

	public SimulationComputerGraph(String name, double mean, double sd, Map<String, Double> reg) {
		this.name = name;
		this.mean = mean;
		this.sd = sd;
		this.dist = new NormalDistribution(mean, sd);
		this.register = reg;
	}

	public void addChildren(SimulationComputerGraph node) {
		chain.add(node);
	}

	public double compute() {
		double ts = Math.abs(dist.sample());
		register.put("_" + name, ts);
		double ret = ts;
		for (SimulationComputerGraph node : chain) {
			double elapsed = node.compute();
			ret += elapsed;
		}
		register.put(name, ret);
		return ret;
	}

}
