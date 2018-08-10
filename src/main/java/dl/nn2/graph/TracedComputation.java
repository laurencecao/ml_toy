package dl.nn2.graph;

import java.util.Arrays;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.log4j.Level;
import org.apache.log4j.LogManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class TracedComputation implements Computation {

	static protected boolean enabled = false;

	protected Logger logger = LoggerFactory.getLogger(TracedComputation.class);

	abstract protected MatrixDataEdge eval0(MatrixDataEdge data);

	@Override
	public String type() {
		return this.getClass().getSimpleName();
	}

	@Override
	public String id() {
		return toString();
	}

	public MatrixDataEdge eval(MatrixDataEdge data) {
		return eval(data, "");
	}

	public MatrixDataEdge eval(MatrixDataEdge data, String rtMsg) {
		MatrixDataEdge ret = null;
		try {
			ret = eval0(data);
		} catch (Exception e) {
			throw e;
		} finally {
			trace(rtMsg, data, ret);
		}
		return ret;
	}

	protected void trace(String rtMsg, MatrixDataEdge data, MatrixDataEdge ret) {
		trace(rtMsg, null, data, ret, null, 1);
	}

	protected void trace(String rtMsg, MatrixDataEdge w, MatrixDataEdge data, MatrixDataEdge ret, String extraInfo,
			int idx) {
		if (enabled && (logger.isTraceEnabled() || logger.isDebugEnabled())) {
			String name = type() + rtMsg + "   ";
			String wMsg = "";
			if (w != null) {
				int[] shapeW = { w.asMat(0).getRowDimension(), w.asMat(0).getColumnDimension() };
				wMsg = "weight=" + Arrays.toString(shapeW) + "; ";
			}
			wMsg += " ";
			int[] shapeIn = { -1, -1 };
			if (data != null) {
				shapeIn = new int[] { data.asMat(0).getRowDimension(), data.asMat(0).getColumnDimension() };
			}

			int[] shapeOut = { -1, -1 };
			if (ret != null) {
				shapeOut = new int[] { ret.asMat(0).getRowDimension(), ret.asMat(0).getColumnDimension() };
			}

			String s = "";
			String s1 = "";
			if (idx == 0) {
				s = " <=={runtime argument}== ";
				s1 = " <=={static initialized}== ";
			} else {
				s = " <=={static initialized}== ";
				s1 = " <=={runtime argument}== ";
			}

			if (logger.isTraceEnabled()) {
				String wMsg1 = " ";
				if (w != null) {
					RealMatrix wt = w.asMat(0);
					wMsg1 = wMsg + s + "\n" + pretty(wt);
				}
				RealMatrix in = null;
				if (data != null) {
					in = data.asMat(0);
				}
				RealMatrix out = ret == null ? null : ret.asMat(0);
				if (extraInfo == null) {
					extraInfo = "";
				}
				String msg = name + " : " + extraInfo + "\n" + wMsg1 + "\ndata before operation: "
						+ Arrays.toString(shapeIn) + s1 + "\n" + pretty(in) + "\ndata after operation: "
						+ Arrays.toString(shapeOut) + "\n" + pretty(out);
				logger.debug(msg + "\n");
			} else if (logger.isDebugEnabled()) {
				if (extraInfo == null) {
					extraInfo = "";
				}
				String msg = name + " : " + extraInfo + "  " + wMsg + Arrays.toString(shapeIn) + " ===> "
						+ Arrays.toString(shapeOut);
				logger.debug(msg + "\n");
			}
		}
	}

	public static String pretty(RealMatrix m) {
		if (m == null) {
			return "\n";
		}
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < m.getRowDimension(); i++) {
			double[] items = m.getRow(i);
			for (int j = 0; j < items.length; j++) {
				sb.append(String.format("%10.5f", items[j])).append("  ");
			}
			sb.append('\n');
		}
		return sb.toString();
	}

	public static String pretty(RealVector m) {
		if (m == null) {
			return "\n";
		}
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < m.getDimension(); i++) {
			sb.append(String.format("%10.5f", m.getEntry(i))).append("  ");
		}
		sb.append('\n');
		return sb.toString();
	}

	public static void debugLevel(int level) {
		switch (level) {
		case 1:
			enabled = true;
			LogManager.getLogger(TracedComputation.class).setLevel(Level.DEBUG);
			break;
		case 2:
			enabled = true;
			LogManager.getLogger(TracedComputation.class).setLevel(Level.TRACE);
			break;
		default:
			enabled = false;
		}
	}

}
