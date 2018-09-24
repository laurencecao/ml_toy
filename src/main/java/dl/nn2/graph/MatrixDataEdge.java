package dl.nn2.graph;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.UUID;

import org.apache.commons.math3.linear.RealMatrix;

/**
 * version maintained data
 * 
 * @author Laurence Cao
 * @date 2018年8月3日
 *
 */
public class MatrixDataEdge {

	protected String id;
	protected String name;
	protected List<RealMatrix> dList = new ArrayList<>();
	protected List<RealMatrix> dListHistory = new ArrayList<>();

	protected Refresher updater;

	public void setUpdater(Refresher updater) {
		this.updater = updater;
	}

	public void setId(String id) {
		this.id = id; // caution: for RMSProp Optimizer only
	}

	public MatrixDataEdge(String name) {
		this(UUID.randomUUID().toString(), name, new ArrayList<RealMatrix>());
	}

	public MatrixDataEdge(String name, RealMatrix data) {
		this(UUID.randomUUID().toString(), name, data);
	}

	public MatrixDataEdge(String name, MatrixDataEdge data) {
		this(UUID.randomUUID().toString(), name, data.dList);
	}

	public MatrixDataEdge(String id, String name, RealMatrix data) {
		this(id, name, Arrays.asList(data));
	}

	public MatrixDataEdge(String id, String name, List<RealMatrix> data) {
		this.name = name;
		for (RealMatrix m : data) {
			if (m != null) {
				this.dList.add(m.copy());
			}
		}
		this.id = id;
	}

	public String getId() {
		return this.id;
	}

	public String getName() {
		return name;
	}

	public int[] shape() {
		RealMatrix data = this.dList.get(0);
		return new int[] { data.getRowDimension(), data.getColumnDimension() };
	}

	public RealMatrix asMat(int version) {
		if (version > 1) {
			throw new IllegalArgumentException("version support [0,1]");
		}
		if (updater != null) {
			MatrixDataEdge d = updater.readVar();
			if (d != null) {
				update(d);
			}
		}
		if (version == 1) {
			return this.dListHistory.get(0);
		}
		return this.dList.get(0);
	}

	public List<RealMatrix> asMatList() {
		if (updater != null) {
			MatrixDataEdge d = updater.readVar();
			if (d != null) {
				update(d);
			}
		}
		return dList;
	}

	public void addToMatList(RealMatrix m) {
		this.dList.add(m);
	}

	public double asDouble(int version) {
		if (version > 1) {
			throw new IllegalArgumentException("version support [0,1]");
		}
		if (updater != null) {
			MatrixDataEdge d = updater.readVar();
			if (d != null) {
				update(d);
			}
		}
		if (version == 1) {
			return this.dListHistory.get(0).getEntry(0, 0);
		}
		return this.dList.get(0).getEntry(0, 0);
	}

	void backup(boolean clearCurrent) {
		this.dListHistory.clear();
		if (clearCurrent) {
			this.dListHistory.addAll(this.dList);
			this.dList.clear();
		} else {
			for (RealMatrix m : this.dList) {
				this.dListHistory.add(m.copy());
			}
		}
	}

	public void update(RealMatrix data) {
		backup(true);
		this.dList.add(data.copy());
	}

	public void update(RealMatrix data, int idx) {
		backup(false);
		this.dList.set(idx, data.copy());
	}

	public void update(MatrixDataEdge data) {
		backup(true);
		for (RealMatrix m : data.dList) {
			this.dList.add(m.copy());
		}
		// isSameShape(this.data, this._data, this.name);
	}

	public String pretty() {
		String msg = "[id=" + id + "   	name=" + name + "	size=" + this.dList.size() + "]\n";
		msg += "version_0: \n" + prettys(this.dList);
		msg += "version_1: \n" + prettys(this.dListHistory);
		return msg;
	}

	static public String prettys(List<RealMatrix> lst) {
		StringBuilder sb = new StringBuilder();
		for (RealMatrix m : lst) {
			sb.append(pretty0(m));
		}
		return sb.toString();
	}

	static public String pretty0(RealMatrix m) {
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

	public static void isSameShape(RealMatrix v0, RealMatrix v1, String msg) {
		if (v0 == null || v1 == null) {
			return;
		}
		boolean b1 = v0.getColumnDimension() != v1.getColumnDimension();
		boolean b2 = v0.getRowDimension() != v1.getRowDimension();
		if (b1 || b2) {
			String m1 = "data_1: \n" + pretty0(v1);
			String m0 = "data_0: \n" + pretty0(v0);
			throw new RuntimeException(msg + " data not same shape: \n" + m1 + " =====> \n" + m0);
		}
	}

}
