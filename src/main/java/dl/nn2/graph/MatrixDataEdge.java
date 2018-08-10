package dl.nn2.graph;

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
	protected RealMatrix data;
	protected RealMatrix _data;

	protected Refresher updater;

	public void setUpdater(Refresher updater) {
		this.updater = updater;
	}

	public MatrixDataEdge(String name, RealMatrix data) {
		this(UUID.randomUUID().toString(), name, data);
	}

	public void setId(String id) {
		this.id = id; // caution: for RMSProp Optimizer only
	}

	public MatrixDataEdge(String id, String name, RealMatrix data) {
		this.name = name;
		this.data = data;
		if (data != null) {
			this._data = data.copy();
		} else {
			this._data = null;
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
			return _data;
		}
		return data;
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
			return _data.getEntry(0, 0);
		}
		return data.getEntry(0, 0);
	}

	public void update(RealMatrix data) {
		this._data = this.data;
		this.data = data;
	}

	public void update(MatrixDataEdge data) {
		if (this.data != null) {
			this._data = this.data.copy();
		} else {
			this._data = null;
		}
		if (data.data != null) {
			this.data = data.data.copy();
		} else {
			this.data = null;
		}
		// isSameShape(this.data, this._data, this.name);
	}

	public String pretty() {
		String msg = "[id=" + id + "   	name=" + name + "]\n";
		msg += "version_0: \n" + pretty0(data);
		msg += "version_1: \n" + pretty0(_data);
		return msg;
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
