package utils;

import org.apache.commons.math3.linear.RealMatrix;

public class MatrixUtil {

	static public String pretty(RealMatrix m) {
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

	static public String pretty(double[] data) {
		if (data == null) {
			return "\n";
		}
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < data.length; i++) {
			sb.append(String.format("%10.5f", data[i])).append("  ");
			sb.append('\n');
		}
		return sb.toString();
	}

}
