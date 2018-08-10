package dataset;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

public class Iris {

	final static String IRIS_PATH = "/Iris.csv";

	static List<IrisItem> _data = new ArrayList<IrisItem>();
	static List<String> labels = new ArrayList<String>();

	static {
		init(_data);
	}

	static void init(List<IrisItem> items) {
		try (InputStream is = WholeSale.class.getResourceAsStream(IRIS_PATH)) {
			BufferedReader br = new BufferedReader(new InputStreamReader(is));
			String ln = br.readLine(); // skip head line
			while ((ln = br.readLine()) != null) {
				ln = ln.trim();
				if (ln.startsWith("%") || ln.startsWith("#")) {
					continue;
				}
				String[] it = ln.split(",");
				double[] r = new double[it.length - 2];
				for (int i = 1; i < it.length - 1; i++) {
					r[i - 1] = Double.valueOf(it[i]);
				}
				IrisItem d = new IrisItem(Integer.valueOf(it[0]), r, it[5]);
				items.add(d);
				if (labels.indexOf(d.getY()) < 0) {
					labels.add(d.getY());
				}
			}
			br.close();
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	public static double[][] getData() {
		double[][] ret = new double[_data.size()][];
		for (int i = 0; i < _data.size(); i++) {
			IrisItem item = _data.get(i);
			ret[i] = item.getX();
		}
		return ret;
	}

	public static String[] getLabel() {
		String[] ret = new String[_data.size()];
		for (int i = 0; i < _data.size(); i++) {
			IrisItem item = _data.get(i);
			ret[i] = item.getY();
		}
		return ret;
	}

	public static int[] getLabelIdx() {
		String[] l = getLabel();
		int[] ret = new int[l.length];
		for (int i = 0; i < ret.length; i++) {
			ret[i] = getLabelIndex(l[i]);
		}
		return ret;
	}

	public static double[] getData(int i) {
		return _data.get(i).getX();
	}

	public static String getLabel(int i) {
		return _data.get(i).getY();
	}

	public static int getLabelIndex(String label) {
		return labels.indexOf(label);
	}

}

class IrisItem {

	int id;
	double sepalLengthCm;
	double sepalWidthCm;
	double petalLengthCm;
	double petalWidthCm;
	String species;

	IrisItem(int id, double[] x, String y) {
		this.id = id;
		this.sepalLengthCm = x[0];
		this.sepalWidthCm = x[1];
		this.petalLengthCm = x[2];
		this.petalWidthCm = x[3];
		this.species = y;
	}

	int getId() {
		return id;
	}

	double[] getX() {
		return new double[] { sepalLengthCm, sepalWidthCm, petalLengthCm, petalWidthCm };
	}

	String getY() {
		return species;
	}

	public double getSepalLengthCm() {
		return sepalLengthCm;
	}

	public double getSepalWidthCm() {
		return sepalWidthCm;
	}

	public double getPetalLengthCm() {
		return petalLengthCm;
	}

	public double getPetalWidthCm() {
		return petalWidthCm;
	}

	public String getSpecies() {
		return species;
	}

}
