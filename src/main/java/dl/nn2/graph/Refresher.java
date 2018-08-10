package dl.nn2.graph;

public class Refresher {

	MatrixDataEdge cached;

	public void writeVar(MatrixDataEdge d) {
		this.cached = d;
	}

	public MatrixDataEdge readVar() {
		return cached;
	}

}
