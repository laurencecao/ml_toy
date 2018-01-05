package dl.alphabeta;

public class Decision {

	public int id;
	public boolean isMax;

	public Integer score;
	public int alpha = Integer.MIN_VALUE;
	public int beta = Integer.MAX_VALUE;

	public Integer parent;
	public Integer[] successors;

	public Decision(int id, boolean isMax, int parent) {
		this.id = id;
		this.isMax = isMax;
		this.parent = parent;
	}

}
