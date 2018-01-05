package dl.alphabeta;

import java.util.HashSet;
import java.util.Set;

public class Decision {

	public int id;
	public boolean isMax;

	public Integer score;
	public int alpha = Integer.MIN_VALUE;
	public int beta = Integer.MAX_VALUE;

	public Integer parent;
	public Integer[] successors;
	public Set<Integer> pruning = new HashSet<Integer>();

	public Decision(int id, boolean isMax, int parent) {
		this.id = id;
		this.isMax = isMax;
		this.parent = parent;
	}

}
