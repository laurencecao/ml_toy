package dataset;

import cern.colt.Arrays;

public class NewsCorpus {

	public Integer tag;
	public Integer[] words;

	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("tag ==> ").append(tag);
		sb.append("; words => ").append(Arrays.toString(words));
		return sb.toString();
	}

}
