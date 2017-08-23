package dl.opt;

import com.beust.jcommander.Parameter;

public class DLParameter {

	@Parameter(names = { "-err" }, description = "error episode")
	public double err = 0.001d;

	@Parameter(names = "-rate", description = "learning rate")
	public double rate = 0.01;

	@Parameter(names = "-debug", description = "Debug mode loop")
	public int debug = 100000;

	@Parameter(names = "-hidden", description = "hidden nodes")
	public int hidden = 100;

	@Parameter(names = "-name", description = "run machine")
	public String name = "digital";

	@Parameter(names = "-k", description = "CD_K")
	public int k = 2;

}
