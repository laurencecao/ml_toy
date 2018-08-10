package dl.nn2.graph;

/**
 * just for demostration, so make it easy
 * <ol>
 * <li>all variable should be initialized at declaring time</li>
 * <li>no named variable binding</li>
 * <li>no delay variable binding</li>
 * </ol>
 * 
 * @see <a href=
 *      "http://www.cs.columbia.edu/~mcollins/cs4705-fall2017/slides/ff2-slides.pdf">Computational
 *      Graphs, and Backpropagation Michael Collins, Columbia University</a>
 * @see <a href="http://www.cs.columbia.edu/~mcollins/ff2.pdf">Computational
 *      Graphs, and Backpropagation - Columbia CS</a>
 * 
 * @author Laurence Cao
 * @date 2018年8月3日
 *
 */
public interface Computation {

	MatrixDataEdge eval(MatrixDataEdge data);

	MatrixDataEdge eval(MatrixDataEdge data, String rtMsg);

	String type();

	String name();

	String id();

	int[] inShape();

	int[] outShape();

}
