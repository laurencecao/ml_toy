package dl.nn2.layer;

public class DenseLayer extends AbstractCompGraphLayer {

	public DenseLayer(int in, int out, String name) {
		super(in, out, name);
	}

	@Override
	protected String typeName() {
		return DenseLayer.class.getSimpleName();
	}

}
