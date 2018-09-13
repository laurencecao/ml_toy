package dl.nn2.layer;

import org.apache.commons.lang3.StringUtils;

import dl.nn2.activation.GateFunction;
import dl.nn2.activation.Sigmoid;
import dl.nn2.activation.Tanh;

public class DenseLayer extends AbstractCompGraphLayer {

	protected String activationName = "sigmoid";

	public void setActivationName(String activationName) {
		this.activationName = activationName;
	}

	public DenseLayer(int in, int out, String name) {
		init(in, out, name);
	}

	@Override
	protected String typeName() {
		return DenseLayer.class.getSimpleName();
	}

	@Override
	protected GateFunction getActivationFunction() {
		if (StringUtils.compareIgnoreCase(activationName, "sigmoid") == 0) {
			return new Sigmoid();
		}
		if (StringUtils.compareIgnoreCase(activationName, "tanh") == 0) {
			return new Tanh();
		}
		throw new IllegalArgumentException("unknown activation: " + activationName);
	}

}
