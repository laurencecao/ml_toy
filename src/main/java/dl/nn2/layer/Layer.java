package dl.nn2.layer;

import org.apache.commons.math3.linear.RealMatrix;

public interface Layer {

	// feed forward, no activation
	RealMatrix feedforward(RealMatrix in);

	// activation only
	RealMatrix activation(RealMatrix z);

	RealMatrix deActivation(RealMatrix x);

	// compute gradient from loss: dLoss/dz
	RealMatrix dLdz(RealMatrix inData, RealMatrix label);

	// compute gradient from data
	RealMatrix dLdw(RealMatrix inData, RealMatrix grad);

	void updateWeights(RealMatrix grad);

	RealMatrix getWeights();

	RealMatrix getLastWeights();

	void setNextLayer(Layer next);

	void setPreLayer(Layer pre);

	Layer getNextLayer();

	Layer getPreLayer();

	String getName();

	int getIn();

	int getOut();

	String debugInfo();

}
