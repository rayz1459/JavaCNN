package javann;

import data.Grads;
import data.GradsFull;
import data.ParamsFull;

public interface JavaNNInterface {
	double[][]zero_pad(double[][] arr, int pad);
	double[][] conv_forward(double[][] A_prev, double[][] W, double[][] b, int stride, int pad);
	public Grads conv_backward(double[][] dZ, double[][] A_prev, double[][] W, double[][] b, int stride, int pad);
	public double[][] relu(double[][] Z);
	public double[][] relu_backward(double[][] dA, double[][] Z);
	public double sigmoid(double z);
	public double[][] leakyRelu(double[][] Z, double alpha);
	public double[][] leakyRelu_backward(double[][] dA, double[][] Z, double alpha);
	public void compileModel(ParamsFull grads, String string);
	public ParamsFull importModel(String string);

}
