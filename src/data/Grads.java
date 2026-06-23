package data;

public class Grads implements java.io.Serializable{

	public final double[][] dA_prev;
	public final double[][] dW;
	public final double[][] db;
	
	public Grads() {
        this.dA_prev = new double[0][0];
        this.dW = new double[0][0];
        this.db = new double[0][0];
	}
	
	public Grads(double[][] dA_prev, double[][] dW, double[][] db) {
		this.dA_prev = dA_prev; 
		this.dW = dW; 
		this.db = db;
	}



}
