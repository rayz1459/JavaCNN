package data;

public class Params implements java.io.Serializable{
	
	public final double[][] W;
	public final double[][] b;
	
	public Params() {
        this.W = new double[0][0];
        this.b = new double[0][0];
	}
	
	public Params(double[][] W, double[][] b) {
		this.W = W; 
		this.b = b; 
	}

}
