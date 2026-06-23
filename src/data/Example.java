package data;

public class Example {
	public final double[][] image; // HxW grayscale standardized
    public final int label;

    public Example(double[][] image, int label) { 
    	this.image = image; 
    	this.label = label; 
    	}

}
