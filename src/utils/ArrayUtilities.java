package utils;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import data.Example;


public class ArrayUtilities implements ArrayUtilitiesInterface{
	
	public ArrayUtilities() {
		
	}
	
	
	public double[][] pad(double[][] arr, int padding) {
		int height = arr.length;
	    int width = arr[0].length;
	    int newHeight = height + 2 * padding;
	    int newWidth = width + 2 * padding;
	    
	    double[][] padded = new double[newHeight][newWidth];
	    
	    
	    
	    //left padding
	    for(int i = 0; i<newHeight; i++) {
	    	padded[i][0] = padding;

	    }
	    
	    //top padding
	    for(int i = 0; i<newWidth; i++) {
	    	padded[0][i] = padding;

	    }
	    
	    //right padding
	    for(int i = 0; i<newHeight; i++) {
	    	padded[i][newWidth-1] = padding;

	    }
		
	    //bottom padding
	    for(int i = 0; i<newWidth; i++) {
	    	padded[newHeight-1][i] = padding;

	    }
	    
	    //copy array
	    for (int i = 0; i < height; i++) { 
	    	for (int j = 0; j < width; j++) { 
	    		padded[i + padding][j + padding] = arr[i][j]; 
	    	}
	    }
	    
	    return padded;
		
	}
	
	
	public double[][] multiply(double[][] a, double[][] b){
		double[][] multiplied = new double[a.length][a[0].length];
		
		for(int i = 0; i<a.length; i++) {
			for(int j = 0; j<a[0].length; j++) {
				multiplied[i][j] = a[i][j]*b[i][j];
			}
		}
		
		
		return multiplied;
	}
	
	public double sum(double[][] a){
		double summed = 0;
		
		for(int i = 0; i<a.length; i++) {
			for(int j = 0; j<a[0].length; j++) {
				summed = summed + a[i][j];
			}
		}
		
		return summed;
	}
	
	public void heInit(double[][] W, Random rnd) {
        double std = Math.sqrt(2.0 / (W.length * W[0].length));
        for (int i = 0; i < W.length; i++) {
            for (int j = 0; j < W[0].length; j++) {
                W[i][j] = rnd.nextGaussian() * std;
            }
        }
    }

    public void glorotInit(double[] W, Random rnd) {
        double limit = Math.sqrt(6.0 / (W.length + 1));
        for (int i = 0; i < W.length; i++) {
            W[i] = (rnd.nextDouble() * 2 - 1) * limit;
        }
    }

    public double[] flatten(double[][] A) {
        int H = A.length, W = A[0].length;
        double[] v = new double[H * W];
        int k = 0;
        for (int i = 0; i < H; i++) { 
        	for (int j = 0; j < W; j++) {
        		v[k++] = A[i][j];
        	}
        }
        return v;
    }

    public double[][] reshape(double[] v, int H, int W) {
        if (v.length != H * W) throw new IllegalArgumentException("Bad shape");
        double[][] A = new double[H][W];
        int k = 0;
        for (int i = 0; i < H; i++) {
        	for (int j = 0; j < W; j++) {
        		A[i][j] = v[k++];
        	}
        }
        return A;
    }

    public double dot(double[] a, double[] b) {
        double s = 0.0;
        for (int i = 0; i < a.length; i++) s += a[i] * b[i];
        return s;
    }

    public double safeLog(double x) {
        double eps = 1e-12;
        return Math.log(Math.max(eps, Math.min(1 - eps, x)));
    }

    public double[] meanStd(List<Example> data) {
        double sum = 0.0, sumSq = 0.0; long n = 0;
        for (Example e : data) {
            for (int i = 0; i < e.image.length; i++) {
                for (int j = 0; j < e.image[0].length; j++) {
                    double v = e.image[i][j];
                    sum += v; 
                    sumSq += v * v; 
                    n++;
                }
            }
        }
        double mean = sum / n;
        double var = Math.max(1e-8, (sumSq / n) - mean * mean);
        return new double[]{ mean, Math.sqrt(var) };
    }

    public void standardizeInPlace(List<Example> data, double mean, double std) {
        for (Example e : data) {
            for (int i = 0; i < e.image.length; i++) {
                for (int j = 0; j < e.image[0].length; j++) {
                    e.image[i][j] = (e.image[i][j] - mean) / std;
                }
            }
        }
    }
	

}
