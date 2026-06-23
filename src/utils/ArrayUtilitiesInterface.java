package utils;
import java.util.*;

import data.Example;

public interface ArrayUtilitiesInterface {
	
	double[][] pad(double[][] arr, int padding);
	
	
	double[][] multiply(double[][] a, double[][] b);
	
	double sum(double[][] a);
	
	public void heInit(double[][] W, Random rnd);
	public void glorotInit(double[] W, Random rnd);
	public double[] flatten(double[][] A);
	public double[][] reshape(double[] v, int H, int W);
	public double dot(double[] a, double[] b);
	public double safeLog(double x);
	public double[] meanStd(List<Example> data);
	public void standardizeInPlace(List<Example> data, double mean, double std);
	

}
