package javann;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import data.Grads;
import data.GradsFull;
import data.ParamsFull;
import utils.ArrayUtilities;

public class JavaNN implements JavaNNInterface{
	
	ArrayUtilities jn = new ArrayUtilities();
	
	
	public double[][] zero_pad(double[][] arr, int pad) {
	    if (pad == 0) {
	    	return arr;
	    }

	    int H = arr.length;
	    int W = arr[0].length;

	    double[][] padded = new double[H + 2 * pad][W + 2 * pad];

	    for (int i = 0; i < H; i++) {
	        for (int j = 0; j < W; j++) {
	            padded[i + pad][j + pad] = arr[i][j];
	        }
	    }
	    return padded;
	}


	
	public double[][] conv_forward(double[][] A_prev, double[][] W, double[][] b, int stride, int pad) {
		int n_H_prev = A_prev.length;
		int n_W_prev = A_prev[0].length;
		int f = W.length;
		
		//dimensions of output volume of the convolution
		int n_H = (int) (Math.floor((n_H_prev-f+2*pad)/stride)+1);
		int n_W = (int) (Math.floor((n_W_prev-f+2*pad)/stride)+1);
		
		double[][] Z = new double[n_H][n_W]; //check later if should set to 0 but might increase time complexity
		double[][]A_prev_pad = zero_pad(A_prev, pad);
		
		for (int h = 0; h<n_H; h++) {
			
			int vert_start = h*stride;
			int vert_end = vert_start + f;
			
			for(int w = 0; w<n_W; w++) {
				int horiz_start = w*stride;
				int horiz_end = horiz_start+f;
				
				//extract slice and compute convolution sum
	            double sum = 0.0;
	            for (int i = 0; i < f; i++) {
	                for (int j = 0; j < f; j++) {
	                    sum += A_prev_pad[vert_start + i][horiz_start + j] * W[i][j];
	                }
	            }

	            //add bias
	            sum += b[0][0]; //assuming b is a 1x1 matrix

	            //save result
	            Z[h][w] = sum;
			}
		}
		
		return Z;
		

		
		
	}
	
	public Grads conv_backward(double[][] dZ, double[][] A_prev, double[][] W, double[][] b, int stride, int pad) {
		int n_H_prev = A_prev.length;
		int n_W_prev = A_prev[0].length;
		int f = W.length;
		
		int n_H = dZ.length;
		int n_W = dZ[0].length;
		
		double[][] dA_prev = new double[n_H_prev][n_W_prev];
		double[][] dW = new double[W.length][W[0].length];
		double[][] db = new double[b.length][b[0].length];
		double[][] A_prev_pad = zero_pad(A_prev, pad);
		double[][] dA_prev_pad = zero_pad(dA_prev, pad);
		
		for(int h = 0; h<n_H; h++) {
			for(int w = 0; w<n_W; w++) {
				int vert_start = h*stride;
				int vert_end = vert_start + f;
				int horiz_start = w*stride;
				int horiz_end = horiz_start+f;
				
				//slice A_prev_pad region and accumulate grads
				for (int i = vert_start; i < vert_end; i++) {
				    for (int j = horiz_start; j < horiz_end; j++) {
				    	double a_ij = A_prev_pad[i][j]; //a_slice[i - vert_start][j - horiz_start]
				    	double w_ij = W[i - vert_start][j - horiz_start];

				        //dA_prev_pad region gets += W * dZ[h, w]
				        dA_prev_pad[i][j] += w_ij * dZ[h][w];

				        //dW gets += a_slice * dZ[h, w]
				        dW[i - vert_start][j - horiz_start] += a_ij * dZ[h][w];
				    }
				}

				//db is sum of dZ over spatial positions (use db[0][0] for 2D bias)
				db[0][0] += dZ[h][w];

				
			}
		}
		
		//remove padding to get dA_prev from dA_prev_pad
		for (int i = 0; i < n_H_prev; i++) {
		    for (int j = 0; j < n_W_prev; j++) {
		        dA_prev[i][j] = dA_prev_pad[i + pad][j + pad];
		    }
		}
		
		return new Grads(dA_prev, dW, db);

		
	}
	
	
	public double[][] relu(double[][] Z) {
        int H = Z.length, W = Z[0].length;
        double[][] A = new double[H][W];
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                A[i][j] = Math.max(0.0, Z[i][j]);
            }
        }
        return A;
    }

	public double[][] relu_backward(double[][] dA, double[][] Z) {
	    int H = dA.length, W = dA[0].length;
	    double[][] dZ = new double[H][W];

	    for (int i = 0; i < H; i++) {
	        for (int j = 0; j < W; j++) {
	            if (Z[i][j] > 0.0) {
	                dZ[i][j] = dA[i][j];
	            } else {
	                dZ[i][j] = 0.0;
	            }
	        }
	    }
	    return dZ;
	}

    public double sigmoid(double z) {
        if (z >= 0) {
            double ez = Math.exp(-z);
            return 1.0 / (1.0 + ez);
        } else {
            double ez = Math.exp(z);
            return ez / (1.0 + ez);
        }
    }

    public double[][] leakyRelu(double[][] Z, double alpha) {
        int H = Z.length, W = Z[0].length;
        double[][] A = new double[H][W];

        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                if (Z[i][j] > 0.0) {
                    A[i][j] = Z[i][j];
                } else {
                    A[i][j] = alpha * Z[i][j];
                }
            }
        }

        return A;
    }

    public double[][] leakyRelu_backward(double[][] dA, double[][] Z, double alpha) {
        int H = dA.length, W = dA[0].length;
        double[][] dZ = new double[H][W];

        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                if (Z[i][j] > 0.0) {
                    dZ[i][j] = dA[i][j];
                } else {
                    dZ[i][j] = alpha * dA[i][j];
                }
            }
        }

        return dZ;
    }
    
    
    
    public void compileModel(ParamsFull params, String string) {
    	try {
			FileOutputStream fos = new FileOutputStream(string+"jnn");
			
			ObjectOutputStream oos = new ObjectOutputStream(fos);
			
			oos.writeObject(params);
			
			oos.close();
			fos.close();
			System.out.println("Model Compiled");
		} 
		catch (IOException ioe) {
			ioe.printStackTrace();
		}
    }
    
    public ParamsFull importModel(String string) {
    	try{
		      FileInputStream fis = new FileInputStream(string+"jnn");
		      
		      ObjectInputStream ois = new ObjectInputStream(fis);
		      
		      ParamsFull params = (ParamsFull)ois.readObject();
		      ois.close();
		      fis.close();
		      return params;

		    }
		    catch(IOException ioe) {
		       ioe.printStackTrace();

		    }
		 catch(ClassNotFoundException cnfe) {
		       cnfe.printStackTrace();

		     }
    	return null;
    	
    }
    


}

