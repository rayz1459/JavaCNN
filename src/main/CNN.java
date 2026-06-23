package main;

import javann.JavaNN;
import utils.ArrayUtilities;

import javax.imageio.ImageIO;

import data.Example;
import data.Grads;
import data.Params;
import data.ParamsFull;

import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class CNN {

    public static String TRAIN_CATS_DIR = "Datasets/Training/gray";
    public static String TRAIN_DOGS_DIR = "Datasets/Training/dogs_copy";
    public static String TEST_CATS_DIR  = "Datasets/Training/random";
    public static String TEST_DOGS_DIR  = "Datasets/Training/Dog";

    public static int IMG_H = 150;
    public static int IMG_W = 150;

    public static int F = 3;      
    public static int STRIDE = 1;
    public static int PAD = 1;

    public static int EPOCHS = 2;
    public static double LR = 0.003;


    // Activations / regularization
    public static double ALPHA = 0.1;
    public static double WD = 1e-4;

    public static void main(String[] args) throws Exception {
        JavaNN nn = new JavaNN();
        ArrayUtilities ap = new ArrayUtilities();
        Random rnd = new Random(0);

        ArrayList<Example> train = new ArrayList<>();
        ArrayList<Example> test  = new ArrayList<>();
        loadFolder(TRAIN_CATS_DIR, 0, train);
        loadFolder(TRAIN_DOGS_DIR, 1, train);
        loadFolder(TEST_CATS_DIR,  0, test);
        loadFolder(TEST_DOGS_DIR,  1, test);
        

        double[] ms = ap.meanStd(train);
        ap.standardizeInPlace(train, ms[0], ms[1]);
        ap.standardizeInPlace(test,  ms[0], ms[1]);

        //(n-f+2p)/s +1
        double[][] W1 = new double[F][F];
        double[][] W2 = new double[F][F];
        double[][] W3 = new double[F][F];
        
        double[][] b1 = new double[][]{{0.01}};
        double[][] b2 = new double[][]{{0.01}};
        double[][] b3 = new double[][]{{0.01}};
        
        ap.heInit(W1, rnd); 
        ap.heInit(W2, rnd); 
        ap.heInit(W3, rnd);

        int D = IMG_H * IMG_W;
        double[] Wfc = new double[D];
        double bfc = 0.0;
        ap.glorotInit(Wfc, rnd);


        //Start training
        System.out.println("Loaded Size: " + train.size());
        for (int epoch = 1; epoch <= EPOCHS; epoch++) {
            Collections.shuffle(train, rnd);
            double epochLoss = 0.0;
            int correct = 0;

            for (Example ex : train) {
                double[][] X = ex.image;
                int y = ex.label;
                
                //forwardprop
                double[][] Z1 = nn.conv_forward(X,  W1, b1, STRIDE, PAD);
                double[][] A1 = nn.leakyRelu(Z1, ALPHA);

                double[][] Z2 = nn.conv_forward(A1, W2, b2, STRIDE, PAD);
                double[][] A2 = nn.leakyRelu(Z2, ALPHA);

                double[][] Z3 = nn.conv_forward(A2, W3, b3, STRIDE, PAD);
                double[][] A3 = nn.leakyRelu(Z3, ALPHA);

                double[] a3vec = ap.flatten(A3);
                double z = ap.dot(Wfc, a3vec) + bfc;
                double a = nn.sigmoid(z);

                double loss = -(y * ap.safeLog(a) + (1 - y) * ap.safeLog(1 - a));
                epochLoss += loss;
                
                //accuracy prediction
                int pred;
                if (a >= 0.5) {
                    pred = 1;
                }
                else {
                    pred = 0;
                }
                if (pred == y) { 
                	correct++;
                }
                
                
                //backprop
                double dz = a - y;

                //FC grads
                double[] dWfc = new double[D];
                for (int i = 0; i < D; i++) {
                	dWfc[i] = dz * a3vec[i];
                }
                double dbfc = dz;

                double[] da3vec = new double[D];
                for (int i = 0; i < D; i++) {
                	da3vec[i] = Wfc[i] * dz;
                }
                double[][] dA3 = ap.reshape(da3vec, IMG_H, IMG_W);

                //backprop conv3
                double[][] dZ3 = nn.leakyRelu_backward(dA3, Z3, ALPHA);
                Grads g3 = nn.conv_backward(dZ3, A2, W3, b3, STRIDE, PAD);

                //backprop conv2
                double[][] dZ2 = nn.leakyRelu_backward(g3.dA_prev, Z2, ALPHA);
                Grads g2 = nn.conv_backward(dZ2, A1, W2, b2, STRIDE, PAD);

                //backprop conv1
                double[][] dZ1 = nn.leakyRelu_backward(g2.dA_prev, Z1, ALPHA);
                Grads g1 = nn.conv_backward(dZ1, X,  W1, b1, STRIDE, PAD);

                //weight updates + L2 Regularization
                for (int i = 0; i < F; i++) {
                	for (int j = 0; j < F; j++) {
                		W1[i][j] = W1[i][j] * (1 - LR * WD) - LR * g1.dW[i][j];
                		W2[i][j] = W2[i][j] * (1 - LR * WD) - LR * g2.dW[i][j];
                		W3[i][j] = W3[i][j] * (1 - LR * WD) - LR * g3.dW[i][j];
                	}
                }
                b1[0][0] = b1[0][0] - LR * g1.db[0][0];
                b2[0][0] = b2[0][0] - LR * g2.db[0][0];
                b3[0][0] = b3[0][0] - LR * g3.db[0][0];

                for (int i = 0; i < D; i++) {
                	Wfc[i] = Wfc[i] * (1 - LR * WD) - LR * dWfc[i];
                }
                bfc -= LR * dbfc;
            }
            
            double acc = correct / (double)train.size();
            System.out.println("Epoch "+ epoch + "/" + EPOCHS + ", Average Loss: " +epochLoss / train.size()+ ", Accuracy: " + acc);
            
        }
        
        /**
        //Serialization of model
        Params param1 = new Params(W1, b1);
        Params param2 = new Params(W2, b2);
        Params param3 = new Params(W3, b3);
        
        ArrayList<Params> model = new ArrayList<>(); //might change to array
        
        model.add(param1);
        model.add(param2);
        model.add(param3);
        
        ParamsFull paramsfull = new ParamsFull(model);
        nn.compileModel(paramsfull, "model");
        
        
        //Deserialization
        ParamsFull deparamsfull = nn.importModel("model");
        W1 = deparamsfull.params.get(0).W;
        b1 = deparamsfull.params.get(0).b;
        W2 = deparamsfull.params.get(1).W;
        b2 = deparamsfull.params.get(1).b;
        W3 = deparamsfull.params.get(2).W;
        b3 = deparamsfull.params.get(2).b;
        **/
        

        
        //Test
        System.out.println();
        System.out.println("Test samples: "+test.size());
        int correct = 0;
        for (Example ex : test) {
            double[][] Z1 = nn.conv_forward(ex.image, W1, b1, STRIDE, PAD);
            double[][] A1 = nn.leakyRelu(Z1, ALPHA);

            double[][] Z2 = nn.conv_forward(A1, W2, b2, STRIDE, PAD);
            double[][] A2 = nn.leakyRelu(Z2, ALPHA);

            double[][] Z3 = nn.conv_forward(A2, W3, b3, STRIDE, PAD);
            double[][] A3 = nn.leakyRelu(Z3, ALPHA);

            double[] a3vec = ap.flatten(A3);
            double a = nn.sigmoid(ap.dot(Wfc, a3vec) + bfc);
            
            int pred;
            
            if(a>=0.5) {
            	pred = 1;
            }
            else {
            	pred = 0;
            }
            
            if (pred == ex.label) correct++;
            System.out.println(a+ ", "+ex.label);
        }
        double acc = correct / (double) test.size();
        System.out.println("Test accuracy: " + acc);
    }


    //folder utils
    static void loadFolder(String folder, int label, List<Example> out) {
        File dir = new File(folder);
        if (!dir.exists() || !dir.isDirectory()) return;
        File[] files = dir.listFiles((d, name) -> {
            String n = name.toLowerCase();
            return n.endsWith(".jpg") || n.endsWith(".jpeg") || n.endsWith(".png") || n.endsWith(".bmp");
        });
        if (files == null) return;
        for (File f : files) {
            try {
                double[][] img = readAsGrayscaleArray(f, IMG_W, IMG_H);
                out.add(new Example(img, label));
            } catch (Exception e) {
                System.err.println("Skip " + f.getName() + ": " + e.getMessage());
            }
        }
    }

    static double[][] readAsGrayscaleArray(File file, int targetW, int targetH) throws Exception {
        BufferedImage src = ImageIO.read(file);
        if (src == null) throw new IllegalArgumentException("Unsupported image: " + file);
        BufferedImage resized = new BufferedImage(targetW, targetH, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g2d = resized.createGraphics();
        Image tmp = src.getScaledInstance(targetW, targetH, Image.SCALE_SMOOTH);
        g2d.drawImage(tmp, 0, 0, null);
        g2d.dispose();

        double[][] arr = new double[targetH][targetW];
        for (int y = 0; y < targetH; y++) {
            for (int x = 0; x < targetW; x++) {
                int rgb = resized.getRGB(x, y);
                int gray = rgb & 0xFF;
                arr[y][x] = gray/255.0;  //standardization
            }
        }
        return arr;
    }


    
}
