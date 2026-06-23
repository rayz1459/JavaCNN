**JavaCNN**

The following project involves the implementation of a Convolutional Neural Network (CNN) built entirely from scratch in Java, without the use of external machine learning libraries. The goal of the project was to develop a complete, end-to-end image classification pipeline, to understand both a conceptual understanding of a CNN, but also the actual ability to translate those concepts into a fully functioning system at a basic code level. The project was also developed and adapted into a formal course assignment under professor supervision, though that direction was ultimately not implemented. However, the underlying system design and implementation were completed independently. Instructions I made for that course assignment can be found here. 

At a high level, the model performs binary image classification using a multi-layer convolution architecture. The implementation included 3 convolutional layers followed by a fully connected layer. Each convolutional layer applied learnt filters with configurable strides and padding which was followed by a leaky ReLU activation function. All of this enabled the network to capture increasingly abstract spatial features. The final layer reduced learnt representations into a scalar output using a sigmoid activation function for a final probabilistic classification. 
The system was structured in an object oriented manner across several packages:

main (CNN.java): Serves as the main class and the entire training and evaluation pipeline. It also handles the dataset loading, preprocessing (Standardization), parameter initialization, training loops, and testing.

javann (JavaNN): Implements core neural network operations and includes forward and backward passes for the convolutional layers, activation functions, and gradient propagation.

data (Example, Grads, Params, ParamsFull): Defines data structures used throughout the project, including labeled training data, gradient containers for backpropagation, and parameter storage for model serialization. 

utils (ArrayUtilities): Provides supporting math operations such as vectorization, reshaping initialization (He and Glorot), normalization, and array operations.

The training process was implemented manually, including both forward and backward propagation, without automatic machine learning libraries or frameworks. As shown below, the main training loop computes binary cross-entropy loss and updates parameters via gradient descent with L2 regularization. Gradients are explicitly calculated for each layer, including filters and the FC layer.
Overall, I hope this project emphasizes algorithmic control and the prioritization of ground-up implementation over abstraction. By reconstructing the internal mechanics of a CNN in Java, the project hopefully demonstrates a deep engagement with both mathematical foundations of data processing and principles underlying modern machine learning. Below are all the packages and code sources used.
