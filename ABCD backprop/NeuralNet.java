import java.io.*;
import java.util.*;
import java.lang.*;
import java.text.DecimalFormat;

/*
* Matthew L.
* 4/29/22
*
* Description: The model stores hidden layers, weights, 
* and auxiliary arrays and provides functionality for constructing an ABCD network,
* fitting it on input and output cases, and running it on test sets.
* Training will be done with the optimized back propagation algorithm for an ABCD network.
* outputs.
*/
class NeuralNet
{
   int numLayers;
   int layerM;
   int layerK;
   int layerJ;
   int layerI;
   double[][] layers;
   double[][] theta;
   double[][][] weights;
   double[] psiI;
   double[] psiJ;
   double[] bigOmega;
   double[][] inputSet;
   double[][] outputSet;
   double[][] predictionSet;
   double randLowBound;
   double randHighBound;
   int maxIterations;
   double lambdaVal;
   double errorCutoff;
   int numTestCases;
   String[] files;
   String runtimeMode;
   String weightsLoad;

/*
* Set configuration parameters in the constructor
* Get the neural network configuration from the ABCD config file
* retrieve the sizes of all layers and store them to allocate memory later
* will also read in configurations for the truth table and training configurations
*
* Parameters:
* String configFile - the file containing network configurations
* String truthTable - the file containing the truth table configurations
* String trainConfig - the file containing training configurations
*/
   public NeuralNet(String configFile, String truthTable, String trainConfig) throws IOException
   {
      BufferedReader br = new BufferedReader(new FileReader(configFile));
      StringTokenizer st = new StringTokenizer(br.readLine());

      st.nextToken();

      //read in the number of layers
      numLayers = Integer.parseInt(st.nextToken());

      st = new StringTokenizer(br.readLine());

      //read in the layer sizes
      layerM = Integer.parseInt(st.nextToken());
      layerK = Integer.parseInt(st.nextToken());
      layerJ = Integer.parseInt(st.nextToken());
      layerI = Integer.parseInt(st.nextToken());

      st = new StringTokenizer(br.readLine());
      st.nextToken();

      //read in the runtime mode (should be either "train" or "run")
      runtimeMode = st.nextToken(); 

      br.close();

      //read in truth table configurations
      br = new BufferedReader(new FileReader(truthTable));
      st = new StringTokenizer(br.readLine());

      //Get Function
      st.nextToken();
      st = new StringTokenizer(br.readLine());

      //read in number of test cases
      numTestCases = Integer.parseInt(st.nextToken());

      br.close();

      //read in training configuration
      br = new BufferedReader(new FileReader(trainConfig));
      st = new StringTokenizer(br.readLine());

      st.nextToken();

      //read in random range
      randLowBound = Double.parseDouble(st.nextToken());
      randHighBound = Double.parseDouble(st.nextToken());

      st = new StringTokenizer(br.readLine());
      st.nextToken();

      //read in maximum number of training iterations
      maxIterations = Integer.parseInt(st.nextToken());

      st = new StringTokenizer(br.readLine());
      st.nextToken();

      //read in lambda
      lambdaVal = Double.parseDouble(st.nextToken());

      st = new StringTokenizer(br.readLine());
      st.nextToken();

      //read in error threshold
      errorCutoff = Double.parseDouble(st.nextToken());

      st = new StringTokenizer(br.readLine());
      st.nextToken();

      //read in whether the weights should be loaded from a file or randomly generated
      weightsLoad = st.nextToken();

      br.close();
   } //public NeuralNet(String configFile, String truthTable, String trainConfig) throws IOException

/*
* Memory allocation (should occur before training and running)
* always creates the layers, weights, and truth table arrays
* if the runtime mode is "train", the thetas, psi layers and big Omega arrays will be allocated 
*/
   public void allocateArrays()
   {
      int numActivationLayers = numLayers-1;

      layers = new double[numLayers][];
      weights = new double[numActivationLayers][][];

      layers[0] = new double[layerM];
      layers[1] = new double[layerK];
      layers[2] = new double[layerJ];
      layers[3] = new double[layerI];

      weights[0] = new double[layerM][layerK];
      weights[1] = new double[layerK][layerJ];
      weights[2] = new double[layerJ][layerI];

      //if runtime mode is training, allocate memory for thetas, errors, psiI, deltaWeights, and bigOmegas
      if (runtimeMode.equals("train"))
      {
         theta = new double[numLayers][];
         theta[0] = new double[layerM];
         theta[1] = new double[layerK];
         theta[2] = new double[layerJ];
         theta[3] = new double[layerI];
         psiI = new double[layerI];
         psiJ = new double[layerJ];
         bigOmega = new double[layerJ];
      } //if (runtimeMode.equals("train"))
      
      //allocate memory for the input/output cells
      inputSet = new double[numTestCases][layerM];
      files = new String[numTestCases];
      outputSet = new double[numTestCases][layerI];
      predictionSet = new double[numTestCases][layerI];
   } //public void allocateArrays()

/*
* Parses the truth table given the truth table file and stores the truth table
* in the input set and the output set
*
* Parameters: 
* String fileName - the file to read the truth table from
*/
   public void parseTruthTable(String fileName) throws IOException
   {
      BufferedReader br = new BufferedReader(new FileReader(fileName));
      StringTokenizer st = new StringTokenizer(br.readLine());

      String sol = st.nextToken();

      st = new StringTokenizer(br.readLine());

      //read in the number of test cases
      int numTestCases = Integer.parseInt(st.nextToken());

      st.nextToken();


/*
temporary
*/
      int len = (int) Math.sqrt(layerM);

      if (sol.equals("image_recognition"))
      {
         for (int x = 0; x < numTestCases; x++)
         {
            st = new StringTokenizer(br.readLine());

            //read in activation file
            String activationFile = st.nextToken();

            files[x] = activationFile;
            //System.out.println(files[x]);

            BufferedReader bfr = new BufferedReader(new FileReader(activationFile));
            StringTokenizer fst;

            for (int y = 0; y < len; y++)
            {
               fst = new StringTokenizer(bfr.readLine());

               for (int z = 0; z < len; z++)
               {
                  inputSet[x][y * len + z] = Double.parseDouble(fst.nextToken());
               }
            }

            st.nextToken();

            //read in the output tokens
            for (int y = 0; y < layerI; y++)
            {
               outputSet[x][y] = Double.parseDouble(st.nextToken());
            }

            bfr.close();
         } //for (int x = 0; x < numTestCases; x++)
      } //if(sol.equals("image_recognition"))
      else
      {
         //Parse values into input and output sets
         for (int x = 0; x < numTestCases; x++)
         {   
            st = new StringTokenizer(br.readLine());

            //read in the input tokens
            for (int y = 0; y < layerM; y++)
            {  
               inputSet[x][y] = Double.parseDouble(st.nextToken());
            }

            st.nextToken();

            //read in the output tokens
            for (int y = 0; y < layerI; y++)
            {
               outputSet[x][y] = Double.parseDouble(st.nextToken());
            }
         } //for (int x = 0; x < numTestCases; x++)
      }

      br.close();
   } //public void parseTruthTable(String fileName) throws IOException

/*
* Echoes the configuration parameters (network config, and training config if the runtime mode is "train")
*/
   public void echoConfigParams()
   {
      //echo configuration parameters
      System.out.println("Network Config: ");
      System.out.println("   Total Number of Layers: " + numLayers);
      System.out.print("   Nodes per Layer: ");

      System.out.print(layerM + " ");
      System.out.print(layerK + " ");
      System.out.print(layerJ + " ");
      System.out.print(layerI + " ");
      System.out.println();

      System.out.println("   Configured to: " + runtimeMode);

      //print training parameters
      if (runtimeMode.equals("train"))
      {
         //printing training parameters
         System.out.println();
         System.out.println("Runtime Training Parameters: ");
         System.out.println("   Random Number Range: (" + randLowBound + ',' + randHighBound + ')');
         System.out.println("   Max Iterations: " + maxIterations);
         System.out.println("   Error Threshold: " + errorCutoff);
         System.out.println("   Lambda: " + lambdaVal);
         System.out.println();
      } //if (runtimeMode.equals("train"))
   } //public void echoConfigParams()

/*
* returns the runtime mode ("train" or "run")
*/
   public String getRuntimeMode()
   {
      return runtimeMode;
   }

/*
* Runs the model for training with the given input case and corresponding output values
* (Ex. for XOR: 0 0 -> 1). It will first loop over the input to assign to the first layer.
* It will then loop over layer k to calculate theta values for each theta k by summing the
* layer values in layer m multiplied by its corresponding weights. Afterwards, an activation
* function is applied to each theta. It will loop over layer j and apply the same algorithm 
* except while summing over layer k. Lastly, it will loop over layer i, calculating the final
* layer with the same method of summing over layer j, while also calculating values of psi for
* layer i. The function will return the sum of the squared error for one input test case.
* 
* Parameters:
* double[] input - an array of inputs corresponding to the first layer
* double[] output - an array of outputs
*/
   public double runForTraining(double[] input, double[] output)
   {
      //set first layer to the input
      for (int m = 0; m < layerM; m++)
      {
         layers[0][m] = input[m];
      }

      // k loop as described in function description
      for (int k = 0; k < layerK; k++)
      {
         theta[1][k] = 0.0;

         // m loop as described in the function description
         for (int m = 0; m < layerM; m++)
         {
            theta[1][k] += layers[0][m] * weights[0][m][k];
         }

         layers[1][k] = ModelUtil.activationFunction(theta[1][k]);
      }

      // j loop as described in the function description
      for (int j = 0; j < layerJ; j++)
      {
         theta[2][j] = 0.0;

         // k loop as described in the function description
         for (int k = 0; k < layerK; k++)
         {
            theta[2][j] += layers[1][k] * weights[1][k][j];
         }

         layers[2][j] = ModelUtil.activationFunction(theta[2][j]);
      } //for (int j = 0; j < layerJ; j++)

      double smallOmega = 0.0;
      double errorVal = 0.0;

      // i loop as described in function description
      for (int i = 0; i < layerI; i++)
      {
         theta[3][i] = 0.0;

         // j loop as described in the function description
         for (int j = 0; j < layerJ; j++)
         {
            theta[3][i] += layers[2][j] * weights[2][j][i];
         }

         layers[3][i] = ModelUtil.activationFunction(theta[3][i]);

         //calculate error (expected value - computed value)
         errorVal = output[i] - layers[3][i];

         //calculate psi for the output layers
         psiI[i] = errorVal * ModelUtil.activationFunctionDerivative(theta[3][i]);

         //add squared error
         smallOmega += errorVal * errorVal;
      } //for (int i = 0; i < layerI; i++)

      return smallOmega;
   } //public double run(double[] input, double[] output)

/*
* Returns a random double between the given random range
*
* Parameters:
* double low: the lower range of the random value
* double high: the higher range of the random value
*/
   public double randomize(double low, double high)
   {
      return Math.random() * (high - low) + low;
   }

/*
* Populates the weights with random values between the given range if the model has been configured
* to generated random weights; otherwise, if it has been set to manual, it will populate weights from a file
*
* Parameters:
* double low - lower bound for random values
* double high - higher bound for random values
* String weightsFile - the file to read weights in from
*/
   public void populateWeightsForTraining(double low, double high, String weightsFile) throws IOException
   {
      if (weightsLoad.equals("manual"))
      {
         BufferedReader br = new BufferedReader(new FileReader(weightsFile));
         StringTokenizer st;

         //read in weights from a file
         for (int x = 0; x < weights.length; x++)
         {
            st = new StringTokenizer(br.readLine());

            for (int y = 0; y < weights[x].length; y++)
            {
               for (int z = 0; z < weights[x][y].length; z++)
               {
                  weights[x][y][z] = Double.parseDouble(st.nextToken());
               }
            }
         } //for (int x = 0; x < weights.length; x++)

         br.close();
      } //if (weightsLoad.equals("manual"))
      else if (weightsLoad.equals("randomized"))
      {
         //randomly generate weights with the randomize() function
         for (int x = 0; x < weights.length; x++)
         {
            //set the weights to random floats
            for (int y = 0; y < weights[x].length; y++)
            {
               for (int z = 0; z < weights[x][y].length; z++)
               {
                  weights[x][y][z] = randomize(low, high);
               }
            }
         } //for (int x = 0; x < weights.length; x++)
      } //else if (weightsLoad.equals("randomized"))
   } //public void populateWeightsForTraining(double low, double high, String weightsFile) throws IOException

/*
* Save the weights to a given file
*
* Parameters:
* String weightsFile - the file to save the weights to
*/
   public void saveWeights(String weightsFile) throws IOException
   {
      //save the weights with one row of weights for each layer
      PrintWriter pw = new PrintWriter(weightsFile);

      for (double[][] x : weights)
      {
         for (double[] y : x)
         {
            for (double z : y)
            {
               pw.print(z + " ");
            }
         }

         pw.println();
      } //for (double[][] x : weights)

      pw.close();
   } //public void saveWeights(String weightsFile) throws IOException

/*
* Prints the training finish information, which includes runtime training parameters
* and training exit information
*
* int epochs - the number of iterations reached
* int iterations - the max number of iterations
* double squaredError - the squared error reached
* double time - the time it took to train the model
*/
   public void printTrainingFinishInfo(int epochs, int iterations, double squaredError, double time) throws IOException
   {  
      //determine the reason for termination
      String reasonForEnd;

      System.out.println("Training Exit Information: ");

      if (iterations >= epochs)
      {
         reasonForEnd = "Max iterations reached";
      }
      else
      {
         reasonForEnd = "Error threshold reached";
      }

      System.out.println("   Reason for end of run: " + reasonForEnd);
      System.out.println("   Iterations Reached: " + iterations);
      System.out.println("   Error reached: " + squaredError);
      System.out.println(".  Finish time: " + time + "ms");
   } //public void printTrainingFinishInfo(int epochs, int iterations, double squaredError, double time) throws IOException

/*
* Fits the ABCD backprop network using the given input and output set, random range, number of iterations,
* lambda value, and error threshold. The input will be a small batch, but will be run many times
* throughout the model to lower its error. The weights will be originally set to a value in
* between the given range or manually read, and the model will be trained reaching either the max number of 
* iterations, or if it is lower than the error threshold first. The lambda value will can be 
* used in weight updates to increase the step/speed of training and to avoid model training
* being stopped at local minima/maxima. Layers[0] refers to layer A and layers[1] refers to 
* layer B, and weights/theta arrays follow the same. The weight layer is also an array of flattened
* weights. Random range as described in the parameters contains the lower and upper bounds of the
* random value to generate. ABCD backprop algorithm - after running the runForTraining() function which calculates
* the psiI values and error, the train function will iterate over the j layer to update weights for layer i
* while calculating large omega J and using large omega J to calculate psi values for layer j. 
* Next, the k layer will be iterated over and the weights for layer j will be updated 
* while calculateing large omega K, and using large omega K to update weights for layer M,
* for a total of 5 loops for each test case.
* 
* Parameters:
* double[][] input - the inputs of the truth table
* double[] output - an array of corresponding outputs in the truth table
* double randomLowBound - lower bound for random range
* double randomHighBound - higher bound for random range
* int epochs - the number of iterations
* double lambda - the lambda value used in updating weights
* double errorThreshold - the error cutoff for the model to stop at
*/
   public void train(String weightsFile, double[][] input, double[][] output, double randomLowBound, 
      double randomHighBound, int epochs, double lambda, double errorThreshold) throws IOException
   {
      //start recording time for training
      double start = System.nanoTime();

      //*Training step - the model fit described in the function description above
      double squaredError = 1.0;
      int iterations = 0;

      //an iteration will be run while the model hasn't reached the maximum iterations or is under the error cutoff
      while (iterations < epochs && squaredError > errorThreshold) 
      {
         //initialize/reset errors
         squaredError = 0.0;
         double error = 0.0;
         double largeOmegaJ = 0.0;
         double largeOmegaK = 0.0;

         //iterate over all test cases
         for (int y = 0; y < input.length; y++)
         {
            //run a single test case and calculate squared error
            error = runForTraining(input[y], output[y]);
            squaredError += error;
 
            //iterate over layer j (j loop)
            for (int j = 0; j < layerJ; j++)
            {
               largeOmegaJ = 0.0;

               //iterate over layer i (i loop)
               for (int i = 0; i < layerI; i++)
               {
                  largeOmegaJ += psiI[i] * weights[2][j][i];
                  weights[2][j][i] += lambda * layers[2][j] * psiI[i];
               }

               psiJ[j] = largeOmegaJ * ModelUtil.activationFunctionDerivative(theta[2][j]);
            } //for (int j = 0; j < layerJ; j++)

            //iterate over layer k (k loop)
            for (int k = 0; k < layerK; k++)
            {
               largeOmegaK = 0.0;

               //iterate over layer j (j loop)
               for (int j = 0; j < layerJ; j++)
               {
                  largeOmegaK += psiJ[j] * weights[1][k][j];
                  weights[1][k][j] += lambda * layers[1][k] * psiJ[j]; 
               }

               //iterate over layer m (m loop)
               for (int m = 0; m < layerM; m++)
               {
                  weights[0][m][k] += lambda * layers[0][m] * largeOmegaK * ModelUtil.activationFunctionDerivative(theta[1][k]);
               }
            } //for (int k = 0; k < layerK; k++)
         } //for (int y = 0; y < input.length; y++)

         //halve the error
         squaredError *= 0.5;
         iterations++;
      } //while (iterations < epochs && squaredError > errorThreshold) 

      //finish recording training time
      double finish = System.nanoTime();

      //save weights and print training finish information
      saveWeights(weightsFile);

      //divide by 1000000 to display time in milliseconds
      final double MILLISECOND_OFFSET = 1000000.0;

      printTrainingFinishInfo(epochs, iterations, squaredError, (finish - start) / MILLISECOND_OFFSET);
   } //public void train(String weightsFile, ...

/*
* Populates the weights for running by loading them from a file
*
* String weightFile - the file to read weights in from
*/
   public void populateWeightsForRunning(String weightFile) throws IOException
   {
      BufferedReader br = new BufferedReader(new FileReader(weightFile));
      StringTokenizer st;

      for (int x = 0; x < weights.length; x++)
      {
         st = new StringTokenizer(br.readLine());

         //set the weights to random floats
         for (int y = 0; y < weights[x].length; y++)
         {
            for (int z = 0; z < weights[x][y].length; z++)
            {
               weights[x][y][z] = Double.parseDouble(st.nextToken());
            }
         }
      } //for (int x = 0; x < weights.length; x++)

      br.close();
   } //public void populateWeightsForRunning(String weightFile) throws IOException

/*
* Reports the results after running the model by printing the truth table
* which is the output set for each input case
*
* double[][] input - the input values
* double[][] predictions - the predicted values generated by the model
*/
   public void printRunningFinishInfo(double[][] input, double[][] predictions)
   {
      for (int p = 0; p < numTestCases; p++)
      {
         /*
         //print input set
         for (int m = 0; m < layerM; m++)
         {
            System.out.print(input[p][m] + " ");
         }*/
         System.out.print(files[p] + " ");

         System.out.print("-> ");

         DecimalFormat df = new DecimalFormat();
         df.setMaximumFractionDigits(2);

         //print output set
         for (int i = 0; i < layerI; i++)
         {
            System.out.printf(df.format(predictions[p][i]) + " ");
         }

         System.out.println();
      } //for (int p = 0; p < numTestCases; p++)
   } //public void printRunningFinishInfo(double[][] input)

/*
* Predicts the values for the given input set and expected output.
* It will only run the forward propagation as described in runForTraining(),
* however, it will be able to calculate predictions separate from training 
* and will not adjust the weights, calculate thetas, or calculate psi values.
*
* double[][] input - the input values (ex. 1 0)
*/
   public void runModel(double[][] input) throws IOException
   {
      //iterate over all test cases
      for (int p = 0; p < input.length; p++)
      {
         //set first layer to the input
         for (int m = 0; m < layerM; m++)
         {
            layers[0][m] = input[p][m];
         }

         // k loop
         for (int k = 0; k < layerK; k++)
         {
            layers[1][k] = 0.0;

            // m loop
            for (int m = 0; m < layerM; m++)
            {
               layers[1][k] += layers[0][m] * weights[0][m][k];
            }

            layers[1][k] = ModelUtil.activationFunction(layers[1][k]);
         } //for (int k = 0; k < layerK; k++)

         // j loop
         for (int j = 0; j < layerJ; j++)
         {
            layers[2][j] = 0.0;

            // k loop
            for (int k = 0; k < layerK; k++)
            {
               layers[2][j] += layers[1][k] * weights[1][k][j];
            }

            layers[2][j] = ModelUtil.activationFunction(layers[2][j]);
         } //for (int j = 0; j < layerJ; j++)

         // i loop as described in function description
         for (int i = 0; i < layerI; i++)
         {
            layers[3][i] = 0.0;

            // j loop
            for (int j = 0; j < layerJ; j++)
            {
               layers[3][i] += layers[2][j] * weights[2][j][i];
            }

            layers[3][i] = ModelUtil.activationFunction(layers[3][i]);

            predictionSet[p][i] = layers[3][i];
         } //for (int i = 0; i < layerI; i++)
      } //for (int p = 0; p < input.length; p++)

      printRunningFinishInfo(input, predictionSet);
   } //public void runModel(double[][] input) throws IOException
} //class NeuralNet