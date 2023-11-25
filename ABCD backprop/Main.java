import java.io.*;
import java.util.*;

/*
* Matthew L.
* 4/29/22
*
* Description: the Main class will be used as a tester of the NeuralNet class,
* by first parsing the configuration file as well as the truth table and generating
* the ABCD network based on the information in the five files. Depending on
* whether the model is configured to run or train, the tester will either fit or run the model.
*/
class Main 
{
   static String networkConfig;
   static String truthTable;
   static String trainConfig;
   static String savedWeights;
   static String manualWeights;

/*
* Parse the control file which contains network configuration, 
* the truth table files, training configuration, saved weights, and manual weights
*
* Parameters:
* String controlFile - the file containing the other configurations and file information
*/
   public static void parseControlFile(String controlFile) throws IOException
   {
      BufferedReader br = new BufferedReader(new FileReader(controlFile));
      StringTokenizer st = new StringTokenizer(br.readLine());

      networkConfig = st.nextToken();

      st = new StringTokenizer(br.readLine());
      truthTable = st.nextToken();

      st = new StringTokenizer(br.readLine());
      trainConfig = st.nextToken();

      st = new StringTokenizer(br.readLine());
      savedWeights = st.nextToken();

      st = new StringTokenizer(br.readLine());
      manualWeights = st.nextToken();
   }

/*
* The main tester is the main function used to test the neural network as described in
* the class documentation.
* Required Structure
* -Set configuration parameters (read file configurations)
* -Echo configuration parameters
* -Allocate memory for arrays (in network constructor)
* -Populate arrays (in network fit() function)
* -Training (in network train() function and after populating arrays)
* -Running (in network runModel() function)
* -Report results (after network runs) 
*
* Parameters: 
* String[] args - command line arguments, which will consist of a control file with five files: 
*    the network configuration, the truth table files, training configuration, saved weights, and manual weights 
*/
   public static void main(String[] args) throws IOException
   {
      //if a control file has been provided, then parse it; otherwise use a default control file
      if (args.length == 0)
         parseControlFile("control_file_default.txt");
      else
         parseControlFile(args[0]);

      //generate model and set configuration parameters inside the constructor
      NeuralNet ABCD = new NeuralNet(networkConfig, truthTable, trainConfig);

      //echo the configuration parameters, allocate arrays, and read in the truth table
      ABCD.echoConfigParams();
      ABCD.allocateArrays();
      ABCD.parseTruthTable(truthTable);

/*
* if the runtime mode is "train", training weight arrays will be populated, the train function will be called
* and the model will be run once with the saved weights; otherwise if the runtime mode is "run",
* just run the model with the saved weights
*/
      if (ABCD.getRuntimeMode().equals("train"))
      {
         ABCD.populateWeightsForTraining(ABCD.randLowBound, ABCD.randHighBound, manualWeights);
         ABCD.train(savedWeights, ABCD.inputSet, ABCD.outputSet, ABCD.randLowBound, 
            ABCD.randHighBound, ABCD.maxIterations, ABCD.lambdaVal, ABCD.errorCutoff);
         ABCD.populateWeightsForRunning(savedWeights);
         ABCD.runModel(ABCD.inputSet);
      }
      else if (ABCD.getRuntimeMode().equals("run"))
      {
         ABCD.populateWeightsForRunning(savedWeights);
         ABCD.runModel(ABCD.inputSet);
      }
   } //public static void main(String[] args) throws IOException
} //class Main