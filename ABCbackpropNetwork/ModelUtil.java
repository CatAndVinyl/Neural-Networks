import java.util.*;

/*
* Author: Matthew L.
* Date 3/11/22
*
* ModelUtil continues tools and functions for the model to use.
* Currently, this includes the activation function sigmoid andz
* its derivative for gradient descent.
*/
class ModelUtil
{
/*
* An empty constructor, since there is no need to create an
* instance of the utilities class, as all functions are static
*/
   public ModelUtil()
   {

   }

/*
* The sigmoid activation function, which will return a value 
* between 0 and 1
*
* double val - the value to apply sigmoid to
*/
   public static double sigmoid(double val)
   {
      return 1.0 / (1.0 + Math.exp(-val));
   }

/*
* The derivative of sigmoid, which will be used in gradient descent
* or other purposes
*
* double val - the value to apply sigmoid' to
*/
   public static double sigmoidDerivative(double val)
   {
      return sigmoid(val) * (1.0 - sigmoid(val));
   }
} //class ModelUtil