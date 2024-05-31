using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML
{
    public static class UtilityFunctions
    {


        // Mean Squared Error Loss
        /*
        Purpose: Used as a loss function in regression problems. It measures the average of the squares of the errors—that is, 
            the average squared difference between the estimated values and the actual value.
        Usage: Commonly used in linear regression, neural networks for regression tasks, etc. It's helpful in optimization 
            problems where minimizing the error is the goal.
         */
        public static double mse(double ActualValue, double predicted)
        {
            return Math.Pow(predicted - ActualValue, 2);
        }
        public static float mse(float ActualValue, float predicted)
        {
            return (float)Math.Pow(predicted - ActualValue, 2);
        }

        // Cross-Entropy Loss for binary classification
        /*
         Purpose: A loss function used for binary classification problems. It measures the performance of a classification model whose
               output is a probability value between 0 and 1. Cross-entropy loss increases as the predicted probability diverges from the actual label.
         Usage: Widely used in models that output probabilities, such as logistic regression and binary classification neural networks.
         */
        public static double binaryCrossEntropy(double value, double predicted)
        {
            return -(value * Math.Log(predicted) + (1 - value) * Math.Log(1 - predicted));
        }
        public static float binaryCrossEntropy(float value, float predicted)
        {
            return -(float)(value * Math.Log(predicted) + (1 - value) * Math.Log(1 - predicted));
        }

        // Min-Max Normalization
        /*
        Purpose: Also known as Standard Score or Standardization, it is a normalization technique where the values are rescaled so that they have the
            properties of a standard normal distribution with μ = 0 and σ = 1, where μ is the mean and σ is the standard deviation.
        Usage: Useful in algorithms that assume the data is normally distributed, or in scenarios where the scale and distribution of the data might 
            affect the learning process.
        */
        public static double minMaxNormalize(double value, double min, double max)
        {
            return (value - min) / (max - min);
        }
        public static float minMaxNormalize(float value, float min, float max)
        {
            return (value - min) / (max - min);
        }

        // Z-Score Normalization
        /*
        Purpose: Also known as Standard Score or Standardization, it is a normalization technique where the values are rescaled so that they have the 
            properties of a standard normal distribution with μ = 0 and σ = 1, where μ is the mean and σ is the standard deviation.
        Usage: Useful in algorithms that assume the data is normally distributed, or in scenarios where the scale and distribution of the data might 
            affect the learning process.
        */
        public static double zScoreNormalize(double value, double mean, double stdDev)
        {
            return (value - mean) / stdDev;
        }

        public static float zScoreNormalize(float value, float mean, float stdDev)
        {
            return (value - mean) / stdDev;
        }
        // Softmax Function
        /*
        Purpose: A function that converts a vector of values into a probability distribution, where the probabilities are proportional to the exponentials of the input
            numbers. Often used in the final layer of a neural network-based classifier to represent the probabilities of the classes.
        Usage: Predominantly used in multi-class classification problems, especially in neural networks to output probabilities for each class in classification tasks.
        */
        public static double[] softmax(double[] values)
        {
            double[] softmaxValues = new double[values.Length];
            double sumExp = 0.0;

            foreach (double value in values)
            {
                sumExp += Math.Exp(value);
            }
            for (int i = 0; i < values.Length; i++)
            {
                softmaxValues[i] = Math.Exp(values[i]) / sumExp;
            }
            return softmaxValues;
        }





        // Define the custom weighted MSE loss function 
        public static double ZeroWeightedMSE(double ActualValue, double predicted)
        {
            // Define weights for different values
            double zeroWeight = 0.25;
            double nonZeroWeight = 1.0;

            // Determine the weight based on the true value
            double weight = (ActualValue == 0) ? zeroWeight : nonZeroWeight;

            // Calculate the weighted MSE for the single prediction
            return weight * Math.Pow(predicted - ActualValue, 2);
        }

        public static double mae(double ActualValue, double predicted)
        {
            return Math.Abs(ActualValue - predicted);
        }
        public static double categoricalCrossEntropy(double ActualValue, double predicted)
        {
            return -ActualValue * Math.Log(predicted + 1e-15); // Adding a small value to avoid log(0)
        }
        public static double huberLoss(double ActualValue, double predicted, float delta)
        {
            double diff = ActualValue - predicted;
            if (Math.Abs(diff) <= delta)
            {
                return 0.5 * diff * diff;
            }
            else
            {
                return delta * (Math.Abs(diff) - 0.5 * delta);
            }
        }

        public static class FirstDerivative
        {
            public static double mse(double ActualValue, double predicted)
            {
                return 2 * (predicted - ActualValue);
            }
            public static float mse(float ActualValue, float predicted)
            {
                return 2 * (predicted - ActualValue);
            }

            public static double ZeroWeightedMSEDerivative(double ActualValue, double predicted)
            {
                // Define weights for different values
                double zeroWeight = 0.25;
                double nonZeroWeight = 1.0;

                // Determine the weight based on the true value
                double weight = (ActualValue == 0) ? zeroWeight : nonZeroWeight;

                // Calculate the derivative of the weighted MSE for the single prediction
                return 2 * weight * (predicted - ActualValue);
            }        // Derivative of Binary Cross-Entropy
            public static double binaryCrossEntropy(double ActualValue, double predicted)
            {
                return (predicted - ActualValue) / (predicted * (1 - predicted));
            }
            public static float binaryCrossEntropy(float ActualValue, float predicted)
            {
                return (predicted - ActualValue) / (predicted * (1 - predicted));
            }
            public static double mae(double ActualValue, double predicted)
            {
                return ActualValue > predicted ? -1 : 1;
            }

            public static double categoricalCrossEntropy(double ActualValue, double predicted)
            {
                return predicted - ActualValue;
            }
            public static double huberLoss(double ActualValue, double predicted, float delta)
            {
                double diff = ActualValue - predicted;
                if (Math.Abs(diff) <= delta)
                {
                    return diff;
                }
                else
                {
                    return delta * Math.Sign(diff);
                }
            }
        }
    }
}
