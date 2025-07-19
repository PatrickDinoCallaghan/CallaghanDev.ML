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
        public static float mse(float ActualValue, float predicted)
        {
            return MathF.Pow(predicted - ActualValue, 2);
        }


        // Cross-Entropy Loss for binary classification
        /*
         Purpose: A loss function used for binary classification problems. It measures the performance of a classification model whose
               output is a probability value between 0 and 1. Cross-entropy loss increases as the predicted probability diverges from the actual label.
         Usage: Widely used in models that output probabilities, such as logistic regression and binary classification neural networks.
         */
        public static float binaryCrossEntropy(float value, float predicted)
        {
            return -(float)(value * MathF.Log(predicted) + (1 - value) * MathF.Log(1 - predicted));
        }

        // Min-Max Normalization
        /*
        Purpose: Also known as Standard Score or Standardization, it is a normalization technique where the values are rescaled so that they have the
            properties of a standard normal distribution with μ = 0 and σ = 1, where μ is the mean and σ is the standard deviation.
        Usage: Useful in algorithms that assume the data is normally distributed, or in scenarios where the scale and distribution of the data might 
            affect the learning process.
        */
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
        public static float[] softmax(float[] values)
        {
            float[] softmaxValues = new float[values.Length];
            float sumExp = 0.0f;

            foreach (float value in values)
            {
                sumExp += MathF.Exp(value);
            }
            for (int i = 0; i < values.Length; i++)
            {
                softmaxValues[i] = MathF.Exp(values[i]) / sumExp;
            }
            return softmaxValues;
        }

        // Define the custom weighted MSE loss function 
        public static float ZeroWeightedMSE(float ActualValue, float predicted)
        {
            // Define weights for different values
            float zeroWeight = 0.25f;
            float nonZeroWeight = 1.0f;

            // Determine the weight based on the true value
            float weight = (ActualValue == 0) ? zeroWeight : nonZeroWeight;

            // Calculate the weighted MSE for the single prediction
            return weight * MathF.Pow(predicted - ActualValue, 2);
        }

        public static float mae(float ActualValue, float predicted)
        {
            return MathF.Abs(ActualValue - predicted);
        }
        public static float categoricalCrossEntropy(float ActualValue, float predicted)
        {
            return -ActualValue * MathF.Log(predicted + 1e-15f); // Adding a small value to avoid log(0)
        }
        public static float huberLoss(float ActualValue, float predicted, float delta)
        {
            float diff = ActualValue - predicted;
            if (MathF.Abs(diff) <= delta)
            {
                return 0.5f * diff * diff;
            }
            else
            {
                return delta * (MathF.Abs(diff) - 0.5f * delta);
            }
        }

        public static class FirstDerivative
        {
            public static float mse(float ActualValue, float predicted)
            {
                return 2 * (predicted - ActualValue);
            }

            public static float ZeroWeightedMSEDerivative(float ActualValue, float predicted)
            {
                // Define weights for different values
                float zeroWeight = 0.25f;
                float nonZeroWeight = 1.0f;

                // Determine the weight based on the true value
                float weight = (ActualValue == 0) ? zeroWeight : nonZeroWeight;

                // Calculate the derivative of the weighted MSE for the single prediction
                return 2 * weight * (predicted - ActualValue);
            }        // Derivative of Binary Cross-Entropy
            public static float binaryCrossEntropy(float ActualValue, float predicted)
            {
                return (predicted - ActualValue) / (predicted * (1 - predicted));
            }

            public static float mae(float ActualValue, float predicted)
            {
                return ActualValue > predicted ? -1 : 1;
            }

            public static float categoricalCrossEntropy(float ActualValue, float predicted)
            {
                return predicted - ActualValue;
            }
            public static float huberLoss(float ActualValue, float predicted, float delta)
            {
                float diff = ActualValue - predicted;
                if (MathF.Abs(diff) <= delta)
                {
                    return diff;
                }
                else
                {
                    return delta * MathF.Sign(diff);
                }
            }

        }
    }
}
