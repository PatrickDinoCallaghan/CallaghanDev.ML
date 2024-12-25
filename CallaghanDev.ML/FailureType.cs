using Microsoft.FSharp.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML
{

    public enum FailureType
    {
        /// <summary>
        /// Activation Explosion
        ///     - What Happens: Activation values become extremely large in one or more neurons, 
        ///       overshadowing other signals and causing potential numeric overflow.
        ///     - Cause: Poor weight initialization, large inputs, overly high learning rates, or 
        ///       lack of normalization can lead to excessively high activations.
        ///     - Impact: The network may produce NaN or Infinity values in subsequent layers, 
        ///       destabilizing or halting training.
        ///     - Solution:
        ///         * Use batch normalization or layer normalization to keep activations in a stable range
        ///         * Adjust or lower the learning rate
        ///         * Employ better weight initialization (e.g., He or Xavier)
        ///         * Perform gradient or activation clipping to limit extreme values
        /// </summary>
        ActivationExplosion,
        /// <summary>
        /// <summary>
        ///Vanishing Gradients,
        ///        - What Happens: Gradients become very small (close to zero) during backpropagation, especially in deep networks.
        ///        - Cause: Activation functions like Sigmoid or Tanh squash values to a small range. In deep layers, gradients vanish when multiplied during backpropagation.
        ///        - Impact: Early layers stop learning as their weights do not get updated.
        ///        - Solution: Use activation functions like ReLU, Leaky ReLU, or Gradient Clipping for RNNs.
        /// </summary>
        VanishingGradient,

        /// <summary>
        /// Exploding Gradients
        ///        - What Happens: Gradients grow exponentially during backpropagation, particularly in deep networks.
        ///        - Cause: Poor weight initialization or large learning rates can amplify gradients.
        ///        - Impact: Loss becomes NaN or extremely large, causing the model to fail.
        ///        - Solution:
        ///            * Gradient clipping
        ///            * Proper weight initialization (e.g., Xavier or He initialization)
        ///            * Smaller learning rates
        /// </summary>
        ExplodingGradient,

        /// <summary>
        /// Activation Saturation
        ///        - What Happens: Neurons get stuck in the saturated regions of activation functions like Sigmoid or Tanh.
        ///        - Cause: Large input values (e.g., weights or biases are too big).
        ///        - Impact: Gradients become close to zero, leading to vanishing gradients.
        ///        - Solution: Use activation functions like ReLU that do not saturate.
        /// </summary>
        ActivationSaturation,

        /// <summary>
        /// Poor Weight Initialization
        ///        - What Happens: Weights are initialized in a way that causes neurons to produce extremely large or small outputs.
        ///        - Cause: Random initialization without considering input size or activation function.
        ///        - Impact:
        ///        - Exploding/vanishing gradients
        ///        - Dead neurons in ReLU
        ///        - Solution:
        ///        - Use Xavier Initialization (for Tanh/Sigmoid) or He Initialization (for ReLU).
        /// </summary>
        PoorWeightInitialization,

        /// <summary>
        /// Dead Neurons
        ///        - What Happens: For ReLU activation, neurons can output zero for all inputs if they enter the negative gradient zone.
        ///        - Cause: Large negative weights/biases can lead neurons to always output zero.
        ///        - Impact: Neurons effectively stop learning.
        ///        - Solution: Use Leaky ReLU or PReLU to allow small gradients in the negative region.
        /// </summary>
        DeadNeurons,

        /// <summary>
        /// Improper Learning Rate
        ///        - What Happens:
        ///             * If the learning rate is too large, weights oscillate or diverge.
        ///             * If the learning rate is too small, training becomes very slow.
        ///        - Solution:
        ///             * Use an adaptive learning rate (e.g., Adam, RMSProp).
        ///             * Tune learning rates carefully.
        /// </summary>
        ImproperLearningRate,

        /// <summary>
        /// Data Issues
        ///        - Unnormalized Input Data:
        ///             * Inputs that are not scaled/normalized can cause instability in activations.
        ///        - Solution: Use techniques like MinMax scaling or Standardization (mean = 0, std = 1).
        ///        - Class Imbalance: If the data has imbalanced classes, the model may learn a biased solution.
        ///        - Noisy or Corrupted Data: Poor-quality data can mislead the learning process.
        /// </summary>
        DataIssues,

        /// <summary>
        /// Overfitting
        ///        - What Happens: The model memorizes training data but fails on unseen data.
        ///        - Cause: Small datasets, large models, or lack of regularization.
        ///        - Solution:
        ///             * Use techniques like Dropout, L2 Regularization, or Data Augmentation.
        ///             * Use more training data.
        /// </summary>
        Overfitting,

        /// <summary>
        /// Poor Model Architecture
        ///        - What Happens: A poorly designed neural network architecture can fail to learn effectively.
        ///        - Examples:
        ///             * Too few neurons/layers (underfitting).
        ///             * Too many neurons/layers with insufficient data (overfitting).
        ///             * No skip connections in very deep networks (gradient issues).
        ///        - Solution: Properly design network depth, width, and connections (e.g., skip connections in ResNets).
        /// </summary>
        PoorModelArchitecture,

        /// <summary>
        /// Batch Size Issues
        ///        - What Happens:
        ///             * Small batch sizes: High variance in gradients.
        ///             * Large batch sizes: Poor generalization and slow convergence.
        ///        - Solution: Use an appropriate batch size (e.g., 32–256 for many problems).
        /// </summary>
        BatchSizeIssues,

        /// <summary>
        /// Incorrect Loss Function
        ///        - What Happens: The loss function does not match the task (e.g., using MSE for classification).
        ///        - Impact: Poor or no convergence.
        ///        - Solution: Choose the appropriate loss function:
        ///             * Cross-Entropy Loss for classification
        ///             * Mean Squared Error for regression
        /// </summary>
        IncorrectLossFunction,

        /// <summary>
        /// Numerical Instability
        ///        - What Happens: Operations like exponentiation or division can cause numerical overflow/underflow.
        ///        - Examples:
        ///             * Softmax can result in large exponentiated values.
        ///        - Solution:
        ///             * Use numerically stable implementations like log-softmax.
        /// </summary>
        NumericalInstability

    }
}
    #region Unit tests template old
    /*
      public class NeuralNetworkFailureTests
      {
          // Placeholder for gradient computation method
          private List<double> ComputeGradients() => new List<double> { /* Gradient values here */


            // Placeholder for activation outputs per layer
           // private List<List<double>> GetLayerActivations() => new List<List<double>> { /* Activations here */ };

            // Placeholder for loss computation
           // private double ComputeLoss() => 0.0;

            // Placeholder for batch gradient variances
          //  private List<double> GetGradientVariances() => new List<double> { /* Variance values */ };

            // Placeholder for detecting numerical issues
           // private bool HasNumericalInstability(double value) => double.IsNaN(value) || double.IsInfinity(value);

    // <summary> Test for Dead Neurons: ReLU outputs zero constantly. </summary>

    /* [Test]
        public void TestDeadNeurons()
        {
            var activations = GetLayerActivations();
            foreach (var layer in activations)
            {
                int zeroCount = layer.Count(a => a == 0);
                double zeroPercentage = (double)zeroCount / layer.Count;
                Assert.Less(zeroPercentage, 0.9, "Dead Neurons Detected: Too many zero activations.");
            }
        }

        /// <summary> Test for Improper Learning Rate: Loss oscillation or slow decrease. </summary>
        [Test]
        public void TestImproperLearningRate()
        {
            var losses = new List<double> { /* Training loss values per epoch */
       
        /*         double lossDecrease = losses.First() - losses.Last();

                Assert.Greater(lossDecrease, 1e-2, "Improper Learning Rate Detected: Loss not decreasing sufficiently.");
            }

            /// <summary> Test for Data Issues: Loss plateau or instability due to unnormalized data. </summary>
            [Test]
            public void TestDataIssues()
            {
                double loss = ComputeLoss();
                Assert.Less(loss, 1e3, "Data Issues Detected: Loss plateau or instability.");
            }

            /// <summary> Test for Overfitting: Training loss diverges from validation loss. </summary>
            [Test]
            public void TestOverfitting()
            {
                var trainingLoss = new List<double> { /* Training loss values */ 
           
        /*     var validationLoss = new List<double> { /* Validation loss values */ 

      
        /*          double trainLossFinal = trainingLoss.Last();
                double valLossFinal = validationLoss.Last();

                Assert.Less(trainLossFinal, valLossFinal, "Overfitting Detected: Validation loss higher than training loss.");
            }

            /// <summary> Test for Batch Size Issues: Gradient variance across batches. </summary>
            [Test]
            public void TestBatchSizeIssues()
            {
                var variances = GetGradientVariances();
                double varianceRange = variances.Max() - variances.Min();

                Assert.Less(varianceRange, 1.0, "Batch Size Issues Detected: Gradient variance too high.");
            }

            /// <summary> Test for Numerical Instability: Check for NaN or Inf in loss/activations. </summary>
            [Test]
            public void TestNumericalInstability()
            {
                double loss = ComputeLoss();
                Assert.IsFalse(HasNumericalInstability(loss), "Numerical Instability Detected: Loss is NaN or Infinity.");
            }
        }*/

    #endregion