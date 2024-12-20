namespace CallaghanDev.ML
{
    /// <summary>
    /// Custom exception for neural network training failures caused by various issues
    /// such as vanishing gradients, poor weight initialization, and others.
    /// </summary>
    public class TrainingFailureException : Exception
    {
        /// <summary>
        /// Gets the type of failure that caused the exception.
        /// </summary>
        public FailureType Failure { get; private set; }

        /// <summary>
        /// Detailed explanation of the failure.
        /// </summary>
        public string FailureDetails { get; private set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="TrainingFailureException"/> class.
        /// </summary>
        /// <param name="failure">The type of failure.</param>
        /// <param name="message">Additional details or context about the failure.</param>
        public TrainingFailureException(FailureType failure, string message = null): base(message ?? GetDefaultMessage(failure))
        {
            Failure = failure;
            FailureDetails = message ?? GetDefaultMessage(failure);
        }

        /// <summary>
        /// Generates a default message based on the failure type.
        /// </summary>
        private static string GetDefaultMessage(FailureType failure)
        {
            switch (failure)
            {
                case FailureType.VanishingGradient:
                    return "Vanishing Gradients: Gradients become very small, stopping learning in early layers. Use ReLU, Leaky ReLU, or gradient clipping.";
                case FailureType.ExplodingGradient:
                    return "Exploding Gradients: Gradients grow excessively, causing instability. Use gradient clipping or smaller learning rates.";
                case FailureType.ActivationSaturation:
                    return "Activation Saturation: Neurons stuck in saturated regions (e.g., sigmoid/tanh). Switch to ReLU or similar activation functions.";
                case FailureType.PoorWeightInitialization:
                    return "Poor Weight Initialization: Improper initialization causes learning issues. Use Xavier or He initialization.";
                case FailureType.DeadNeurons:
                    return "Dead Neurons: Neurons consistently output zero (ReLU issue). Switch to Leaky ReLU or PReLU.";
                case FailureType.ImproperLearningRate:
                    return "Improper Learning Rate: Ensure the learning rate is tuned or adaptive (e.g., Adam, RMSProp).";
                case FailureType.DataIssues:
                    return "Data Issues: Check for normalization, class imbalance, or noisy data.";
                case FailureType.Overfitting:
                    return "Overfitting: Regularize the model, increase data, or use dropout techniques.";
                case FailureType.PoorModelArchitecture:
                    return "Poor Model Architecture: Optimize layer depth/width or add skip connections.";
                case FailureType.BatchSizeIssues:
                    return "Batch Size Issues: Use an appropriate batch size for stability and generalization.";
                case FailureType.IncorrectLossFunction:
                    return "Incorrect Loss Function: Ensure the loss function matches the task type (e.g., classification vs regression).";
                case FailureType.NumericalInstability:
                    return "Numerical Instability: Use numerically stable implementations (e.g., log-softmax).";
                default:
                    return "Unknown failure occurred during training.";
            }
        }

        /// <summary>
        /// Provides a formatted error string.
        /// </summary>
        public override string ToString()
        {
            return $"[FailureType: {Failure}] - {Message}";
        }
    }

}
