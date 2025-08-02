using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML.Neural_Network.Exceptions
{
    [Serializable]
    public class OverfittingException : TrainingFailureExceptionBase
    {
        public float TrainingError { get; }
        public float ValidationError { get; }

        public OverfittingException(string message, float trainingError = 0, float validationError = 0) : base(message)
        {
            TrainingError = trainingError;
            ValidationError = validationError;
        }
    }
}
