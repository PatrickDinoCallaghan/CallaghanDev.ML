using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML.Neural_Network.Exceptions
{
    [Serializable]
    public class IncreasingErrorException : TrainingFailureExceptionBase
    {
        public float CurrentError { get; }

        public IncreasingErrorException(string message, float currentError = 0) : base(message)
        {
            CurrentError = currentError;
        }
    }
}
