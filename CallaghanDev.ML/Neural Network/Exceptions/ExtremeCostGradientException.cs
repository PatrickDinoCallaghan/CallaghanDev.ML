using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML.Neural_Network.Exceptions
{
    [Serializable]
    public class ExtremeCostGradientException : TrainingFailureExceptionBase
    {
        public float GradientNorm { get; }

        public ExtremeCostGradientException(string message, float gradientNorm = 0) : base(message)
        {
            GradientNorm = gradientNorm;
        }
    }

}
