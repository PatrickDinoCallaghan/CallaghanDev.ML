using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML.Neural_Network.Exceptions
{
    [Serializable]
    public class LowDirectionalAccuracyException : TrainingFailureExceptionBase
    {
        public float Accuracy { get; }

        public LowDirectionalAccuracyException(string message, float accuracy = 0) : base(message)
        {
            Accuracy = accuracy;
        }
    }
}
