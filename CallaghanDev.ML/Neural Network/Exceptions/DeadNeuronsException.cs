using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML.Neural_Network.Exceptions
{
    [Serializable]
    public class DeadNeuronsException : TrainingFailureExceptionBase
    {
        public int DeadNeuronCount { get; }

        public DeadNeuronsException(string message, int deadNeuronCount = 0) : base(message)
        {
            DeadNeuronCount = deadNeuronCount;
        }

    }

}
