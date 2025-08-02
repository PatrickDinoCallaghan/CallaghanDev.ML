using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML.Neural_Network.Exceptions
{
    [Serializable]
    public class StagnatingErrorException : TrainingFailureExceptionBase
    {
        public int Iterations { get; }

        public StagnatingErrorException(string message, int iterations = 0) : base(message)
        {
            Iterations = iterations;
        }
    }

}
