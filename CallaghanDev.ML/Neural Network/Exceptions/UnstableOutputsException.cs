using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML.Neural_Network.Exceptions
{
    [Serializable]
    public class UnstableOutputsException : TrainingFailureExceptionBase
    {
        public float[] OffendingOutputs { get; }

        public UnstableOutputsException(string message, float[] outputs = null) : base(message)
        {
            OffendingOutputs = outputs ?? Array.Empty<float>();
        }
    }
}
