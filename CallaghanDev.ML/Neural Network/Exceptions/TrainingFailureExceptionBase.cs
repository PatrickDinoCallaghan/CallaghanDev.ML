using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML.Neural_Network.Exceptions
{
    public class TrainingFailureExceptionBase : Exception
    {
        public TrainingFailureExceptionBase() { }

        public TrainingFailureExceptionBase(string message) : base(message) { }

        public TrainingFailureExceptionBase(string message, Exception inner) : base(message, inner) { }

        protected TrainingFailureExceptionBase(System.Runtime.Serialization.SerializationInfo info, System.Runtime.Serialization.StreamingContext context) : base(info, context) { }
    }
}
