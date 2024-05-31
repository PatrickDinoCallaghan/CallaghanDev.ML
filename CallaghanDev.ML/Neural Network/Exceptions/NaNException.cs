using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML.Neural_Network.Exceptions
{
    public class NaNException : Exception
    {
        public NaNException() : base("NaN detected") { }

        public NaNException(string message) : base(message) { }

        public NaNException(string message, Exception inner) : base(message, inner) { }
    }

}
